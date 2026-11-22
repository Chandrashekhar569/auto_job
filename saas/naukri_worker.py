"""
Per-user Naukri automation for the SaaS app.

Reuses the existing Naukri bot (search, job details, Apply + chatbot screening),
ATS matcher and screening-answer filler, but gives every user:
  - their own browser profile (data/saas/users/<id>/autojob_browser_u<id>), so logins never mix,
  - their own resume, skill, saved answers and application history (from the database),
  - no terminal prompts: questions the resume cannot answer are stored for the user to
    answer in the dashboard, and the job is skipped this time.

Three jobs:
  login()            sign in to Naukri with the user's stored credentials
  refresh_profile()  the once-a-day small profile edit (headline toggle, or resume re-upload)
  run_apply()        search with the skill's keywords, score each job, apply to the good fits
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Optional

# Headless SaaS runs: a CAPTCHA wait cannot be solved by anyone, keep it short.
os.environ.setdefault("HUMAN_CHECK_WAIT_S", "15")

from playwright.sync_api import sync_playwright  # noqa: E402
from sqlalchemy import select  # noqa: E402

from autojob.jobs import naukri_service as ns  # noqa: E402
from saas.config import BROWSER_CHANNEL, HARD_DAILY_LIMIT, HEADLESS, browser_profile_dir  # noqa: E402
from saas.db import (Application, NaukriAccount, Question, Resume, Settings, applied_today,  # noqa: E402
                     get_or_create, session_scope)
from saas.security import decrypt  # noqa: E402

logger = logging.getLogger("saas.naukri")

LOGIN_URL = "https://www.naukri.com/nlogin/login"
PROFILE_URL = "https://www.naukri.com/mnjuser/profile"


class NeedsAttention(RuntimeError):
    """Naukri wants something only the user can give (OTP, CAPTCHA, wrong password)."""


# ---------------------------------------------------------------------------
# Screening answers from this user's data only
# ---------------------------------------------------------------------------

class UserFormFiller(ns.ScreeningFormFiller):
    def __init__(self, cv_profile: dict, cv_raw_text: str, answers_text: str, about_text: str):
        super().__init__(cv_profile=cv_profile, cv_raw_text=cv_raw_text)
        self._answers_text = answers_text
        self._about_text = about_text
        self.new_questions: list[tuple[str, list[str]]] = []

    def _user_answers(self) -> str:  # type: ignore[override]
        return self._answers_text

    def skill_years(self, question: str) -> Optional[str]:
        from autojob.jobs.skill_years import answer_skill_years
        try:
            return answer_skill_years(question, self._about_text,
                                      self.cv_raw_text + " " + json.dumps(self._profile_facts()),
                                      self._answers_text)
        except Exception:
            return None

    def _remember_unanswered(self, question: str, options: list[str]) -> None:
        if question and question not in self.unanswered:
            self.unanswered.append(question)
            self.new_questions.append((question, list(options or [])))


# ---------------------------------------------------------------------------
# Browser bot on the user's own profile
# ---------------------------------------------------------------------------

class UserNaukriBot(ns.NaukriJobBot):
    def __init__(self, user_id: int):
        super().__init__()
        self.user_id = user_id
        self.profile_dir = browser_profile_dir(user_id)

    def start(self, headless: Optional[bool] = None):  # type: ignore[override]
        if self._is_running:
            return
        self.status = "starting"
        self._should_stop = False
        self.playwright = sync_playwright().start()
        kwargs = dict(
            headless=HEADLESS if headless is None else headless,
            viewport={"width": 1366, "height": 768},
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/128.0.0.0 Safari/537.36 Edg/128.0.0.0"),
            args=["--disable-blink-features=AutomationControlled", "--no-first-run",
                  "--no-default-browser-check"],
        )
        if BROWSER_CHANNEL:
            kwargs["channel"] = BROWSER_CHANNEL
        try:
            self.context = self.playwright.chromium.launch_persistent_context(str(self.profile_dir), **kwargs)
        except Exception:
            self.playwright.stop()
            self.playwright = None
            self.status = "error"
            raise
        self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
        self._is_running = True
        self.status = "running"

    def stop(self):  # type: ignore[override]
        self._is_running = False
        try:
            if self.context:
                self.context.close()
            if self.playwright:
                self.playwright.stop()
        except Exception as exc:
            logger.warning("Error closing browser for user %s: %s", self.user_id, exc)
        self.context = self.playwright = self.page = None
        self.status = "idle"

    # -- login -----------------------------------------------------------

    def login(self, email: str, password: str) -> None:
        if self.verify_login():
            return
        if not email or not password:
            raise NeedsAttention("Naukri session expired. Reconnect your Naukri account.")
        page = self.page
        page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=30000)
        time.sleep(random.uniform(2, 3.5))
        user_box = page.locator("#usernameField, input[placeholder*='Email' i], input[type='email']").first
        pass_box = page.locator("#passwordField, input[type='password']").first
        user_box.fill(email)
        time.sleep(random.uniform(0.5, 1.2))
        pass_box.fill(password)
        time.sleep(random.uniform(0.5, 1.2))
        page.locator("button[type='submit'], button:has-text('Login')").first.click()
        time.sleep(6)
        body = (page.content() or "").lower()
        if re.search(r"\botp\b|one time password|verify your (mobile|email)", body) and "nlogin" in page.url:
            raise NeedsAttention("Naukri asked for an OTP. Use 'Sign in with a browser window' once to finish it.")
        if any(page.locator(s).first.is_visible() for s in ("iframe[src*='recaptcha']", "div[class*='captcha']")
               if page.locator(s).count()):
            raise NeedsAttention("Naukri showed a CAPTCHA. Use 'Sign in with a browser window' once to pass it.")
        if "invalid" in body and ("password" in body or "credentials" in body) and "nlogin" in page.url:
            raise NeedsAttention("Naukri rejected the email/password.")
        if not self.verify_login():
            raise NeedsAttention("Could not confirm the Naukri login. Try 'Sign in with a browser window'.")

    def wait_for_manual_login(self, timeout_s: int = 600, should_stop: Callable[[], bool] = lambda: False) -> bool:
        """Visible window: the user signs in themself (OTP / CAPTCHA). Polls until logged in."""
        self.page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=30000)
        try:
            self.page.bring_to_front()
        except Exception:
            pass
        deadline = time.time() + timeout_s
        while time.time() < deadline and not should_stop():
            time.sleep(4)
            try:
                if self.has_session_cookies():
                    time.sleep(3)   # let Naukri finish writing the session
                    return True
            except Exception:
                return False
        return False

    def has_session_cookies(self) -> bool:
        """Naukri sets these only for a signed-in user, whatever tab or page the user ended on."""
        names = {c["name"]: c.get("value", "") for c in self.context.cookies("https://www.naukri.com")}
        return bool(names.get("nauk_at")) or names.get("is_login") in ("1", "true")

    # -- daily profile refresh ------------------------------------------------

    def refresh_profile(self, resume_path: Optional[Path]) -> str:
        """
        A small real edit so Naukri marks the profile 'updated today' (recruiters see recent
        profiles first). Toggles a trailing full stop on the resume headline; if that widget
        cannot be found, re-uploads the same resume.
        """
        page = self.page
        page.goto(PROFILE_URL, wait_until="domcontentloaded", timeout=30000)
        time.sleep(random.uniform(4, 6))
        note = self._toggle_headline()
        if note:
            return note
        if resume_path and resume_path.exists():
            page.goto(PROFILE_URL, wait_until="domcontentloaded", timeout=30000)
            time.sleep(4)
            inp = page.locator("input#attachCV, input[type='file']").first
            if inp.count():
                inp.set_input_files(str(resume_path))
                time.sleep(8)
                return f"Re-uploaded resume ({resume_path.name})"
        raise RuntimeError("Could not find the resume headline or resume upload on the Naukri profile page.")

    def _toggle_headline(self) -> str:
        page = self.page
        edit_selectors = [
            ".resumeHeadline .edit", "div.resumeHeadline span.edit",
            "div[class*='resumeHeadline'] span[class*='edit']",
            "xpath=//span[contains(normalize-space(.),'Resume headline')]/following::span[contains(@class,'edit')][1]",
        ]
        for sel in edit_selectors:
            try:
                btn = page.locator(sel).first
                if btn.count() and btn.is_visible(timeout=2000):
                    btn.click()
                    break
            except Exception:
                continue
        else:
            return ""
        time.sleep(2)
        box = page.locator("#resumeHeadlineTxt, form textarea").first
        try:
            box.wait_for(state="visible", timeout=6000)
        except Exception:
            return ""
        old = box.input_value() or ""
        new = old.rstrip()[:-1].rstrip() if old.rstrip().endswith(".") else old.rstrip() + "."
        if not new.strip():
            return ""
        box.fill(new)
        time.sleep(1)
        for sel in ("button:has-text('Save')", "button[type='submit']"):
            b = page.locator(sel).last
            if b.count():
                b.click()
                time.sleep(3)
                return "Resume headline updated (" + ("removed" if old.rstrip().endswith(".") else "added") + \
                       " trailing full stop)"
        return ""


# ---------------------------------------------------------------------------
# Jobs run by the run manager (runs.py)
# ---------------------------------------------------------------------------

def _account(user_id: int) -> tuple[str, str]:
    with session_scope() as s:
        acc = get_or_create(s, NaukriAccount, user_id)
        return acc.email, decrypt(acc.password_enc)


def _set_account(user_id: int, **fields) -> None:
    with session_scope() as s:
        acc = get_or_create(s, NaukriAccount, user_id)
        for k, v in fields.items():
            setattr(acc, k, v)


def _signed_in_bot(user_id: int, headless: Optional[bool] = None) -> UserNaukriBot:
    email, password = _account(user_id)
    bot = UserNaukriBot(user_id)
    bot.start(headless=headless)
    try:
        bot.login(email, password)
    except NeedsAttention as exc:
        bot.stop()
        _set_account(user_id, status="needs_attention", status_message=str(exc))
        raise
    except Exception:
        bot.stop()
        raise
    _set_account(user_id, status="connected", status_message="Signed in", last_login_at=datetime.utcnow())
    return bot


def job_login(user_id: int, ctx) -> str:
    bot = _signed_in_bot(user_id)
    bot.stop()
    return "Naukri account connected."


def job_manual_login(user_id: int, ctx) -> str:
    bot = UserNaukriBot(user_id)
    bot.start(headless=False)
    try:
        ok = bot.wait_for_manual_login(should_stop=ctx.should_stop)
    finally:
        bot.stop()
    if not ok:
        _set_account(user_id, status="needs_attention", status_message="Browser sign-in was not completed.")
        raise NeedsAttention("Browser sign-in was not completed in time.")
    _set_account(user_id, status="connected", status_message="Signed in with browser window",
                 last_login_at=datetime.utcnow())
    return "Naukri account connected (browser sign-in)."


def job_profile_refresh(user_id: int, ctx, bot: Optional[UserNaukriBot] = None) -> str:
    with session_scope() as s:
        r = s.scalar(select(Resume).where(Resume.user_id == user_id))
        resume_path = Path(r.path) if r else None
    own = bot is None
    bot = bot or _signed_in_bot(user_id)
    try:
        note = bot.refresh_profile(resume_path)
    finally:
        if own:
            bot.stop()
    _set_account(user_id, last_profile_refresh=date.today(), last_profile_refresh_note=note)
    return note


def _load_user_data(user_id: int) -> dict:
    with session_scope() as s:
        r = s.scalar(select(Resume).where(Resume.user_id == user_id))
        if r is None or r.status != "ready":
            raise RuntimeError("Upload your resume first (and wait until its skill is ready).")
        st = get_or_create(s, Settings, user_id)
        acc = get_or_create(s, NaukriAccount, user_id)
        answers = s.scalars(select(Question).where(Question.user_id == user_id, Question.answer != "")).all()
        asked = {q.question.strip().lower() for q in s.scalars(select(Question).where(Question.user_id == user_id))}
        seen = set(s.scalars(select(Application.url).where(Application.user_id == user_id,
                                                           Application.status.in_(("applied", "external",
                                                                                   "skipped")))).all())
        return {
            "profile": r.profile, "skill": r.skill, "raw_text": r.raw_text, "resume_path": r.path,
            "settings": {c: getattr(st, c) for c in ("daily_limit", "min_match", "keywords", "locations",
                                                     "daily_profile_refresh")},
            "refreshed_today": acc.last_profile_refresh == date.today(),
            "answers_text": "\n".join(f"- {q.question} -> {q.answer}" for q in answers),
            "asked": asked, "seen": seen, "applied_today": applied_today(s, user_id),
        }


def _record(user_id: int, run_id: int, job: dict, score: int, status: str, message: str) -> None:
    with session_scope() as s:
        s.add(Application(user_id=user_id, run_id=run_id, title=job.get("title", "")[:500],
                          company=job.get("company", "")[:500], url=job.get("url", "")[:2000],
                          match_score=score, status=status, message=message[:2000]))


def _save_questions(user_id: int, filler: UserFormFiller, asked: set) -> None:
    with session_scope() as s:
        for q, opts in filler.new_questions:
            if q.strip().lower() in asked:
                continue
            asked.add(q.strip().lower())
            s.add(Question(user_id=user_id, question=q, options_json=json.dumps(opts, ensure_ascii=False)))
    filler.new_questions.clear()


def _split(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"[,;\n]+", text or "") if p.strip()]


def job_apply(user_id: int, ctx, dry_run: bool = False) -> str:
    d = _load_user_data(user_id)
    skill, profile, st = d["skill"], dict(d["profile"]), d["settings"]
    limit = min(int(st["daily_limit"] or 0), HARD_DAILY_LIMIT)
    remaining = limit - d["applied_today"]
    if remaining <= 0 and not dry_run:
        return f"Daily limit reached ({limit} applications today)."

    # The skill sharpens the parsed CV: its titles/skills drive matching.
    profile["skills"] = list(dict.fromkeys((skill.get("core_skills") or []) + (profile.get("skills") or [])
                                           + (skill.get("secondary_skills") or [])))
    profile["job_titles"] = skill.get("target_titles") or profile.get("job_titles") or []
    if skill.get("experience_years") not in (None, ""):
        profile["experience_years"] = skill["experience_years"]
    for k, v in (skill.get("standard_answers") or {}).items():
        if v:
            profile[k] = v

    keywords = _split(st["keywords"]) or skill.get("search_keywords") or profile.get("search_keywords") or []
    keywords = keywords[:3] or ["software developer"]
    locations = ns.search_locations(st["locations"] or ", ".join(skill.get("preferred_locations") or []))
    avoid = [a.lower() for a in skill.get("avoid_keywords") or [] if a]
    min_match = int(st["min_match"] or 60)

    about = json.dumps(skill, ensure_ascii=False, indent=1)
    matcher = ns.JobMatcher(cv_profile=profile, cv_raw_text=d["raw_text"])
    filler = UserFormFiller(profile, about + "\n\n" + d["raw_text"], d["answers_text"], about)

    bot = _signed_in_bot(user_id)
    ctx.bot = bot
    applied = skipped = errors = 0
    try:
        if not dry_run and st["daily_profile_refresh"] and not d["refreshed_today"]:
            try:
                ctx.log("Daily profile refresh: " + job_profile_refresh(user_id, ctx, bot=bot))
            except Exception as exc:
                ctx.log(f"Profile refresh failed: {exc}")

        bot.status = "searching"
        exp = profile.get("experience_years")
        lists = []
        for kw in keywords:
            for loc in locations:
                if ctx.should_stop():
                    break
                lists.append(bot.search_jobs(keywords=kw, location=loc, experience=exp))
        jobs = ns.interleave_jobs(lists)
        seen_now = set()
        jobs = [j for j in jobs if j.get("url") and j["url"] not in d["seen"]
                and not (j["url"] in seen_now or seen_now.add(j["url"]))]
        ctx.log(f"Found {len(jobs)} new jobs for {', '.join(keywords)} in "
                f"{', '.join(loc or 'All India' for loc in locations)}")

        bot.status = "applying"
        for job in jobs:
            if ctx.should_stop() or applied >= (limit if dry_run else remaining):
                break
            title_l = (job.get("title") or "").lower()
            if any(re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", title_l) for a in avoid):
                _record(user_id, ctx.run_id, job, 0, "skipped", "Title matches an avoid keyword")
                skipped += 1
                continue
            details = bot.get_job_details(job["url"])
            if details.get("error"):
                _record(user_id, ctx.run_id, job, 0, "error", details["error"])
                errors += 1
                continue
            job = {**job, "title": details.get("title") or job.get("title", ""),
                   "company": details.get("company") or job.get("company", "")}
            sc = matcher.score_job(job["title"], job["company"],
                                   details.get("description") or job.get("description_snippet", ""))
            score = int(sc.get("score") or 0)
            if score < min_match:
                _record(user_id, ctx.run_id, job, score, "skipped", f"Low match: {sc.get('reasoning', '')}")
                skipped += 1
                continue
            if dry_run:
                res = {"status": "dry_run", "message": "Dry run: would apply"}
            else:
                res = bot.apply_to_job(filler)
            _save_questions(user_id, filler, d["asked"])
            status = res.get("status", "error")
            msg = res.get("message", "")
            if status == "external" and res.get("external_url"):
                msg = f"{msg} {res['external_url']}".strip()
            _record(user_id, ctx.run_id, job, score, status, msg)
            if status in ("applied", "dry_run"):
                applied += 1
                ctx.applied = applied
                ctx.log(f"{'Would apply' if dry_run else 'Applied'}: {job['title']} at {job['company']} ({score})")
            elif status in ("error", "unconfirmed"):
                errors += 1
            else:
                skipped += 1
            ns._human_delay(ns.LONG_DELAY)
    finally:
        ctx.bot = None
        bot.stop()
    if dry_run:
        return f"Dry run: would apply to {applied}, skipped {skipped}, errors {errors}. Nothing was submitted."
    return f"Applied {applied}, skipped {skipped}, errors {errors}."
