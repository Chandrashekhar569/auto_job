"""
Continuous job applying on the one board Boss picks: Naukri, Indeed or LinkedIn.

Conversation:
    Boss:      "Continuous job apply karo"
    Veronica:  "Kahan apply karun Boss - Naukri, Indeed ya LinkedIn?"
    Boss:      "LinkedIn"
    Veronica:  start_continuous_apply(portal="linkedin") -> the plan (needs_confirmation),
               read out in one line; Boss says yes -> confirm_continuous_apply(approve=true)

That single yes covers the run: it applies job after job by itself until the
run's limit, the end of the results, a stop (stop_job_applications), or
something only Boss can fix - a sign-in, the board's daily limit, another
browser task needing Edge - where it pauses (resume_continuous_apply). A
CAPTCHA is left on screen: the run waits (HUMAN_CHECK_WAIT_S, default 5 min)
while Boss completes it in the Edge window, then applies that same job; only
if it is still there after that does the run pause. A paused job is opened
again first on resume.

Rules of a run:
    - Only the board's own quick-apply flow: LinkedIn Easy Apply, Indeed Apply,
      Naukri's Apply (with its screening chatbot). Naukri jobs that apply on a
      company site go to the company-site queue as before, where every account
      and every Submit still waits for Boss (apply_saved_company_jobs).
    - Every job is scored against the CV first; weak matches are skipped.
    - Answers come only from the CV, candidate_profile.json and the answers Boss
      gave (screening_answers). A question they do not cover skips that job; the
      questions are read out at the end so Boss answers each once
      (teach_job_agent), and the skipped jobs are retried in the next run.
    - Human pace: a pause between applications, a cap per run (default 20,
      CONTINUOUS_APPLY_MAX) and per board per day (CONTINUOUS_APPLY_DAILY_CAP).
    - Jobs seen before are remembered in data/job_applications/board_seen.json,
      so a new run does not re-open them.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import re
import threading
import time
from datetime import datetime
from typing import Any, Optional

from langchain_core.tools import tool

from autojob.core.paths import DATA_DIR

logger = logging.getLogger(__name__)

SEEN_FILE = DATA_DIR / "job_applications" / "board_seen.json"
PORTALS = {"naukri": "Naukri", "indeed": "Indeed", "linkedin": "LinkedIn"}
QUICK_APPLY = {"naukri": "Naukri Apply", "indeed": "Indeed Apply (Easily apply)", "linkedin": "Easy Apply"}
DEFAULT_MAX = 20
MAX_PER_RUN = int(os.getenv("CONTINUOUS_APPLY_MAX", "50"))
DAILY_CAP = int(os.getenv("CONTINUOUS_APPLY_DAILY_CAP", "50"))
GAP_S = (6.0, 14.0)        # pause between two applications
MAX_FAILS_IN_ROW = 4
_UI_AGENT = "naukri"       # the dashboard's job-application agent node

_PORTAL_ALIASES = [
    ("linkedin", r"linked\s*-?\s*in|linkdin|linkden|लिंक्डइन"),
    ("indeed", r"\bin+de+d\b|इंडीड"),
    ("naukri", r"naukri|nokri|naukari|नौकरी"),
]


def normalize_portal(text: str) -> Optional[str]:
    """'LinkedIn', 'linked in', 'naukri.com', 'Indeed pe' -> 'linkedin' / 'naukri' / 'indeed'."""
    t = str(text or "").strip().lower()
    if t in PORTALS:
        return t
    for key, pattern in _PORTAL_ALIASES:
        if re.search(pattern, t, re.I):
            return key
    return None


def split_keywords(text: str) -> list[str]:
    """'AI engineer, data scientist or ML engineer' -> three searches."""
    parts = re.split(r",|;|/|\bor\b|\band\b|\baur\b|\bya\b", str(text or ""), flags=re.I)
    out = []
    for p in parts:
        p = " ".join(p.split()).strip(" .")
        if p and p.lower() not in (o.lower() for o in out):
            out.append(p)
    return out[:5]


def cv_keywords(profile: dict) -> list[str]:
    """Searches built from the profile: current title, then its job titles / search keywords."""
    found: list[str] = []
    if profile.get("current_title"):
        found.append(str(profile["current_title"]))
    for key in ("job_titles", "search_keywords"):
        value = profile.get(key)
        if isinstance(value, list):
            found += [str(v) for v in value if v]
    out: list[str] = []
    for k in found:
        k = " ".join(k.split())
        if k and k.lower() not in (o.lower() for o in out):
            out.append(k)
    return out[:3]


def _answers_mtime() -> float:
    try:
        from autojob.jobs.job_deep_agent import MEM_ROOT

        path = MEM_ROOT / "answers.md"
        return path.stat().st_mtime if path.exists() else 0.0
    except Exception:
        return 0.0


def _ui(*events: dict) -> None:
    try:
        from autojob.core.veronica_ui_bridge import emit

        for event in events:
            emit(event)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Jobs seen before (per board), so runs do not re-open them
# ---------------------------------------------------------------------------

class SeenJobs:
    RETRY_DAYS = {"failed": 3, "skipped": 30, "closed": 60, "external": 30}

    def __init__(self, path=SEEN_FILE):
        self.path = path
        self.lock = threading.Lock()

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        except Exception as exc:
            logger.warning("board_seen.json unreadable (%s); starting fresh", exc)
            return {}

    def get(self, portal: str, job_id: str) -> Optional[dict]:
        with self.lock:
            return self._load().get(portal, {}).get(str(job_id))

    def put(self, portal: str, job_id: str, status: str, title: str = "", company: str = "") -> None:
        with self.lock:
            data = self._load()
            data.setdefault(portal, {})[str(job_id)] = {
                "status": status, "ts": time.time(), "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "title": (title or "")[:120], "company": (company or "")[:80],
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")

    def skip_reason(self, portal: str, job_id: str, answers_mtime: float = 0.0) -> Optional[str]:
        """Why this job should not be opened again now, or None."""
        entry = self.get(portal, job_id)
        if not entry:
            return None
        status, ts = entry.get("status"), float(entry.get("ts") or 0)
        if status == "applied":
            return "applied before"
        if status == "unconfirmed":
            return "submitted before (not confirmed) - check it in the browser"
        if status == "needs_user":
            return None if answers_mtime > ts else "waiting for your answer to its question"
        days = self.RETRY_DAYS.get(status)
        if days is not None and time.time() - ts < days * 86400:
            return f"{status} recently"
        return None

    def applied_today(self, portal: str) -> int:
        today = datetime.now().strftime("%Y-%m-%d")
        with self.lock:
            entries = self._load().get(portal, {}).values()
        return sum(1 for e in entries if e.get("status") == "applied" and str(e.get("time", "")).startswith(today))


_seen = SeenJobs()


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

class _ContinuousRun:
    def __init__(self):
        self.state = "idle"   # idle | awaiting_approval | running | paused | finished | stopped
        self.plan: Optional[dict] = None
        self.stop_requested = False
        self._reset({})

    def _reset(self, plan: dict) -> None:
        self.portal = plan.get("portal", "")
        self.sources: list[dict] = plan.get("sources", [])
        self.max_apps = plan.get("max_applications", DEFAULT_MAX)
        self.dry_run = bool(plan.get("dry_run"))
        self.cursor = {"source": 0, "page": 0}
        self.seen_this_run: set[str] = set()
        self.applied: list[dict] = []
        self.needs_user: list[dict] = []
        self.failed: list[dict] = []
        self.skipped: list[dict] = []
        self.dry_runs: list[dict] = []
        self.questions: list[str] = []
        self.external_saved = 0
        self.external_skipped = 0
        self.fails_in_row = 0
        self.pause_reason = ""
        self.current = ""
        self.current_id = ""   # the job being handled; a pause leaves it to be retried on resume
        self.answerer = None
        self.started = time.time()

    @property
    def label(self) -> str:
        return PORTALS.get(self.portal, self.portal)

    # -- control ----------------------------------------------------------

    def _should_stop(self) -> bool:
        if self.stop_requested:
            return True
        if self.portal != "naukri":
            from autojob.integrations import browser_service as bs

            if bs._agent_cancel.is_set():  # a newer browser task wants Edge
                self.pause_reason = "browser_needed"
                return True
        return False

    def _gap(self) -> None:
        end = time.time() + random.uniform(*GAP_S)
        while time.time() < end and not self.stop_requested:
            time.sleep(0.5)

    # -- recording --------------------------------------------------------

    def _entry(self, job, title: str, company: str, **extra) -> dict:
        return {"title": title or job.title or "?", "company": company or job.company or "?", "url": job.url, **extra}

    def _log(self, status: str, e: dict, notes: str) -> None:
        if self.dry_run:
            return
        try:
            from autojob.jobs.job_apply_service import log_application

            log_application(e["company"], e["title"], e["url"], status, notes, source=self.portal)
        except Exception as exc:
            logger.warning("Could not log the application: %s", exc)

    def _progress(self) -> None:
        _ui({"type": "agent", "id": _UI_AGENT, "status": "active",
             "task": f"{self.label}: applied {len(self.applied)}/{self.max_apps}",
             "action": self.current[:80]})

    # -- one job ----------------------------------------------------------

    def _one(self, board, matcher, job) -> Optional[dict]:
        """Handle one listed job. Returns a final/paused response to stop the run, else None."""
        why = _seen.skip_reason(self.portal, job.job_id, _answers_mtime())
        if why:
            return None
        if job.applied:
            _seen.put(self.portal, job.job_id, "applied", job.title, job.company)
            return None
        if job.easy is False and self.portal != "naukri":
            self.external_skipped += 1
            _seen.put(self.portal, job.job_id, "external", job.title, job.company)
            return None

        self.current = " at ".join(p for p in (job.title, job.company) if p)
        self._progress()
        try:
            info = board.inspect(job)
        except Exception as exc:
            return self._failed(job, job.title, job.company, f"Could not open the job: {str(exc)[:150]}")
        if info.get("captcha"):  # still there after waiting on screen for Boss
            return self._pause("captcha")
        if info.get("login"):
            return self._pause("login")
        if info.get("limit"):
            return self._finish(f"{self.label}'s daily quick-apply limit is reached; try again tomorrow.")
        title, company = info.get("title") or job.title, info.get("company") or job.company
        if info.get("error"):
            return self._failed(job, title, company, str(info["error"])[:150])
        if info.get("applied"):
            _seen.put(self.portal, job.job_id, "applied", title, company)
            return None
        if info.get("closed"):
            _seen.put(self.portal, job.job_id, "closed", title, company)
            return None
        if not info.get("easy"):
            self.external_skipped += 1
            _seen.put(self.portal, job.job_id, "external", title, company)
            return None

        score = matcher.score_job(job_title=title, company=company, job_description=info.get("description") or "")
        match = int(score.get("score") or 0)
        if not score.get("should_apply"):
            self.skipped.append(self._entry(job, title, company, reason=f"weak match {match}: {score.get('reasoning', '')}"[:160]))
            _seen.put(self.portal, job.job_id, "skipped", title, company)
            return None

        self.current = f"{title} at {company}"
        self._progress()
        logger.info("[%s] applying: %s at %s (match %d)", self.portal, title, company, match)
        res = board.apply(job, self.answerer, dry_run=self.dry_run)
        status = res.get("status")
        e = self._entry(job, title, company, match=match)

        if status == "applied":
            self.fails_in_row = 0
            self.applied.append(e)
            _seen.put(self.portal, job.job_id, "applied", title, company)
            self._log("applied", e, f"{QUICK_APPLY[self.portal]}, match {match}")
            _ui({"type": "log", "label": f"Applied on {self.label}: {title} at {company}", "tone": "success"})
        elif status == "needs_user":
            self.fails_in_row = 0
            qs = [q for q in res.get("questions") or [] if q]
            for q in qs:
                if q not in self.questions:
                    self.questions.append(q)
            self.needs_user.append({**e, "questions": qs})
            _seen.put(self.portal, job.job_id, "needs_user", title, company)
            self._log("needs_user", e, res.get("reason") or "; ".join(qs))
        elif status == "external":
            url = res.get("external_url") or ""
            if url and not self.dry_run:
                from autojob.jobs.job_apply_service import _run as portal_run

                portal_run.enqueue({"url": url, "company": company, "title": title, "source": self.portal,
                                    "naukri_url": job.url, "match_score": match, "via_naukri": False})
                self.external_saved += 1
            else:
                self.external_skipped += 1
            _seen.put(self.portal, job.job_id, "external", title, company)
        elif status == "skipped":
            _seen.put(self.portal, job.job_id, "skipped", title, company)
        elif status == "unconfirmed":
            self.failed.append({**e, "reason": res.get("reason", "")})
            _seen.put(self.portal, job.job_id, "unconfirmed", title, company)
            self._log("needs_user", e, res.get("reason", ""))
        elif status == "dry_run":
            self.dry_runs.append({**e, "filled": res.get("filled", [])[:12]})
        elif status == "paused":
            return self._pause(res.get("reason") or "captcha")
        elif status == "limit":
            return self._finish(f"{self.label}'s daily quick-apply limit is reached; try again tomorrow.")
        elif status == "stopped":
            return None  # the loop sees the stop and reports
        else:
            return self._failed(job, title, company, res.get("reason") or "unknown error", match)

        self._gap()
        return None

    def _failed(self, job, title: str, company: str, reason: str, match: int = 0) -> Optional[dict]:
        e = self._entry(job, title, company, match=match, reason=reason[:200])
        self.failed.append(e)
        _seen.put(self.portal, job.job_id, "failed", title, company)
        self._log("failed", e, reason)
        self.fails_in_row += 1
        if self.fails_in_row >= MAX_FAILS_IN_ROW:
            return self._pause("too_many_failures")
        return None

    # -- the loop ---------------------------------------------------------

    def _loop(self, board, matcher) -> dict:
        from autojob.jobs.board_flows import NeedsBoss

        while self.cursor["source"] < len(self.sources):
            if self._should_stop():
                break
            src = self.sources[self.cursor["source"]]
            if self.cursor["page"] >= board.pages_per_source.get(src["kind"], 3):
                self.cursor = {"source": self.cursor["source"] + 1, "page": 0}
                continue
            try:
                jobs = board.fetch(src, self.cursor["page"])
            except NeedsBoss as nb:
                return self._pause(nb.reason)
            except Exception as exc:
                logger.warning("[%s] could not list jobs (%s): %s", self.portal, src, exc)
                jobs = []
            fresh = 0
            for job in jobs:
                if self._should_stop():
                    break
                if len(self.applied) >= self.max_apps:
                    return self._finish(f"Reached this run's limit of {self.max_apps} applications.")
                if not self.dry_run and _seen.applied_today(self.portal) >= DAILY_CAP:
                    return self._finish(f"Applied to {DAILY_CAP} jobs on {self.label} today - that is the daily cap "
                                        "that keeps the account safe. More tomorrow.")
                if job.job_id in self.seen_this_run:
                    continue
                self.seen_this_run.add(job.job_id)
                fresh += 1
                self.current_id = job.job_id
                out = self._one(board, matcher, job)
                self.current_id = ""
                if out is not None:
                    return out
            if self._should_stop():
                break
            self.cursor["page"] += 1
            if fresh == 0:
                self.cursor = {"source": self.cursor["source"] + 1, "page": 0}

        if self.pause_reason == "browser_needed" and not self.stop_requested:
            return self._pause("browser_needed")
        if self.stop_requested:
            self.state = "stopped"
            return self.report("stopped", "Stopped by Boss.")
        return self._finish("No more matching jobs on the list right now.")

    def work(self) -> dict:
        """Run (or continue) until done or paused. Caller holds nothing; this takes the job-run lock."""
        from autojob.integrations import browser_service as bs
        from autojob.jobs import job_apply_service as jas
        from autojob.jobs.board_flows import BOARDS, NaukriBoard
        from autojob.jobs.naukri_service import JobMatcher
        from autojob.jobs.screening_answers import BoardAnswerer

        if not jas._run.lock.acquire(blocking=False):
            return {"error": "Another job-application run is using the browser right now. Wait for it to finish "
                             "or call stop_job_applications first."}
        try:
            if jas._run.pending:
                return {"error": "A company-site application is waiting for Boss's yes or no (confirm_job_step) "
                                 "first."}
            try:
                profile = jas.load_profile()
            except Exception as exc:
                return {"error": f"Could not read the CV/profile: {exc}"}
            cv_text = jas._cv_text()
            if self.answerer is None:
                cv = jas._find_cv()
                self.answerer = BoardAnswerer(profile, cv_text, cv_file=str(cv) if cv else None)
            matcher = JobMatcher(cv_profile=profile, cv_raw_text=cv_text)

            self.state, self.pause_reason, self.stop_requested = "running", "", False
            _ui({"type": "agent", "id": _UI_AGENT, "status": "active",
                 "task": f"Continuous apply on {self.label}", "action": "starting"})

            if self.portal == "naukri":
                board = NaukriBoard(should_stop=self._should_stop)
                hold = contextlib.nullcontext()
            else:
                board = BOARDS[self.portal](should_stop=self._should_stop)
                hold = bs._exclusive_browser()  # other browser tools wait; a new browser task pauses this run
            with hold:
                try:
                    problem = board.open()
                    if problem:
                        if problem.get("reason") == "error":
                            self.state = "finished"
                            return {"error": f"Could not start on {self.label}: {problem.get('detail')}"}
                        return self._pause(problem["reason"], problem.get("detail", ""))
                    return self._loop(board, matcher)
                except Exception as exc:
                    logger.exception("Continuous apply run failed")
                    return self._pause("error", str(exc)[:200])
                finally:
                    try:
                        board.close()
                    except Exception as exc:
                        logger.warning("Could not close %s cleanly: %s", self.label, exc)
        finally:
            jas._run.lock.release()
            done = self.state in ("finished", "stopped")
            _ui({"type": "agent", "id": _UI_AGENT, "status": "complete" if done else "waiting",
                 "output": f"{self.label}: applied to {len(self.applied)} job(s)"})

    # -- outcomes ---------------------------------------------------------

    _PAUSE_TEXT = {
        "login": "{label} is not signed in in Veronica's Edge window. Boss signs in there once; then say 'continue'.",
        "captcha": "{label} showed a human-verification check (CAPTCHA) and it was not completed in the Edge window "
                   "in time. Boss completes it there, then says 'continue' - the same job is applied first. "
                   "Veronica never solves these herself.",
        "browser_needed": "Another browser task needed the Edge window, so applying on {label} paused. Say "
                          "'continue applying' when that is done.",
        "too_many_failures": "Several jobs in a row failed on {label}, so it paused - the site may have changed or "
                             "Edge is in a bad state. Have a look at the Edge window, then say 'continue'.",
        "error": "Applying on {label} hit an error: {detail}",
        "limit": "{label}'s daily limit is reached; try again tomorrow.",
    }

    def _pause(self, reason: str, detail: str = "") -> dict:
        self.state = "paused"
        self.pause_reason = reason
        if self.current_id:
            # Not finished (CAPTCHA, sign-in, an error): resume opens this same job again.
            self.seen_this_run.discard(self.current_id)
            self.current_id = ""
        note = self._PAUSE_TEXT.get(reason, "Applying on {label} paused: {detail}").format(
            label=self.label, detail=detail or reason)
        logger.info("[%s] paused: %s", self.portal, note)
        return self.report("paused", note)

    def _finish(self, why: str) -> dict:
        self.state = "finished"
        return self.report("finished", why)

    def report(self, status: Optional[str] = None, note: str = "") -> dict:
        status = status or self.state
        label = self.label
        applied = [f"{a['title']} at {a['company']}" for a in self.applied]
        parts = [f"{label}{' (DRY RUN - nothing submitted)' if self.dry_run else ''}: applied to {len(applied)} "
                 f"job(s) this run"]
        if self.needs_user:
            parts.append(f"{len(self.needs_user)} skipped because of questions your CV/profile do not answer")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        if self.skipped:
            parts.append(f"{len(self.skipped)} weak matches skipped")
        if self.external_saved:
            parts.append(f"{self.external_saved} company-site job(s) saved for apply_saved_company_jobs")
        if self.external_skipped:
            parts.append(f"{self.external_skipped} need the company's own site (not applied)")
        if self.dry_runs:
            parts.append(f"{len(self.dry_runs)} filled up to Submit in the dry run")
        summary = ", ".join(parts) + "." + (f" {note}" if note else "")
        out = {
            "status": status,
            "portal": label,
            "summary": summary,
            "applied_this_run": applied,
            "questions_for_boss": self.questions[:12],
            "skipped_for_questions": [f"{n['title']} at {n['company']}" for n in self.needs_user][:10],
            "failed": [f"{f['title']} at {f['company']}: {f.get('reason', '')[:120]}" for f in self.failed][:8],
        }
        if self.dry_runs:
            out["dry_run_filled"] = self.dry_runs[:5]
        if status == "paused":
            out["pause_reason"] = self.pause_reason
            out["instruction"] = ("Tell Boss in one or two sentences what is needed and how many were applied so far. "
                                  "When Boss says it is done / continue, call resume_continuous_apply.")
        else:
            out["instruction"] = ("Tell Boss how many jobs were applied to and the top few companies. Only jobs in "
                                  "applied_this_run were applied - never say others were. Then read out the "
                                  "questions_for_boss one at a time; save each answer Boss gives with "
                                  "teach_job_agent(kind='answer') so the next run fills it and retries those jobs.")
        return out


_cont = _ContinuousRun()
_tool_lock = threading.Lock()


def request_stop() -> bool:
    """Called by stop_job_applications. True if a continuous run was asked to stop."""
    if _cont.state in ("running",):
        _cont.stop_requested = True
        return True
    if _cont.state in ("paused", "awaiting_approval"):
        _cont.state = "stopped"
        _cont.plan = None
        return True
    return False


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def _build_plan(portal: str, keywords: str, location: str, max_applications: int, dry_run: bool) -> dict:
    from autojob.jobs.job_apply_service import load_profile

    profile = load_profile()
    given = split_keywords(keywords)
    searches = given or cv_keywords(profile)
    if not searches:
        try:
            from autojob.jobs.naukri_service import CVParser

            parsed = CVParser().parse()
            searches = cv_keywords({"job_titles": parsed.get("job_titles") or [],
                                    "search_keywords": parsed.get("search_keywords") or []})
        except Exception as exc:
            logger.warning("Could not derive searches from the CV: %s", exc)
    sources = [] if given else [{"kind": "recommended"}]
    from autojob.jobs.naukri_service import search_locations

    # Pan-India by default: each search runs in the main cities and all-India.
    locations = search_locations(location)
    sources += [{"kind": "search", "keywords": k, "location": loc} for k in searches for loc in locations]
    if not sources:
        sources = [{"kind": "recommended"}]
    return {
        "portal": portal,
        "sources": sources,
        "searches": searches,
        "location": ", ".join(loc or "All India" for loc in locations),
        "max_applications": max(1, min(int(max_applications or DEFAULT_MAX), MAX_PER_RUN)),
        "dry_run": bool(dry_run),
    }


def _plan_text(plan: dict) -> str:
    label = PORTALS[plan["portal"]]
    where = []
    if any(s["kind"] == "recommended" for s in plan["sources"]):
        where.append(f"the jobs {label} recommends for Boss")
    if plan["searches"]:
        where.append("searches for " + ", ".join(f"'{k}'" for k in plan["searches"])
                     + (f" in {plan['location']}" if plan["location"] else ""))
    text = (f"Apply continuously on {label} ({QUICK_APPLY[plan['portal']]} only), up to {plan['max_applications']} "
            f"applications: {' then '.join(where)}. Each job is matched against the CV first; answers come only from "
            "the CV, profile and Boss's earlier answers, and a job asking something they do not cover is skipped "
            "and its question saved for Boss.")
    if plan["portal"] == "naukri":
        text += " Naukri jobs that apply on a company site are saved for later, where every submit still needs a yes."
    else:
        text += " Jobs that apply only on the company's own site are not applied in this run."
    if plan["dry_run"]:
        text += " DRY RUN: it fills everything but never clicks Submit."
    return text


@tool("start_continuous_apply")
def start_continuous_apply(
    portal: str = "", keywords: str = "", location: str = "", max_applications: int = DEFAULT_MAX,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Start applying to jobs continuously on ONE job board Boss picks: 'naukri', 'indeed' or 'linkedin'. If Boss has not said which board, never guess: ask "Kahan apply karun Boss - Naukri, Indeed ya LinkedIn?" (without a portal this returns needs_portal). Pass keywords / location / max_applications only if Boss said them; otherwise it uses the jobs that board recommends for Boss, then searches built from the CV. Returns needs_confirmation with the plan: tell Boss the plan in one line and wait for a clear yes, then call confirm_continuous_apply(approve=true). That one yes covers the whole run. dry_run=true only when Boss asks for a test run (fills everything, never submits).
    """
    key = normalize_portal(portal)
    if key is None:
        return {"status": "needs_portal", "options": list(PORTALS.values()),
                "summary": "Ask Boss where to apply: Naukri, Indeed or LinkedIn. Do not pick one yourself."}
    with _tool_lock:
        if _cont.state == "running":
            return {"error": f"Already applying on {_cont.label}. Use continuous_apply_status for progress or "
                             "stop_job_applications to stop it."}
        try:
            plan = _build_plan(key, keywords, location, max_applications, dry_run)
        except Exception as exc:
            return {"error": f"Could not read the CV/profile: {exc}"}
        _cont.plan = plan
        _cont.state = "awaiting_approval"
        text = _plan_text(plan)
        return {"status": "needs_confirmation", "portal": PORTALS[key], "plan": text,
                "summary": (f"{text} Read this plan to Boss in one short line and wait for a clear yes; then call "
                            "confirm_continuous_apply(approve=true). No means approve=false.")}


@tool("confirm_continuous_apply")
def confirm_continuous_apply(approve: bool) -> dict[str, Any]:
    """
    Boss's answer to the continuous-apply plan from start_continuous_apply. approve=true only after Boss clearly said yes to that plan: it then applies job after job by itself in the background and reports when it finishes or pauses (sign-in, CAPTCHA, daily limit). approve=false cancels the plan.
    """
    with _tool_lock:
        plan = _cont.plan
        if not plan or _cont.state != "awaiting_approval":
            return {"error": "There is no continuous-apply plan waiting for approval. Call start_continuous_apply first."}
        _cont.plan = None
        if not approve:
            _cont.state = "idle"
            return {"status": "cancelled", "summary": "Okay - not applying."}
        _cont._reset(plan)
        _cont.stop_requested = False
        _cont.state = "running"
    logger.info("Continuous apply approved: %s", json.dumps({k: v for k, v in plan.items() if k != "sources"}))
    return _cont.work()


@tool("resume_continuous_apply")
def resume_continuous_apply() -> dict[str, Any]:
    """
    Continue a paused continuous-apply run after Boss fixed what it needed (signed in, finished the human check, the other browser task is done). Same plan and same approval; it continues where it stopped.
    """
    with _tool_lock:
        if _cont.state != "paused":
            return {"error": f"Nothing is paused (state: {_cont.state}). To start a new run use start_continuous_apply."}
        _cont.state = "running"
        _cont.stop_requested = False
        _cont.fails_in_row = 0
    return _cont.work()


@tool("continuous_apply_status")
def continuous_apply_status() -> dict[str, Any]:
    """How the continuous-apply run is going: applied so far, the job being applied now, skipped jobs, questions waiting for Boss, or why it is paused."""
    if _cont.state == "idle":
        return {"status": "idle", "summary": "No continuous-apply run right now."}
    rep = _cont.report(_cont.state)
    rep["current_job"] = _cont.current if _cont.state == "running" else ""
    rep["max_applications"] = _cont.max_apps
    if _cont.state == "awaiting_approval" and _cont.plan:
        rep["summary"] = "Waiting for Boss's yes to: " + _plan_text(_cont.plan)
    rep.pop("instruction", None)
    return rep


CONTINUOUS_APPLY_LANGCHAIN_TOOLS = [
    start_continuous_apply,
    confirm_continuous_apply,
    resume_continuous_apply,
    continuous_apply_status,
]

# Long-running: main.py runs these in the background and reports with a System Notification.
BACKGROUND_TOOLS = ("confirm_continuous_apply", "resume_continuous_apply")


def get_continuous_apply_tools() -> list[dict]:
    from autojob.core.langchain_bridge import get_gemini_tools_from_langchain

    return get_gemini_tools_from_langchain(CONTINUOUS_APPLY_LANGCHAIN_TOOLS)


def execute_continuous_apply_function(name: str, args: dict) -> dict[str, Any]:
    for t in CONTINUOUS_APPLY_LANGCHAIN_TOOLS:
        if t.name == name:
            try:
                return t.invoke(args)
            except Exception as exc:
                logger.exception("%s failed", name)
                return {"error": f"Invalid arguments or execution failed for {name}: {exc}"}
    return {"error": f"Unknown continuous apply function: {name}"}
