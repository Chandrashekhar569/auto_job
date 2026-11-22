"""
Job applications on company career portals, with the user in the loop.

Flow (auto_apply_jobs):
    1. Naukri: search with the CV profile, apply to Naukri-hosted jobs, and
       collect jobs whose Apply button goes to a company career portal.
    2. Career portals: for each collected job, a temporary worker agent (the
       vision browser agent from browser_service, in its own protected tab)
       opens the portal and fills everything — account sign-up/sign-in, email
       verification codes, resume upload, screening questions — using the CV
       and the editable candidate profile. It starts from main memory: general
       lessons plus the playbooks for this hiring platform and this site.
    3. Debrief: when a worker finishes (applied or not), its step log is turned
       into techniques and merged into those playbooks
       (job_deep_agent.debrief), so the next worker is faster and avoids the
       same traps. The next worker waits for that debrief before starting.

The user stays in control:
    - Every account creation and every final Submit/Apply click pauses the run
      with status 'needs_confirmation'. Nothing is created or submitted until
      the user says yes (confirm_job_step(approve=True)).
    - CAPTCHAs, payment requests, SSO-only sign-in and questions the profile
      cannot answer honestly are handed back to the user ('needs_user') and the
      run moves on to the next job.
    - Portal passwords are generated per site and kept in Windows Credential
      Manager (service "Veronica-JobPortals"), never in plain text files.

Everything is logged under data/job_applications/:
    applications_log.txt   one readable line per job attempt
    applications.json      the same, machine-readable (used for the report)
    portal_accounts.txt    which portals have an account, with which email
    candidate_profile.json details used to fill forms — edit it to correct them
"""

import base64
import html
import json
import logging
import re
import secrets
import string
import threading
import time
from datetime import datetime
from pathlib import Path
from autojob.core.paths import PROJECT_ROOT, CV_DIR
from typing import Any, Optional
from urllib.parse import urlparse

from langchain_core.tools import tool

from autojob.integrations import browser_service as bs

logger = logging.getLogger(__name__)

_PROJECT_DIR = PROJECT_ROOT
DATA_DIR = _PROJECT_DIR / "data" / "job_applications"
PROFILE_FILE = DATA_DIR / "candidate_profile.json"
LOG_TXT = DATA_DIR / "applications_log.txt"
LOG_JSON = DATA_DIR / "applications.json"
ACCOUNTS_TXT = DATA_DIR / "portal_accounts.txt"
QUEUE_FILE = DATA_DIR / "portal_queue.json"  # company-site jobs waiting to be applied

KEYRING_SERVICE = "Veronica-JobPortals"
JOB_MAX_STEPS = 40  # multi-page portals (Workday, SuccessFactors) plus sign-up need more than 25
EMAIL_WAIT_S = 120
EMAIL_POLL_S = 8

_file_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Candidate profile
# ---------------------------------------------------------------------------

_PROFILE_DEFAULTS = {
    "name": "",
    "email": "",
    "phone": "",
    "location": "",
    "address": "",
    "linkedin": "",
    "experience_years": None,
    "current_title": "",
    "current_company": "",
    "current_ctc": "",
    "expected_ctc": "",
    "notice_period": "",
    "education": "",
    "skills": [],
    "work_authorization": "",
    "willing_to_relocate": "",
    "summary": "",
    "_note": (
        "Veronica fills application forms from this file and your CV. Correct anything wrong and "
        "fill empty fields; empty fields are never guessed - the job is handed back to you instead. "
        "'email' must be the Gmail account Veronica can read, so she can enter verification codes."
    ),
}


def _find_cv() -> Optional[Path]:
    for pattern in ("*.pdf", "*.PDF", "*.docx", "*.DOCX"):
        files = sorted(CV_DIR.glob(pattern))
        if files:
            return files[0]
    return None


def load_profile() -> dict:
    """Load the editable profile, creating it from the CV the first time (and refreshing it when the CV changes)."""
    from autojob.core.cv_sync import sync_if_cv_changed

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sync_if_cv_changed()
    if PROFILE_FILE.exists():
        try:
            return {**_PROFILE_DEFAULTS, **json.loads(PROFILE_FILE.read_text(encoding="utf-8"))}
        except Exception as exc:
            logger.warning("candidate_profile.json unreadable (%s); rebuilding from CV", exc)

    from autojob.jobs.naukri_service import CVParser

    parsed = CVParser().parse()
    if "error" in parsed:
        raise RuntimeError(parsed["error"])
    # CVParser's regex fallback uses placeholders ("Candidate", 0 years); never
    # let those reach an application form — leave them blank instead.
    if parsed.get("name") == "Candidate":
        parsed["name"] = ""
        if not parsed.get("experience_years"):
            parsed["experience_years"] = None
    profile = dict(_PROFILE_DEFAULTS)
    for key in profile:
        if parsed.get(key) not in (None, "", []):
            profile[key] = parsed[key]
    if not profile["current_title"] and parsed.get("job_titles"):
        profile["current_title"] = parsed["job_titles"][0]
    PROFILE_FILE.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Created %s from the CV", PROFILE_FILE)
    return profile


def _cv_text() -> str:
    """Candidate facts for the agent: ABOUT_ME.md when present, else the CV's raw text."""
    try:
        from autojob.core.user_profile import about_me_text

        about = about_me_text()
        if about.strip():
            return about
    except Exception as exc:
        logger.warning("Could not read ABOUT_ME.md: %s", exc)
    cv = _find_cv()
    if not cv or cv.suffix.lower() != ".pdf":
        return ""
    try:
        import pymupdf

        with pymupdf.open(str(cv)) as doc:
            return "\n".join(page.get_text("text") for page in doc)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Logs and report
# ---------------------------------------------------------------------------

def _load_log() -> list[dict]:
    if LOG_JSON.exists():
        try:
            return json.loads(LOG_JSON.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def log_application(
    company: str, title: str, url: str, status: str, notes: str = "", source: str = "portal"
) -> None:
    """Append one job attempt to both the readable and the JSON log."""
    entry = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "status": status,
        "company": company or "?",
        "title": title or "?",
        "site": urlparse(url).hostname or "" if url else "",
        "url": url,
        "source": source,
        "notes": (notes or "")[:300],
    }
    with _file_lock:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        log = _load_log()
        log.append(entry)
        LOG_JSON.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
        new_file = not LOG_TXT.exists()
        with LOG_TXT.open("a", encoding="utf-8") as fh:
            if new_file:
                fh.write("time             | status             | company | job title | site | notes\n")
            fh.write(
                f"{entry['time']} | {status.upper():<18} | {entry['company']} | {entry['title']} | "
                f"{entry['site']} | {entry['notes'].replace(chr(10), ' ')}\n"
            )


def _already_applied(url: str) -> bool:
    return any(e.get("url") == url and e.get("status") == "applied" for e in _load_log())


# ---------------------------------------------------------------------------
# Portal accounts (passwords live in Windows Credential Manager)
# ---------------------------------------------------------------------------

def _site_key(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _generate_password() -> str:
    # Satisfies the usual portal rules: length, upper, lower, digit, symbol.
    alphabet = string.ascii_letters + string.digits + "!@#$%&*?"
    while True:
        pw = "".join(secrets.choice(alphabet) for _ in range(16))
        if (any(c.islower() for c in pw) and any(c.isupper() for c in pw)
                and any(c.isdigit() for c in pw) and any(c in "!@#$%&*?" for c in pw)):
            return pw


def _get_password(site: str) -> Optional[str]:
    import keyring

    return keyring.get_password(KEYRING_SERVICE, site)


def _create_password(site: str, email: str) -> str:
    import keyring

    pw = _generate_password()
    keyring.set_password(KEYRING_SERVICE, site, pw)
    with _file_lock:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        new_file = not ACCOUNTS_TXT.exists()
        with ACCOUNTS_TXT.open("a", encoding="utf-8") as fh:
            if new_file:
                fh.write(
                    "Career portal accounts created by Veronica.\n"
                    f"Passwords are in Windows Credential Manager > Windows Credentials > Generic, "
                    f"entries named '{KEYRING_SERVICE}'.\n\n"
                    "created          | site | login email\n"
                )
            fh.write(f"{datetime.now():%Y-%m-%d %H:%M} | {site} | {email}\n")
    return pw


def _has_account(site: str) -> bool:
    try:
        return bool(_get_password(site))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Gmail: verification codes and links
# ---------------------------------------------------------------------------

def _decode_part(data: str) -> str:
    try:
        return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8", "replace")
    except Exception:
        return ""


def _message_content(msg: dict) -> tuple[str, list[str]]:
    """Plain text and hrefs of a Gmail API message (format='full')."""
    texts, links = [], []

    def walk(part):
        mime = part.get("mimeType", "")
        data = (part.get("body") or {}).get("data")
        if data and mime in ("text/plain", "text/html"):
            raw = _decode_part(data)
            if mime == "text/html":
                links.extend(html.unescape(h) for h in re.findall(r'href=["\']([^"\']+)', raw, re.I))
                raw = re.sub(r"<(script|style).*?</\1>", " ", raw, flags=re.S | re.I)
                raw = re.sub(r"<[^>]+>", " ", raw)
            texts.append(html.unescape(raw))
        for sub in part.get("parts", []) or []:
            walk(sub)

    walk(msg.get("payload", {}))
    text = re.sub(r"[ \t\r\f\v]+", " ", "\n".join(texts))
    links.extend(re.findall(r"https?://[^\s<>\"')]+", text))
    return text, links


def _related(site: str, sender: str, subject: str, company: str) -> bool:
    """Does an email plausibly come from this portal/company?"""
    site_labels = {p for p in re.split(r"[.\-]", site) if len(p) > 3} - {"jobs", "careers", "career", "apply", "myworkdayjobs"}
    haystack = f"{sender} {subject}".lower()
    if any(label in haystack for label in site_labels):
        return True
    company_words = [w for w in re.split(r"\W+", (company or "").lower()) if len(w) > 3]
    return any(w in haystack for w in company_words)


def _recent_portal_emails(since: float, site: str, company: str) -> list[dict]:
    from autojob.integrations.gmail_service import get_gmail_service

    service = get_gmail_service()
    if not service:
        raise RuntimeError("Gmail is not connected (run authenticate_gmail.py).")
    resp = service.users().messages().list(
        userId="me", q=f"after:{int(since) - 60}", maxResults=15
    ).execute()
    out = []
    for m in resp.get("messages", []):
        msg = service.users().messages().get(userId="me", id=m["id"], format="full").execute()
        if int(msg.get("internalDate", 0)) / 1000 < since - 60:
            continue
        headers = {h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])}
        sender, subject = headers.get("from", ""), headers.get("subject", "")
        if not _related(site, sender, subject, company):
            continue
        text, links = _message_content(msg)
        out.append({"from": sender, "subject": subject, "text": text, "links": links,
                    "time": int(msg.get("internalDate", 0))})
    out.sort(key=lambda e: e["time"], reverse=True)
    return out


_CODE_CONTEXT = re.compile(r"(otp|code|verification|verify|one[- ]time|passcode|pin)", re.I)


def _extract_code(text: str) -> Optional[str]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        if _CODE_CONTEXT.search(ln):
            window = " ".join(lines[i:i + 3])
            m = re.search(r"\b(\d{4,8})\b", window) or re.search(r"\b(?=[A-Z0-9]*\d)([A-Z0-9]{6,8})\b", window)
            if m:
                return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Extra agent actions for career portals
# ---------------------------------------------------------------------------

class _JobContext:
    def __init__(self, job: dict, profile: dict):
        self.job = job
        self.profile = profile
        self.started = time.time()
        self.site = _site_key(job["url"])
        # Codes and links are only fetched for a portal account the user has
        # approved: one created earlier, or one they approve during this job.
        self.email_access_ok = _has_account(self.site)
        # The worker's whole step log, across pauses for the user, for the debrief.
        self.steps: list[str] = []


def _type_into(sess, ref: int, value: str) -> None:
    def _do():
        el = sess._find(ref)
        el.scroll_into_view_if_needed(timeout=5000)
        el.fill("")
        el.type(value, delay=25)
    sess.submit(_do)


def _saved_answers() -> str:
    try:
        from autojob.jobs.job_deep_agent import MEM_ROOT

        return (MEM_ROOT / "answers.md").read_text(encoding="utf-8")[:4000]
    except Exception:
        return ""


def _make_handlers(ctx: _JobContext) -> dict:
    def type_password(sess, act, done):
        try:
            ref = int(act.get("ref"))
        except (TypeError, ValueError):
            return {"error": "type_password needs a numeric 'ref'."}
        purpose = str(act.get("purpose", "login")).lower()
        site = _site_key(sess.submit(lambda: sess._page.url)) or ctx.site
        pw = _get_password(site)
        if not pw:
            if purpose != "signup":
                # Not a dead end: most portals let a new candidate register.
                return {"error": f"There is no saved account for {site}. Do not sign in: open the portal's "
                                 "'Sign up' / 'Create account' / 'Register' form and fill it, using "
                                 "type_password with purpose 'signup' for the password fields. Only if the "
                                 "portal has no way to register, reply NEEDS_USER."}
            pw = _create_password(site, ctx.profile.get("email", ""))
            log_application(ctx.job.get("company"), ctx.job.get("title"), ctx.job["url"],
                            "account_started", f"password generated for {site}")
        _type_into(sess, ref, pw)
        done.append({"do": "type_password", "ref": ref, "purpose": purpose})
        return None

    def upload_resume(sess, act, done):
        cv = _find_cv()
        if not cv:
            return {"error": "No resume found in the CV folder."}
        return bs._run_actions(sess, [{"do": "upload", "ref": act.get("ref"), "path": str(cv)}], False, done)

    def _wait_for_email(kind: str):
        if not ctx.email_access_ok:
            return None, {
                "error": f"Reading the {kind} from email is only allowed after the user approved the "
                         "account creation or sign-in step. Click that button first."
            }
        deadline = time.time() + EMAIL_WAIT_S
        while time.time() < deadline:
            try:
                mails = _recent_portal_emails(ctx.started, ctx.site, ctx.job.get("company", ""))
            except Exception as exc:
                return None, {"status": "needs_user", "summary": f"Could not read Gmail: {exc}"}
            if mails:
                return mails, None
            time.sleep(EMAIL_POLL_S)
        return None, {"status": "needs_user",
                      "summary": f"No email from {ctx.site} arrived within {EMAIL_WAIT_S}s."}

    def fill_email_code(sess, act, done):
        try:
            ref = int(act.get("ref"))
        except (TypeError, ValueError):
            return {"error": "fill_email_code needs a numeric 'ref'."}
        deadline = time.time() + EMAIL_WAIT_S
        while True:
            mails, stop = _wait_for_email("verification code")
            if stop:
                return stop
            for mail in mails:
                code = _extract_code(mail["text"])
                if code:
                    _type_into(sess, ref, code)
                    done.append({"do": "fill_email_code", "ref": ref, "from": mail["from"][:60]})
                    return None
            if time.time() > deadline:
                return {"status": "needs_user", "summary": f"Email from {ctx.site} had no readable code."}
            time.sleep(EMAIL_POLL_S)

    def open_email_verification_link(sess, act, done):
        mails, stop = _wait_for_email("verification link")
        if stop:
            return stop
        site_root = ".".join(ctx.site.split(".")[-2:])
        for mail in mails:
            for link in mail["links"]:
                host = (urlparse(link).hostname or "").lower()
                if not host or not re.search(r"verif|activat|confirm|validat", link, re.I):
                    continue
                sender_domain = mail["from"].split("@")[-1].strip("> ").lower()
                sender_root = ".".join(sender_domain.split(".")[-2:])
                if host.endswith(site_root) or (sender_root and host.endswith(sender_root)):
                    return bs._run_actions(sess, [{"do": "goto", "url": link}], False, done)
        return {"status": "needs_user", "summary": f"No verification link from {ctx.site} found in email."}

    def fill_form(sess, act, done):
        # One-shot fill of every answerable field on the page (see form_filler).
        # Never clicks anything: Next/Submit stay with the agent and its rails.
        from autojob.jobs.form_filler import fill_form_page

        profile = {k: v for k, v in ctx.profile.items() if not k.startswith("_")}
        facts = (
            f"CANDIDATE PROFILE:\n{json.dumps(profile, ensure_ascii=False)}\n\n"
            f"ABOUT THE CANDIDATE:\n{_cv_text()[:9000]}\n\n"
            f"ANSWERS THE USER GAVE:\n{_saved_answers()}"
        )
        try:
            res = fill_form_page(sess, facts)
        except Exception as exc:
            logger.exception("fill_form failed")
            return {"error": f"fill_form failed: {str(exc)[:200]}"}
        done.append({"do": "fill_form", "fields_seen": res["fields_seen"], "filled": res["filled"][:25],
                     "failed": res["failed"][:8], "needs_answer": res["unanswered_required"][:8]})
        if res["unanswered_required"]:
            try:
                from autojob.jobs.job_deep_agent import note_unanswered

                for q in res["unanswered_required"][:8]:
                    note_unanswered(q)
            except Exception:
                pass
            return {"status": "needs_user",
                    "summary": "Required question(s) the CV/profile do not answer, nothing was guessed: "
                               + "; ".join(res["unanswered_required"][:8])}
        return None

    def close_tab(sess, act, done):
        # Closing the tab being applied in lost the application on company
        # sites; only leftover tabs may be closed.
        try:
            index = int(act.get("index", -1))
        except (TypeError, ValueError):
            return {"error": "close_tab needs a numeric 'index'."}
        tabs = sess.submit(sess._tabs)
        target = tabs[index] if -len(tabs) <= index < len(tabs) else None
        if target is None:
            return {"error": f"No tab with index {index}."}
        if target["active"]:
            return {"error": "That is the tab you are applying in - keep it open and continue there."}
        return bs._run_actions(sess, [{"do": "close_tab", "index": index}], False, done)

    return {
        "fill_form": fill_form,
        "close_tab": close_tab,
        "type_password": type_password,
        "upload_resume": upload_resume,
        "fill_email_code": fill_email_code,
        "open_email_verification_link": open_email_verification_link,
    }


_JOB_SYSTEM_PROMPT = bs._AGENT_SYSTEM_PROMPT + """

YOU ARE NOW APPLYING TO A JOB on a company career portal for the candidate described in CANDIDATE PROFILE.
Goal: get the application fully filled in and submitted, then confirm the portal shows it was received.

Extra actions available for this job:
  {"do": "type_password", "ref": 5, "purpose": "login" | "signup"}   types this portal's password (never type one yourself)
  {"do": "upload_resume", "ref": 9}             uploads the candidate's CV file into a file input
  {"do": "fill_email_code", "ref": 7}           waits for the portal's email and types the verification code / OTP
  {"do": "open_email_verification_link"}        waits for the portal's email and opens its verify/activate link
  {"do": "fill_form"}                           PREFERRED for application forms: reads every field on the page (and
                                                embedded frames) and fills all the ones the profile answers, in one step.
                                                It clicks nothing. Use it on each form page, upload the resume, then check
                                                the screenshot, fix anything left with normal actions, and click Next/Submit.
                                                If it reports a question it cannot answer, reply NEEDS_USER with it.

Account handling:
- ACCOUNT STATUS tells you whether a saved account exists for this portal.
- Saved account: sign in with the profile email and {"do":"type_password","purpose":"login"}.
- No account: create one with the profile email; use {"do":"type_password","purpose":"signup"} for BOTH the
  password and confirm-password fields. Then click the create/register button.
- Do NOT use "Sign in with Google / LinkedIn / Facebook" buttons. If the portal only offers those, reply NEEDS_USER.
- Email verification (code or link) comes after the account step; use the email actions for it.

Filling the application:
- Look for "Apply", "Apply now", "I'm interested" to start. Prefer "autofill from resume" when offered, then fix fields.
- Answer ONLY from the CANDIDATE PROFILE, ABOUT THE CANDIDATE and ANSWERS THE USER GAVE BEFORE. Never invent or estimate
  experience, skills, project counts, degrees, salary or dates. Type only the answer value itself - never an
  explanation or your reasoning.
- Required question you cannot answer truthfully from them -> NEEDS_USER, quoting the question.
- Voluntary diversity / EEO questions: choose "Prefer not to say" / "Decline to answer" when available.
- Multi-page forms: fill the page, click Next/Continue, repeat.
- The account-creation click and the final Submit click will PAUSE for the user's approval. That is expected;
  just include that click as the last action when everything before it is filled.
- After submitting, reply DONE only when the page confirms the application was received; put the confirmation
  text in "answer". If the job is closed or no longer accepting applications, reply DONE with answer "JOB CLOSED: ...".

Tabs and page structure:
- Apply buttons often open the portal in a NEW TAB; you are switched to it automatically. Keep working in the tab
  that has the job/application. Never close that tab, and never close a tab just because it is new.
- If the page you land on is a careers home / search page instead of this job, search for the JOB title there
  and open it. If you land on a login wall, look for "Apply" / "Apply without account" / "Continue as guest" first.
- Some forms live inside an embedded frame: their fields are in INTERACTIVE ELEMENTS like any other (refs of 1000+)
  and their text is marked [inside embedded frame] in VISIBLE TEXT. Use them normally.
- After each click that should change the page, check the screenshot actually changed; if nothing happened, scroll
  the button into view, close any overlay, or click by x/y.

How the common career-portal systems work (check the URL to know which one you are on):
- Workday (myworkdayjobs.com): "Apply" -> choose "Apply Manually" (or "Autofill with Resume" then fix fields).
  Needs an account: "Create Account" (email, password, verify password, consent checkbox). Pages: My Information,
  My Experience (upload resume here), Application Questions, Voluntary Disclosures, Self Identify, Review ->
  Submit. Use "Save and Continue" at the bottom of every page. Dropdowns are custom: use the select action.
- Greenhouse (greenhouse.io, or a form embedded in the company site): one page, no account. Fill fields, attach
  resume ("Attach"), answer questions, "Submit application".
- Lever (jobs.lever.co): "Apply for this job" -> one page, resume upload first (it autofills), then Submit.
- SmartRecruiters / Ashby: "I'm interested" / "Apply" -> one page with resume upload; no account needed.
- iCIMS (icims.com): form inside an embedded frame; asks for email first, then may create an account.
- SuccessFactors / Oracle / Taleo: "Apply" -> sign in or create account, then several steps with "Next".
- Darwinbox / Keka / Zoho Recruit / Freshteam (common in India): usually "Apply" -> a single form with resume upload,
  sometimes an email OTP (use fill_email_code).
- Plain company sites: may use a simple contact-style form or say "send your CV to <email>". An email-only
  application is NEEDS_USER (say the address) - do not send mail yourself.

NOTES FROM PAST ATTEMPTS, when present, come from Veronica's memory of this site/platform. Use them to go
straight to what worked before and avoid what failed, but check the screenshot: sites change.

Stop with NEEDS_USER (and say why) if: a CAPTCHA or "are you a robot" check appears; the site asks for any
payment or fee (likely a scam); it needs a document other than the CV; or you are stuck after trying twice.
"""


# Clicks that only OPEN an application or a sign-up form. They used to pause
# for the user like a real submit, so runs stalled before anything was filled.
_START_CLICK_RE = re.compile(
    r"^(apply|apply now|easy apply|quick apply|apply online|apply here|apply today"
    r"|apply (for|to) (this |the )?(job|position|role|opening|vacancy)"
    r"|apply on (the )?company('s)? (site|website)|i'?m interested|i am interested"
    r"|start (your |my )?application|begin (your )?application|view (and|&) apply)\W*$",
    re.I,
)
_ACCOUNT_CLICK_RE = re.compile(r"sign ?up|register|create( an| my| your)? account", re.I)
# One-click job boards: there "Apply" itself sends the application.
_ONE_CLICK_BOARDS = ("naukri.com", "linkedin.com", "indeed.com", "foundit.in", "instahyre.com", "glassdoor")

# Counts fields the agent has filled on the page (and its frames): text,
# selects and chosen files; passwords separately.
_JS_FORM_STATE = r"""() => {
  let filled = 0, pw = 0;
  for (const el of document.querySelectorAll('input, textarea, select')) {
    const t = (el.type || '').toLowerCase();
    if (['hidden', 'submit', 'button', 'reset', 'image', 'search', 'checkbox', 'radio'].includes(t)) continue;
    if (/search/i.test((el.name || '') + ' ' + (el.placeholder || '') + ' ' + (el.id || ''))) continue;
    const r = el.getBoundingClientRect();
    if (!r.width || !r.height) continue;
    const has = t === 'file' ? !!(el.files && el.files.length) : !!String(el.value || '').trim();
    if (!has || el.tagName === 'SELECT' && el.selectedIndex <= 0) continue;
    filled++;
    if (t === 'password') pw++;
  }
  return {filled, pw};
}"""


def _form_state(sess) -> dict:
    def _read():
        total = {"filled": 0, "pw": 0, "url": sess._page.url}
        for frame in sess._page.frames:
            try:
                st = frame.evaluate(_JS_FORM_STATE) or {}
            except Exception:
                continue
            total["filled"] += st.get("filled", 0)
            total["pw"] += st.get("pw", 0)
        return total
    return sess.submit(_read)


def _job_click_approver(sess):
    """
    confirm= callable for the job agent's clicks. Approves only clicks that
    open a form: "Apply now" on a job page with nothing filled in yet, or a
    "Sign up / Create account" link before any password was typed. The real
    account creation and the final Submit still pause for the user.
    """
    def approve(label: str) -> bool:
        text = " ".join(str(label or "").split())
        if not text:
            return False
        state = _form_state(sess)
        host = (urlparse(state["url"]).hostname or "").lower()
        if _ACCOUNT_CLICK_RE.search(text):
            ok = state["pw"] == 0
        elif _START_CLICK_RE.match(text):
            one_click = any(b in host for b in _ONE_CLICK_BOARDS) and "company" not in text.lower()
            ok = state["filled"] == 0 and not one_click
        else:
            ok = False
        if ok:
            logger.info("Auto-approved '%s' on %s (it only opens a form)", text[:60], host)
        return ok
    return approve


# What a portal shows once an application really went through.
_CONFIRMED_RE = re.compile(
    r"thank(s| you)\b.{0,80}\b(appl|interest|submi)"
    r"|application\s+(has\s+been\s+|was\s+|is\s+)?(successfully\s+)?(received|submitted|sent|complete)"
    r"|(successfully|already)\s+(applied|submitted)|applied\s+successfully|you('ve| have)\s+(already\s+)?applied"
    r"|(we('ve| have)|have been)\s+received\s+your\s+application|submission\s+(received|successful)",
    re.I | re.S,
)


def _job_context_text(ctx: _JobContext, cv_text: str) -> str:
    profile = {k: v for k, v in ctx.profile.items() if not k.startswith("_")}
    return (
        f"JOB: {ctx.job.get('title', '?')} at {ctx.job.get('company', '?')}\n"
        f"ACCOUNT STATUS: {'saved account exists for ' + ctx.site if _has_account(ctx.site) else 'no saved account for ' + ctx.site}\n"
        f"CANDIDATE PROFILE (trusted, from the user):\n{json.dumps(profile, ensure_ascii=False)}\n\n"
        # Trimmed: this goes with EVERY step, and ~9000 chars made each call
        # ~6k tokens (slow, and over Groq's 7k tokens/minute). fill_form gets
        # the full CV; use it for the long application forms.
        f"ABOUT THE CANDIDATE (trusted, from the user; the only source for facts - UNKNOWN means ask; "
        f"fill_form knows the full CV):\n{cv_text[:3000]}"
    )


# ---------------------------------------------------------------------------
# Run state machine
# ---------------------------------------------------------------------------

def _job_id(url: str) -> str:
    import hashlib

    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]


def _load_queue() -> list[dict]:
    if QUEUE_FILE.exists():
        try:
            return json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("portal_queue.json unreadable: %s", exc)
    return []


class _ApplyRun:
    """
    Company-site applications. Jobs wait in a queue saved to portal_queue.json,
    so ones skipped during a Naukri run (or left over when a run is stopped)
    are applied later instead of being lost. The deep agent in job_deep_agent
    works through the queue and learns from each attempt; if it cannot run,
    the plain one-by-one loop is used instead.
    """

    def __init__(self):
        self.queue: list[dict] = _load_queue()
        self.pending: Optional[dict] = None
        self.ctx: Optional[_JobContext] = None
        self.stop_requested = False
        self.results: list[dict] = []
        self.lock = threading.Lock()
        self.profile: dict = {}
        self.cv_text = ""
        self.agent_thread: Optional[str] = None
        self.agent_notes = ""
        self.last_outcome: Optional[dict] = None
        self._job_lock = threading.Lock()
        self.debriefs: list[threading.Thread] = []
        self.lessons: list[str] = []  # what this run's workers taught main memory

    # -- queue ------------------------------------------------------------

    def save_queue(self) -> None:
        with _file_lock:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            QUEUE_FILE.write_text(json.dumps(self.queue, indent=2, ensure_ascii=False), encoding="utf-8")

    def enqueue(self, job: dict, front: bool = False) -> None:
        job = {**job, "id": job.get("id") or _job_id(job["url"])}
        if any(j["id"] == job["id"] for j in self.queue) or _already_applied(job["url"]):
            return
        if front:
            self.queue.insert(0, job)
        else:
            self.queue.append(job)
        self.save_queue()

    def _take(self, job_id: str) -> Optional[dict]:
        for i, j in enumerate(self.queue):
            if j["id"] == job_id:
                job = self.queue.pop(i)
                self.save_queue()
                return job
        return None

    # -- one job ----------------------------------------------------------

    def _finish_job(self, result: dict) -> None:
        job = self.ctx.job
        status = result.get("status")
        answer = str(result.get("answer") or result.get("summary") or result.get("error") or "")
        if status == "done":
            final = "job_closed" if answer.upper().startswith("JOB CLOSED") else "applied"
        elif status == "needs_user":
            final = "needs_user"
        elif status == "stopped":
            final = "stopped"
        else:
            final = "failed"
        log_application(job.get("company"), job.get("title"), job["url"], final, answer, job.get("source", "portal"))
        self.results.append({"company": job.get("company"), "title": job.get("title"),
                             "status": final, "notes": answer[:200]})
        steps = list(self.ctx.steps) or [str(s) for s in (result.get("steps") or [])]
        self.last_outcome = {
            "job_id": job.get("id"), "title": job.get("title"), "company": job.get("company"),
            "site": self.ctx.site, "status": final, "notes": answer[:400],
            "last_browser_steps": [s[:300] for s in steps[-8:]],
        }
        try:
            from autojob.jobs.job_deep_agent import record_attempt

            record_attempt(job, final, answer, steps)
        except Exception as exc:
            logger.warning("Could not write job memory: %s", exc)
        self._start_debrief(job, final, answer, steps)
        if final == "stopped":
            # Not attempted to the end: keep it for the next run.
            self.enqueue(job, front=True)
        # The job's tab is closed only once the application is confirmed; a
        # failed or unfinished one stays open so the user can complete it.
        sess = bs._get_session()
        if sess.is_running:
            try:
                sess.submit(lambda: sess.end_job(close_tabs=final in ("applied", "job_closed")))
            except Exception as exc:
                logger.warning("Could not tidy the job tab: %s", exc)
        self.ctx = None

    # -- debrief: the finished worker reports its techniques to main memory --

    def _start_debrief(self, job: dict, final: str, answer: str, steps: list[str]) -> None:
        """Runs in the background so the next job starts right away."""
        if not steps:
            return
        from autojob.jobs.job_deep_agent import debrief

        profile = dict(self.profile)

        def _work():
            try:
                learned = debrief(job, final, answer, steps, profile)
                if learned:
                    self.lessons.append(f"{_site_key(job['url'])}: {learned}")
            except Exception as exc:
                logger.warning("Debrief failed for %s: %s", job.get("url"), exc)

        t = threading.Thread(target=_work, name="job-debrief", daemon=True)
        self.debriefs.append(t)
        t.start()

    def wait_for_debriefs(self, timeout_s: float = 45) -> None:
        """Let pending debriefs land in memory before the next worker reads it."""
        deadline = time.time() + timeout_s
        for t in list(self.debriefs):
            t.join(max(0.0, deadline - time.time()))
        self.debriefs = [t for t in self.debriefs if t.is_alive()]

    def _handle(self, result: dict) -> Optional[dict]:
        """Returns a response for Veronica if the run must pause, else None."""
        if result.get("status") == "needs_confirmation":
            job = self.ctx.job
            element = result.get("element") or "a button"
            is_account = bool(re.search(r"sign ?up|register|create", element, re.I))
            is_submit = bool(re.search(r"submit|send|finish|complete", element, re.I))
            if is_account:
                what = f"create an account on {self.ctx.site} with {self.profile.get('email') or 'your email'}"
            elif is_submit:
                what = f"submit the application ('{element}')"
            else:
                what = f"click '{element}' on {self.ctx.site}"
            if (self.profile.get("name") or "") == "":
                what += " (note: your name is empty in candidate_profile.json)"
            response = {
                "status": "needs_confirmation",
                "job": f"{job.get('title')} at {job.get('company')}",
                "site": self.ctx.site,
                "url": result.get("url"),
                "summary": (
                    f"Ready to {what} for {job.get('title')} at {job.get('company')}. "
                    "Ask the user; call confirm_job_step(approve=true) to go ahead or approve=false to skip this job."
                ),
            }
            self.pending = {"pending_actions": result.get("pending_actions") or [], "element": element,
                            "is_account": is_account, "response": response}
            return response
        self._finish_job(result)
        return None

    def _run_job(self, job: dict, tips: str = "") -> Optional[dict]:
        # A fresh, temporary worker per job. It starts from main memory, so the
        # previous worker's debrief must be in it first.
        self.wait_for_debriefs()
        self.ctx = _JobContext(job, self.profile)
        logger.info("Worker agent started for %s at %s (%s)", job.get("title"), job.get("company"), self.ctx.site)
        # Each job gets its own protected tab: it is not closed until the
        # portal confirms the application, and a failed one stays open for
        # the user while the next job opens in a new tab.
        sess = bs._get_session()
        try:
            busy = bs._busy()
            if busy:
                raise RuntimeError(busy["error"])
            if not sess.is_running:
                started = sess.start()
                if "error" in started:
                    raise RuntimeError(started["error"])
            sess.submit(lambda: sess.begin_job_tab(job["url"]))
        except Exception as exc:
            self._finish_job({"status": "error", "error": f"Could not open the job page: {exc}"})
            return None
        task = f"Apply to the job '{job.get('title')}' at {job.get('company')} on this career portal."
        if job.get("via_naukri"):
            task = (f"This is the Naukri page of '{job.get('title')}' at {job.get('company')}. Click "
                    "'Apply on company site' (it may open a new tab), then apply on the company's career portal.")
        return self._handle(self._browser_loop(task, tips))

    def _browser_loop(self, task: str, tips: str = "", handlers: Optional[dict] = None) -> dict:
        from autojob.jobs.job_deep_agent import memory_context_for

        context = _job_context_text(self.ctx, self.cv_text)
        notes = memory_context_for(self.ctx.job)
        if notes:
            context += "\n\n" + notes
        if tips.strip():
            context += "\n\nEXTRA TIPS FROM THE PLANNING AGENT (hints, not rules):\n" + tips.strip()[:1500]
        sess = bs._get_session()
        handlers = handlers or _make_handlers(self.ctx)

        def run(t: str, steps: int) -> dict:
            res = bs._agent_loop(
                sess, task=t, max_steps=steps, system_prompt=_JOB_SYSTEM_PROMPT,
                extra_context=context, handlers=handlers,
                should_stop=lambda: self.stop_requested or sess.job_tab_lost,
                confirm=_job_click_approver(sess),
            )
            self.ctx.steps.extend(str(s) for s in res.get("steps") or [])
            return res

        result = run(task, JOB_MAX_STEPS)
        # The agent said DONE: believe it only if the portal shows it.
        if self._unconfirmed_done(sess, result):
            logger.info("Agent reported DONE without a confirmation on the page; continuing")
            result = run(
                task + " You replied DONE, but the page does not show that the application was received. "
                "Finish anything still unfilled or unsubmitted. Reply DONE only when the page itself "
                "confirms the application was submitted/received.",
                12,
            )
            if self._unconfirmed_done(sess, result):
                # "answer" is dropped: it is the agent's unproven success claim.
                result = {**result, "status": "needs_user", "answer": "",
                          "summary": "The portal never showed a confirmation that the application was "
                                     "received, so it is NOT counted as applied. The tab is left open - "
                                     "please check and submit it there."}
        if sess.job_tab_lost and not self.stop_requested:
            result = {**result, "status": "error", "answer": "", "summary": "",
                      "error": "The application tab was closed before the portal confirmed the "
                               "submission, so the job is NOT applied."}
        return result

    def _unconfirmed_done(self, sess, result: dict) -> bool:
        if result.get("status") != "done" or sess.job_tab_lost:
            return False
        if str(result.get("answer") or "").upper().startswith("JOB CLOSED"):
            return False
        try:
            text = sess.submit(lambda: (sess._page.evaluate(bs._JS_TEXT) or "") + "\n" + sess._frames_text())
        except Exception as exc:
            logger.warning("Could not read the page to check the confirmation: %s", exc)
            return True
        return not _CONFIRMED_RE.search(text or "")

    # -- called by the deep agent's tools ----------------------------------

    def run_job_for_agent(self, job_id: str, tips: str = "") -> dict:
        if self.pending:
            return {"status": "needs_confirmation",
                    "instruction": "A job is already waiting for the user's approval. STOP now and end your turn."}
        if self.stop_requested:
            return {"status": "stopped", "instruction": "The user stopped the run. End your turn now."}
        if not self._job_lock.acquire(blocking=False):
            return {"error": "Another job is being applied to right now. Apply to one job at a time."}
        try:
            job = self._take(str(job_id).strip())
            if job is None:
                return {"error": f"No queued job with id {job_id}. Call list_portal_jobs."}
            if _already_applied(job["url"]):
                return {"job_id": job_id, "status": "already_applied"}
            logger.info("Career portal apply: %s at %s (%s)", job.get("title"), job.get("company"), job["url"])
            self.last_outcome = None
            paused = self._run_job(job, tips)
            if paused:
                return {"job_id": job_id, "status": "needs_confirmation", "waiting_for": paused["summary"],
                        "instruction": "STOP now: do not call any more tools; reply 'PAUSED: ...' and end your turn."}
            return self.last_outcome or {"job_id": job_id, "status": "unknown"}
        finally:
            self._job_lock.release()

    def skip_job_for_agent(self, job_id: str, reason: str) -> dict:
        job = self._take(str(job_id).strip())
        if job is None:
            return {"error": f"No queued job with id {job_id}."}
        log_application(job.get("company"), job.get("title"), job["url"], "skipped",
                        f"Agent: {reason}", job.get("source", "portal"))
        self.results.append({"company": job.get("company"), "title": job.get("title"),
                             "status": "skipped", "notes": reason[:200]})
        return {"job_id": job_id, "status": "skipped"}

    # -- driving the run ---------------------------------------------------

    def _after_agent(self) -> dict:
        if self.pending:
            paused = dict(self.pending["response"])
            paused["jobs_left_after_this"] = len(self.queue)
            return paused
        return self.report()

    def _drive(self, message: str) -> dict:
        """
        Work through the queue. The plain loop is the default: the planning
        (deep) agent sometimes ended its turn after writing a to-do list
        without applying to anything, and the run was then reported as done.
        Set JOB_DEEP_AGENT=1 to use it; whatever it leaves in the queue is
        still applied to by the plain loop.
        """
        import os

        if os.getenv("JOB_DEEP_AGENT", "0").strip().lower() not in ("1", "true", "yes"):
            return self._process_plain()

        from autojob.jobs.job_deep_agent import AgentUnavailable, run_agent

        try:
            self.agent_notes = run_agent(self, message)
        except AgentUnavailable as exc:
            logger.warning("Job deep agent unavailable (%s); applying one by one instead.", exc)
            return self._process_plain()
        except Exception as exc:
            logger.exception("Job deep agent failed; continuing one by one")
            self.agent_notes = f"(planning agent error: {str(exc)[:200]})"
            if self.pending:
                return self._after_agent()
            return self._process_plain()
        if not self.pending and self.queue and not self.stop_requested:
            logger.warning("Job agent ended with %d job(s) still queued; applying to them directly", len(self.queue))
            self.agent_thread = None
            return self._process_plain()
        return self._after_agent()

    def process(self) -> dict:
        """Work through the queue until a job needs confirmation or the queue is empty."""
        if self.pending:
            return self._after_agent()
        if not self.queue or self.stop_requested:
            return self.report()
        self.agent_thread = None  # fresh agent thread for a fresh pass over the queue
        return self._drive(
            f"There are {len(self.queue)} company-site job(s) in the queue. Apply to them one by one, "
            "reflecting and updating your memory after each."
        )

    def _process_plain(self) -> dict:
        while self.queue and not self.stop_requested:
            job = self.queue.pop(0)
            self.save_queue()
            if _already_applied(job["url"]):
                continue
            logger.info("Career portal apply: %s at %s (%s)", job.get("title"), job.get("company"), job["url"])
            paused = self._run_job(job)
            if paused:
                paused["jobs_left_after_this"] = len(self.queue)
                return paused
        return self.report()

    def _continue_after_decision(self) -> dict:
        outcome = self.last_outcome or {}
        if self.stop_requested:
            return self.report()
        if self.agent_thread and self.queue:
            return self._drive(
                "The paused job is finished after the user's decision. Outcome: "
                + json.dumps(outcome, ensure_ascii=False)[:2500]
                + "\nReflect on it and update your memory, then continue with the rest of the queue."
            )
        return self._process_plain()

    def confirm(self, approve: bool) -> dict:
        if not self.pending or not self.ctx:
            return {"error": "Nothing is waiting for confirmation."}
        pending, self.pending = self.pending, None
        sess = bs._get_session()
        if not approve:
            self._finish_job({"status": "needs_user", "summary": f"Skipped by user at '{pending['element']}'."})
            return self._continue_after_decision()

        if pending["is_account"]:
            self.ctx.email_access_ok = True
        # Other browser use while paused may have switched tabs; the approved
        # click must land in this job's tab.
        try:
            tab_ok = sess.submit(sess.resume_job_tab)
        except Exception:
            tab_ok = False
        if not tab_ok:
            self._finish_job({"status": "error", "error": "The application tab was closed while waiting for "
                                                          "your approval, so the job is NOT applied."})
            return self._continue_after_decision()
        actions = pending["pending_actions"]
        done: list[dict] = []
        handlers = _make_handlers(self.ctx)
        try:
            # Only the approved click runs with confirm=True; anything after it
            # goes through the normal rails again.
            stopped = bs._run_actions(sess, actions[:1], True, done, handlers=handlers)
            if not stopped and len(actions) > 1:
                stopped = bs._run_actions(sess, actions[1:], _job_click_approver(sess), done, handlers=handlers)
        except Exception as exc:
            stopped = {"error": str(exc)}
        if stopped and stopped.get("status") == "needs_confirmation":
            paused = self._handle({**stopped, "pending_actions": actions[len(done):],
                                   "url": sess.submit(lambda: sess._page.url)})
            if paused:
                return paused
            return self._continue_after_decision()

        note = f"The user approved '{pending['element']}' and it was clicked."
        if stopped:
            note += f" A following action failed: {bs._short_error(stopped)}"
        result = self._browser_loop(
            f"Apply to the job '{self.ctx.job.get('title')}' at {self.ctx.job.get('company')} on this "
            f"career portal. {note} Continue from here.",
            handlers=handlers,
        )
        paused = self._handle(result)
        if paused:
            paused["jobs_left_after_this"] = len(self.queue)
            return paused
        return self._continue_after_decision()

    def report(self) -> dict:
        self.wait_for_debriefs(timeout_s=30)
        rep = job_application_report.invoke({})
        rep["this_run"] = self.results
        if self.lessons:
            rep["learned_this_run"] = self.lessons[-10:]
        # The lifetime totals read like "this run applied to N jobs" even when
        # every company-site attempt failed; lead with what happened now.
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        rep["summary"] = (
            f"Company-site jobs this run: {counts.get('applied', 0)} applied, "
            f"{counts.get('failed', 0)} failed, {counts.get('needs_user', 0)} need you, "
            f"{counts.get('skipped', 0)} skipped, {len(self.queue)} still queued. "
            f"(All time: {rep['summary']})"
        )
        rep["applied_this_run"] = [f"{r['title']} at {r['company']}" for r in self.results if r["status"] == "applied"]
        rep["not_applied_this_run"] = [f"{r['title']} at {r['company']}: {r['status']} - {r['notes'][:120]}"
                                       for r in self.results if r["status"] != "applied"]
        rep["instruction"] = ("Tell the user exactly this summary. Only jobs in applied_this_run were applied; "
                              "never say other jobs were applied or that applying is done when they were not.")
        if self.agent_notes:
            rep["agent_notes"] = self.agent_notes[:1500]
        return rep


_run = _ApplyRun()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool("auto_apply_jobs")
def auto_apply_jobs(
    keywords: str = "", location: str = "", max_applications: int = 15, include_company_sites: bool = True
) -> dict[str, Any]:
    """
    Apply to jobs that match the user's CV. Searches Naukri with the CV profile and applies to matching Naukri jobs; jobs that apply on a company career portal are skipped during the Naukri pass, saved, and then applied to afterwards by the learning job agent (sign-up/sign-in, verification email codes, resume upload, screening questions), which remembers what worked on each site. Before creating any account and before every final Submit it returns status 'needs_confirmation' — tell the user exactly what will happen and call confirm_job_step. Everything is logged to data/job_applications/applications_log.txt.
    """
    # A repeated "apply jobs" request must not queue up behind the running one
    # and then start a second full run as soon as it finishes.
    if not _run.lock.acquire(blocking=False):
        return {"error": "Job applications are already running. Use job_application_report for progress "
                         "or stop_job_applications to stop."}
    try:
        return _auto_apply_locked(keywords, location, max_applications, include_company_sites)
    finally:
        _run.lock.release()


def _auto_apply_locked(keywords: str, location: str, max_applications: int, include_company_sites: bool) -> dict:
    """Body of auto_apply_jobs; caller holds _run.lock."""
    if _run.pending:
        return {"error": "A job is waiting for confirmation. Call confirm_job_step first."}
    try:
        _run.profile = load_profile()
    except Exception as exc:
        return {"error": f"Could not read the CV/profile: {exc}"}
    _run.cv_text = _cv_text()
    _run.stop_requested = False
    _run.results = []
    _run.lessons = []

    # The Naukri bot drives its own Edge on the same profile; release ours first.
    bs._get_session().stop()
    from autojob.jobs.naukri_service import run_naukri_apply

    # Company-site jobs go into the saved queue the moment they are found, so a
    # Naukri run that fails or is stopped part-way does not lose them.
    naukri = run_naukri_apply(keywords=keywords, location=location, max_applications=max_applications,
                              on_external=_run.enqueue)
    if "error" in naukri:
        if not (include_company_sites and _run.queue):
            return naukri
        logger.warning("Naukri pass failed (%s); applying to the saved company-site jobs anyway.", naukri["error"])
        out = _run.process()
        out["naukri_summary"] = naukri["error"]
        return out
    for r in naukri.get("all_results", []):
        if r.get("status") == "applied":
            log_application(r.get("company"), r.get("job_title"), r.get("url"), "applied",
                            f"Naukri, match {r.get('match_score')}", "naukri")
        elif r.get("status") == "needs_user":
            log_application(r.get("company"), r.get("job_title"), r.get("url"), "needs_user",
                            r.get("message", ""), "naukri")

    # Company-site jobs were skipped during the Naukri pass and are already in
    # the queue (enqueue ignores duplicates); they are applied to now.
    for job in naukri.get("external_jobs", []):
        _run.enqueue(job)
    if not include_company_sites:
        out = _run.report()
        out["summary"] = (f"{out.get('summary', '')} {len(_run.queue)} company-site job(s) saved for later "
                          "(apply_saved_company_jobs).").strip()
    else:
        out = _run.process()
    out["naukri_summary"] = naukri.get("summary")
    return out


@tool("apply_to_job_url")
def apply_to_job_url(url: str, company: str = "", title: str = "") -> dict[str, Any]:
    """
    Apply to one specific job given its URL (a company career page, LinkedIn, Indeed, etc.) using the user's CV and profile. Pauses with 'needs_confirmation' before creating an account and before the final Submit; call confirm_job_step with the user's answer.
    """
    with _run.lock:
        if _run.pending:
            return {"error": "A job is waiting for confirmation. Call confirm_job_step first."}
        try:
            _run.profile = load_profile()
        except Exception as exc:
            return {"error": f"Could not read the CV/profile: {exc}"}
        _run.cv_text = _cv_text()
        _run.stop_requested = False
        target = url.strip()
        if "://" not in target:
            target = "https://" + target
        _run.enqueue({"url": target, "company": company, "title": title, "source": "manual"}, front=True)
        return _run.process()


@tool("apply_saved_company_jobs")
def apply_saved_company_jobs() -> dict[str, Any]:
    """
    Apply to the company-site jobs saved earlier (skipped during a Naukri run, or left when a run was stopped) without a new Naukri search. Same confirmations as auto_apply_jobs: pauses with 'needs_confirmation' before any account creation or final Submit.
    """
    if not _run.lock.acquire(blocking=False):
        return {"error": "Job applications are already running."}
    try:
        if _run.pending:
            return {"error": "A job is waiting for confirmation. Call confirm_job_step first."}
        if not _run.queue:
            return {"status": "empty", "summary": "No saved company-site jobs are waiting."}
        try:
            _run.profile = load_profile()
        except Exception as exc:
            return {"error": f"Could not read the CV/profile: {exc}"}
        _run.cv_text = _cv_text()
        _run.stop_requested = False
        _run.results = []
        _run.lessons = []
        return _run.process()
    finally:
        _run.lock.release()


@tool("teach_job_agent")
def teach_job_agent(note: str, kind: str = "instruction") -> dict[str, Any]:
    """
    Save something the user tells Veronica for future job applications, into the job agent's long-term memory. kind='answer' for an answer to an application question the CV/profile does not cover (e.g. 'Willing to work night shifts: No', 'Current CTC: 6 LPA'); kind='instruction' for a standing preference (e.g. 'Skip jobs that need relocation outside Gujarat'). Only save what the user actually said.
    """
    try:
        from autojob.jobs.job_deep_agent import teach

        path = teach(note, "answer" if str(kind).lower().startswith("answer") else "instruction")
        return {"status": "saved", "file": path, "summary": "Saved. The job agent will use it from now on."}
    except Exception as exc:
        return {"error": f"Could not save: {exc}"}


@tool("confirm_job_step")
def confirm_job_step(approve: bool) -> dict[str, Any]:
    """
    Answer a paused job application. approve=true only after the user clearly said yes to exactly what was described (creating that account / submitting that application); approve=false skips this job and moves to the next one.
    """
    with _run.lock:
        return _run.confirm(bool(approve))


@tool("stop_job_applications")
def stop_job_applications() -> dict[str, Any]:
    """Stop the job application run (Naukri, Indeed, LinkedIn continuous applying and company portals) after the current step."""
    _run.stop_requested = True
    try:
        from autojob.jobs.naukri_service import stop_naukri_apply

        stop_naukri_apply.invoke({})
    except Exception:
        pass
    try:
        from autojob.jobs.continuous_apply import request_stop

        request_stop()
    except Exception:
        logger.warning("Could not stop continuous applying", exc_info=True)
    # A running run holds the lock until its current step ends; it sees
    # stop_requested by itself, so do not make the user's "stop" wait for it.
    if _run.lock.acquire(timeout=5):
        try:
            if _run.pending and _run.ctx:
                _run.pending = None
                _run._finish_job({"status": "stopped", "summary": "Run stopped by user."})
        finally:
            _run.lock.release()
    return {"status": "stopping", "summary": (
        f"Stopping job applications. {len(_run.queue)} company-site job(s) stay saved; "
        "say 'apply saved company jobs' to continue them later.")}


@tool("job_application_report")
def job_application_report() -> dict[str, Any]:
    """
    Report how many jobs have been applied to, on which companies/sites, and what needs the user's attention. Reads data/job_applications/applications_log.txt.
    """
    log = _load_log()
    by_status: dict[str, int] = {}
    companies: dict[str, int] = {}
    for e in log:
        by_status[e["status"]] = by_status.get(e["status"], 0) + 1
        if e["status"] == "applied":
            companies[e["company"]] = companies.get(e["company"], 0) + 1
    accounts = 0
    if ACCOUNTS_TXT.exists():
        accounts = sum(1 for ln in ACCOUNTS_TXT.read_text(encoding="utf-8").splitlines() if ln.count("|") == 2
                       and not ln.startswith("created"))
    attention = [e for e in log if e["status"] == "needs_user"][-10:]
    return {
        "total_applied": by_status.get("applied", 0),
        "companies_applied_to": len(companies),
        "applied_per_company": dict(sorted(companies.items(), key=lambda kv: -kv[1])),
        "by_status": by_status,
        "portal_accounts_created": accounts,
        "needs_your_attention": attention,
        "queued_jobs": len(_run.queue),
        "waiting_for_confirmation": bool(_run.pending),
        "log_file": str(LOG_TXT),
        "summary": (
            f"Applied to {by_status.get('applied', 0)} job(s) at {len(companies)} compan(y/ies); "
            f"{by_status.get('needs_user', 0)} need your attention."
        ),
    }


from autojob.jobs.live_questions import answer_job_question  # noqa: E402 - Boss answers a live screening question

JOB_APPLY_LANGCHAIN_TOOLS = [
    answer_job_question,
    auto_apply_jobs,
    apply_to_job_url,
    apply_saved_company_jobs,
    teach_job_agent,
    confirm_job_step,
    stop_job_applications,
    job_application_report,
]


def get_job_apply_tools() -> list[dict]:
    from autojob.core.langchain_bridge import get_gemini_tools_from_langchain

    return get_gemini_tools_from_langchain(JOB_APPLY_LANGCHAIN_TOOLS)


def execute_job_apply_function(name: str, args: dict) -> dict[str, Any]:
    for t in JOB_APPLY_LANGCHAIN_TOOLS:
        if t.name == name:
            try:
                return t.invoke(args)
            except Exception as exc:
                logger.exception("%s failed", name)
                return {"error": f"Invalid arguments or execution failed for {name}: {exc}"}
    return {"error": f"Unknown job apply function: {name}"}
