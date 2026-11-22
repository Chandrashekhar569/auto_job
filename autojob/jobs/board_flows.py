"""
The three job boards continuous applying works on: LinkedIn, Indeed, Naukri.

Each board knows how to
    open()      start the browser and check Boss is signed in;
    fetch()     list jobs from one source page (the board's own recommendations
                for Boss first, then keyword searches), newest first;
    inspect()   open a job: title, company, description, and whether it is
                already applied / closed / quick-apply / company-site only;
    apply()     open the quick-apply flow and complete it (easy_apply.ApplyFlow;
                Naukri keeps its own chatbot handling in naukri_service).

LinkedIn and Indeed run in Veronica's own Edge session (browser_service), so
popups, the answer engine and the vision fallback all work the same there.
Naukri drives its own Edge through NaukriJobBot, which already handles
Naukri's screening chatbot and company-site redirects.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional
from urllib.parse import urlencode

from autojob.integrations import browser_service as bs
from autojob.jobs.easy_apply import ApplyFlow, FlowResult, FlowSpec

logger = logging.getLogger(__name__)

LINKEDIN = "https://www.linkedin.com"
INDEED = os.getenv("INDEED_BASE_URL", "https://in.indeed.com").rstrip("/")


@dataclass
class BoardJob:
    portal: str
    job_id: str
    url: str
    title: str = ""
    company: str = ""
    easy: Optional[bool] = None     # from the list card, when it says
    applied: bool = False           # the list card already says "Applied"
    source: str = ""


def _dedupe_jobs(jobs: list[BoardJob]) -> list[BoardJob]:
    seen, out = set(), []
    for j in jobs:
        if j.job_id and j.job_id not in seen:
            seen.add(j.job_id)
            out.append(j)
    return out


class Board:
    name = ""
    label = ""
    pages_per_source = {"recommended": 3, "search": 5}

    def __init__(self, should_stop: Callable[[], bool] = lambda: False):
        self.should_stop = should_stop

    def open(self) -> Optional[dict]:
        """None when ready, else {"reason": "login" | "captcha" | "error", "detail": ...}."""
        raise NotImplementedError

    def fetch(self, source: dict, page: int) -> list[BoardJob]:
        raise NotImplementedError

    def inspect(self, job: BoardJob) -> dict:
        raise NotImplementedError

    def apply(self, job: BoardJob, answerer, dry_run: bool = False) -> FlowResult:
        raise NotImplementedError

    def request_stop(self) -> None:
        pass

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Boards in Veronica's Edge session (LinkedIn, Indeed)
# ---------------------------------------------------------------------------

class _SessionBoard(Board):
    """Common plumbing for boards driven through browser_service's session."""

    def __init__(self, should_stop: Callable[[], bool] = lambda: False):
        super().__init__(should_stop)
        self.sess = bs._get_session()

    def _goto(self, url: str, settle_ms: int = 1500) -> str:
        def _go():
            self.sess._page.goto(url, wait_until="domcontentloaded", timeout=bs.NAV_TIMEOUT_MS)
            self.sess._settle(quiet_ms=500, max_ms=4000)
            self.sess._page.wait_for_timeout(settle_ms)
            return self.sess._page.url
        return self.sess.submit(_go)

    def _eval(self, script: str, arg: Any = None):
        if arg is None:
            return self.sess.submit(lambda: self.sess._page.evaluate(script))
        return self.sess.submit(lambda: self.sess._page.evaluate(script, arg))

    def _start_browser(self) -> Optional[dict]:
        if not self.sess.is_running:
            started = self.sess.start()
            if "error" in started:
                return {"reason": "error", "detail": started["error"]}
        return None

    def _captcha(self) -> bool:
        from autojob.jobs.easy_apply import _JS_CAPTCHA

        try:
            return bool(self._eval(_JS_CAPTCHA))
        except Exception:
            return False

    def _wait_captcha(self) -> bool:
        """CAPTCHA on the page: wait on screen for Boss to finish it. True = carry on."""
        from autojob.jobs.easy_apply import wait_for_human_check

        return wait_for_human_check(self._captcha, self.should_stop, self.label,
                                    bring_front=lambda: self.sess.submit(lambda: self.sess._page.bring_to_front()))


# -- LinkedIn ---------------------------------------------------------------

def linkedin_search_url(keywords: str, location: str = "", start: int = 0, days: int = 7) -> str:
    params = {"keywords": keywords, "location": location or "India", "f_AL": "true", "sortBy": "DD",
              "f_TPR": f"r{days * 86400}"}
    if start:
        params["start"] = start
    return f"{LINKEDIN}/jobs/search/?{urlencode(params)}"


def linkedin_collection_url(start: int = 0) -> str:
    return f"{LINKEDIN}/jobs/collections/easy-apply/" + (f"?start={start}" if start else "")


_JS_LINKEDIN_LIST = r"""
async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const clean = s => (s || '').replace(/\s+/g, ' ').trim();
  // The results list scrolls inside its own pane and renders cards lazily.
  const first = document.querySelector('[data-occludable-job-id], [data-job-id]');
  let pane = first ? first.parentElement : null;
  while (pane && pane !== document.body) {
    const st = getComputedStyle(pane);
    if (pane.scrollHeight > pane.clientHeight + 20 && /(auto|scroll)/.test(st.overflowY)) break;
    pane = pane.parentElement;
  }
  for (let i = 0; i < 14; i++) {
    if (pane && pane !== document.body) pane.scrollBy(0, 500); else window.scrollBy(0, 500);
    await sleep(220);
  }
  const out = [], seen = new Set();
  const add = (id, card) => {
    if (!id || !/^\d+$/.test(id) || seen.has(id)) return;
    seen.add(id);
    const t = card && card.querySelector('a[href*="/jobs/view/"] strong, a[href*="/jobs/view/"], .job-card-list__title, strong');
    const c = card && card.querySelector('.artdeco-entity-lockup__subtitle, .job-card-container__primary-description, [class*="company" i]');
    const text = card ? (card.innerText || '') : '';
    out.push({id, title: t ? clean(t.innerText).split(' with verification')[0] : '', company: c ? clean(c.innerText) : '',
              applied: /(^|\n)\s*Applied\b/.test(text), easy: /Easy Apply/i.test(text) ? true : null});
  };
  for (const el of document.querySelectorAll('[data-occludable-job-id], [data-job-id]')) {
    add(el.getAttribute('data-occludable-job-id') || el.getAttribute('data-job-id'), el);
  }
  for (const a of document.querySelectorAll('a[href*="/jobs/view/"]')) {
    const m = (a.getAttribute('href') || '').match(/\/jobs\/view\/(\d+)/);
    if (m) add(m[1], a.closest('li') || a.parentElement);
  }
  return out;
}
"""

_JS_LINKEDIN_JOB = r"""
() => {
  const clean = s => (s || '').replace(/\s+/g, ' ').trim();
  document.querySelectorAll('[data-vro-apply]').forEach(e => e.removeAttribute('data-vro-apply'));
  const top = document.querySelector('.job-details-jobs-unified-top-card__container--two-pane, '
    + '.jobs-unified-top-card, .job-view-layout, .jobs-details__main-content, main') || document.body;
  const h1 = top.querySelector('h1') || document.querySelector('h1');
  const comp = top.querySelector('.job-details-jobs-unified-top-card__company-name a, '
    + '.job-details-jobs-unified-top-card__company-name, .jobs-unified-top-card__company-name a, a[href*="/company/"]');
  const desc = document.querySelector('#job-details, .jobs-description__content, .jobs-box__html-content, '
    + '[class*="jobs-description"]');
  const shown = b => { const r = b.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
  const btns = Array.from(top.querySelectorAll('button, a[role=button], a')).filter(shown);
  const label = b => clean(b.innerText) + ' | ' + (b.getAttribute('aria-label') || '');
  const easy = btns.find(b => /^easy apply\b/i.test(clean(b.innerText)) || /^easy apply to\b/i.test(b.getAttribute('aria-label') || ''));
  const external = !easy && btns.find(b => /^apply\b/i.test(clean(b.innerText)) && /company|website/i.test(label(b))
                                          || /on company website/i.test(b.getAttribute('aria-label') || ''));
  if (easy) easy.setAttribute('data-vro-apply', 'easy');
  const head = clean(top.innerText).slice(0, 2500);
  return {
    title: h1 ? clean(h1.innerText) : '',
    company: comp ? clean(comp.innerText) : '',
    description: desc ? clean(desc.innerText).slice(0, 4000) : '',
    applied: /\bApplied \d+ \w+ ago\b|\bApplication submitted\b|\bYou applied\b|See application/i.test(head),
    closed: /No longer accepting applications|no longer available|job (has )?expired/i.test(head),
    limit: /reached .{0,60}limit|Easy Apply limit|limit daily submissions/i.test(head),
    easy: !!easy,
    external: !!external,
    login: /\/(login|authwall|checkpoint|uas)\b/.test(location.pathname)
      || /Sign in to (view|apply|see)|Join LinkedIn|Join now to apply/i.test(head.slice(0, 600)),
  };
}
"""


def _linkedin_abandon(page) -> None:
    """Close the Easy Apply window and discard the draft (never 'Save')."""
    for sel in ('.jobs-easy-apply-modal button[aria-label="Dismiss"]', 'div[role=dialog] button[aria-label="Dismiss"]',
                'button.artdeco-modal__dismiss'):
        loc = page.locator(sel)
        if loc.count() and loc.last.is_visible():
            loc.last.click(timeout=4000)
            break
    page.wait_for_timeout(900)
    for sel in ('button[data-control-name="discard_application_confirm_btn"]',
                '[role=alertdialog] button:has-text("Discard")', 'div[role=dialog] button:has-text("Discard")',
                'button:has-text("Discard")'):
        loc = page.locator(sel)
        if loc.count() and loc.first.is_visible():
            loc.first.click(timeout=4000)
            break
    page.wait_for_timeout(500)


LINKEDIN_SPEC = dict(
    portal="linkedin",
    roots=[[".jobs-easy-apply-modal", ""], ["[data-test-modal-id='easy-apply-modal']", ""],
           ["div[role=dialog]", r"apply to|contact info|additional questions|review your application|"
                                r"work experience|screening questions|home address"]],
    success=re.compile(r"application (was )?sent|your application was sent|\bapplication submitted\b|"
                       r"\bapplied \d+ (second|minute)s? ago", re.I),
    hosts=("linkedin.com",),
)


class LinkedInBoard(_SessionBoard):
    name, label = "linkedin", "LinkedIn"

    def open(self) -> Optional[dict]:
        err = self._start_browser()
        if err:
            return err
        url = self._goto(f"{LINKEDIN}/jobs/")
        # Signed out, LinkedIn either redirects to a login page or shows the guest jobs page.
        guest = self._eval(
            "() => !document.querySelector('#global-nav, .global-nav, .global-nav__me')"
            " && !!document.querySelector('a[href*=\"/login\"], a[href*=\"/signup\"], "
            "a[data-tracking-control-name*=\"sign-in\" i]')")
        if guest or re.search(r"/(login|authwall|checkpoint|uas)\b|signup", url or ""):
            return {"reason": "login", "detail": "LinkedIn is not signed in in Veronica's Edge window."}
        if self._captcha() and not self._wait_captcha():
            return {"reason": "captcha", "detail": "LinkedIn is showing a security check."}
        return None

    def fetch(self, source: dict, page: int) -> list[BoardJob]:
        if source["kind"] == "recommended":
            url = linkedin_collection_url(start=24 * page)
        else:
            url = linkedin_search_url(source["keywords"], source.get("location", ""), start=25 * page)
        self._goto(url, settle_ms=2500)
        rows = self._eval(_JS_LINKEDIN_LIST) or []
        label = "recommended" if source["kind"] == "recommended" else f"search '{source['keywords']}'"
        jobs = [BoardJob("linkedin", r["id"], f"{LINKEDIN}/jobs/view/{r['id']}/", r.get("title", ""),
                         r.get("company", ""), r.get("easy"), bool(r.get("applied")), label) for r in rows]
        logger.info("LinkedIn %s page %d: %d job(s)", label, page + 1, len(jobs))
        return _dedupe_jobs(jobs)

    def inspect(self, job: BoardJob) -> dict:
        self._goto(job.url, settle_ms=2000)
        info = self._eval(_JS_LINKEDIN_JOB) or {}
        if not info.get("easy") and not info.get("applied"):
            # The top card renders late on slow connections.
            self.sess.submit(lambda: self.sess._page.wait_for_timeout(2500))
            info = self._eval(_JS_LINKEDIN_JOB) or info
        if self._captcha() and self._wait_captcha():
            info = self._eval(_JS_LINKEDIN_JOB) or {}
        info["captcha"] = self._captcha()
        return info

    def apply(self, job: BoardJob, answerer, dry_run: bool = False) -> FlowResult:
        def _open():
            loc = self.sess._page.locator('[data-vro-apply="easy"]')
            if not loc.count():
                raise LookupError("Easy Apply button not found")
            loc.first.click(timeout=8000)
            self.sess._page.wait_for_timeout(1800)
        try:
            self.sess.submit(_open)
        except Exception as exc:
            return FlowResult(status="failed", reason=f"Could not open Easy Apply: {bs._short_error({'error': str(exc)})}")
        spec = FlowSpec(**LINKEDIN_SPEC, abandon=_linkedin_abandon)
        return ApplyFlow(self.sess, spec, answerer, should_stop=self.should_stop, dry_run=dry_run).run()


# -- Indeed -----------------------------------------------------------------

def indeed_search_url(keywords: str, location: str = "", start: int = 0, days: int = 14) -> str:
    params = {"q": keywords, "l": location or "", "fromage": days, "sort": "date"}
    if start:
        params["start"] = start
    return f"{INDEED}/jobs?{urlencode(params)}"


_JS_INDEED_LIST = r"""
async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const clean = s => (s || '').replace(/\s+/g, ' ').trim();
  for (let i = 0; i < 6; i++) { window.scrollBy(0, 900); await sleep(250); }
  window.scrollTo(0, 0);
  const out = [], seen = new Set();
  for (const el of document.querySelectorAll('[data-jk], a[href*="jk="]')) {
    let jk = el.getAttribute('data-jk');
    if (!jk) { const m = (el.getAttribute('href') || '').match(/[?&](?:jk|vjk)=([a-f0-9]{8,})/i); jk = m && m[1]; }
    if (!jk || seen.has(jk)) continue;
    seen.add(jk);
    const card = el.closest('.job_seen_beacon, .cardOutline, [class*="job_seen"], [data-testid*="card" i], li') || el;
    const text = card.innerText || '';
    const t = card.querySelector('h2 a span[title], h2 span[title], h2 a, [class*="jobTitle"] span, h2, h3');
    const c = card.querySelector('[data-testid="company-name"], .companyName, [class*="companyName"], [class*="company" i]');
    out.push({id: jk, title: t ? clean(t.getAttribute('title') || t.innerText) : '',
              company: c ? clean(c.innerText) : '',
              easy: /easily apply|easy apply|indeed apply|apply with indeed/i.test(text),
              applied: /(^|\n)\s*(Applied|You applied)\b/i.test(text)});
  }
  return out;
}
"""

_JS_INDEED_JOB = r"""
() => {
  const clean = s => (s || '').replace(/\s+/g, ' ').trim();
  document.querySelectorAll('[data-vro-apply]').forEach(e => e.removeAttribute('data-vro-apply'));
  // Two layouts: the classic one (h1, #jobDescriptionText) and the 2026 React one (data-testid="vj-...").
  const h1 = document.querySelector('[data-testid="vj-job-title"], [data-testid="jobsearch-JobInfoHeader-title"], h1');
  const comp = document.querySelector('[data-testid="inlineHeader-companyName"], [data-company-name], '
    + '[data-testid="company-info-metadata"], .jobsearch-CompanyInfoContainer a, [class*="companyName"]');
  const desc = document.querySelector('#jobDescriptionText, [data-testid="viewjob-job-content"], '
    + '.react-native-html-content, [class*="jobDescription"]');
  const shown = b => { const r = b.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
  const btns = Array.from(document.querySelectorAll('button, a, [role=button]')).filter(shown);
  const label = b => clean(b.innerText) + ' ' + (b.getAttribute('aria-label') || '');
  const external = btns.find(b => /apply on company site|company site/i.test(label(b)));
  const iaBox = document.querySelector('[data-testid="viewjob-indeed-apply"], [data-testid="primary-apply-action"]');
  const easy = !external && (document.querySelector('#indeedApplyButton, [id*="indeedApplyButton"], '
                 + 'button[class*="IndeedApplyButton"], [data-testid*="indeedApply" i] button')
               || (iaBox && Array.from(iaBox.querySelectorAll('button, a, [role=button]')).filter(shown)
                      .find(b => /apply now|easily apply/i.test(label(b))))
               || btns.find(b => /^apply now$|^easily apply$/i.test(clean(b.innerText))));
  if (easy) easy.setAttribute('data-vro-apply', 'easy');
  const text = clean(document.body.innerText).slice(0, 5000);
  return {
    title: h1 ? clean(h1.innerText).replace(/ - job post$/i, '') : '',
    company: comp ? ((comp.innerText || '').split(/\n|·/).map(clean).find(Boolean) || '') : '',
    description: desc ? clean(desc.innerText).slice(0, 4000) : '',
    applied: btns.some(b => /^applied$/i.test(clean(b.innerText))) || /You applied|Application submitted|Applied on \w+/i.test(text.slice(0, 1500)),
    closed: /This job has expired|no longer accepting|job is no longer available|this job post is closed/i.test(text),
    easy: !!easy,
    external: !!external,
    login: /secure\.indeed\.com\/(account|auth)/.test(location.href),
    limit: false,
  };
}
"""

INDEED_SPEC = dict(
    portal="indeed",
    roots=[["#ia-container", ""], ["[class*='ia-BasePage']", ""], ["main form", ""],
           ["form", r"continue|review|submit"], ["main", r"continue|review|submit|application"]],
    success=re.compile(r"application (has been |was )?submitted|you('ve| have) applied|your application has been sent|"
                       r"application sent", re.I),
    success_url=re.compile(r"/post-apply|/applied\b|confirmation", re.I),
    # Only Indeed's apply pages are the form; the job page's own search box never is.
    root_url=re.compile(r"smartapply\.indeed\.com|indeedapply|/apply\b|/beta/", re.I),
    hosts=("indeed.com",),
)


class IndeedBoard(_SessionBoard):
    name, label = "indeed", "Indeed"
    pages_per_source = {"recommended": 1, "search": 5}

    def open(self) -> Optional[dict]:
        err = self._start_browser()
        if err:
            return err
        self._goto(f"{INDEED}/", settle_ms=2000)
        if self._captcha() and not self._wait_captcha():
            return {"reason": "captcha", "detail": "Indeed is showing a 'verify you are human' check."}
        signed_out = self._eval(
            "() => !!document.querySelector('a[href*=\"/account/login\"], a[href*=\"secure.indeed.com/auth\"], "
            "[data-gnav-element-name=\"SignIn\"]')")
        if signed_out:
            return {"reason": "login", "detail": "Indeed is not signed in in Veronica's Edge window."}
        return None

    def fetch(self, source: dict, page: int) -> list[BoardJob]:
        if source["kind"] == "recommended":
            if page:
                return []
            url = f"{INDEED}/"
        else:
            url = indeed_search_url(source["keywords"], source.get("location", ""), start=10 * page)
        landed = self._goto(url, settle_ms=2500)
        if self._captcha():
            if not self._wait_captcha():
                raise _NeedsBoss("captcha", "Indeed is showing a 'verify you are human' check.")
            landed = self.sess.submit(lambda: self.sess._page.url)
        if re.search(r"secure\.indeed\.com/(auth|account)", landed or ""):
            # Signed out, Indeed shows only the first page of results.
            raise _NeedsBoss("login", "Indeed wants Boss signed in to show more jobs.")
        rows = self._eval(_JS_INDEED_LIST) or []
        label = "recommended" if source["kind"] == "recommended" else f"search '{source['keywords']}'"
        jobs = [BoardJob("indeed", r["id"], f"{INDEED}/viewjob?jk={r['id']}", r.get("title", ""),
                         r.get("company", ""), bool(r.get("easy")), bool(r.get("applied")), label) for r in rows]
        logger.info("Indeed %s page %d: %d job(s), %d easily apply", label, page + 1, len(jobs),
                    sum(1 for j in jobs if j.easy))
        return _dedupe_jobs(jobs)

    def inspect(self, job: BoardJob) -> dict:
        self._goto(job.url, settle_ms=1800)
        info = self._eval(_JS_INDEED_JOB) or {}
        if self._captcha() and self._wait_captcha():
            info = self._eval(_JS_INDEED_JOB) or {}
        info["captcha"] = self._captcha()
        return info

    def apply(self, job: BoardJob, answerer, dry_run: bool = False) -> FlowResult:
        sess = self.sess
        state: dict = {}

        def _open():
            state["orig"] = sess._page
            before = sess._open_pages()
            loc = sess._page.locator('[data-vro-apply="easy"]')
            if not loc.count():
                raise LookupError("Apply now button not found")
            loc.first.click(timeout=8000)
            sess._page.wait_for_timeout(1500)
            sess._adopt_new_tab(before)
            state["tab"] = sess._page if sess._page is not state["orig"] else None
            try:
                sess._page.wait_for_load_state("domcontentloaded", timeout=bs.NAV_TIMEOUT_MS)
            except Exception:
                pass
            sess._page.wait_for_timeout(1500)

        def _leave(page=None):
            tab = state.get("tab")
            if tab is not None and not tab.is_closed():
                tab.close()
            orig = state.get("orig")
            if orig is not None and not orig.is_closed():
                sess._page = orig
                sess._snapshot = []
                orig.bring_to_front()

        try:
            sess.submit(_open)
        except Exception as exc:
            return FlowResult(status="failed", reason=f"Could not open Indeed Apply: {bs._short_error({'error': str(exc)})}")
        spec = FlowSpec(**INDEED_SPEC)
        try:
            return ApplyFlow(sess, spec, answerer, should_stop=self.should_stop, dry_run=dry_run).run()
        finally:
            try:
                sess.submit(_leave)
            except Exception as exc:
                logger.debug("Could not close the Indeed apply tab: %s", exc)


# ---------------------------------------------------------------------------
# Naukri (its own Edge, through NaukriJobBot)
# ---------------------------------------------------------------------------

class _NeedsBoss(RuntimeError):
    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason, self.detail = reason, detail


class NaukriBoard(Board):
    name, label = "naukri", "Naukri"
    pages_per_source = {"recommended": 1, "search": 4}

    def __init__(self, should_stop: Callable[[], bool] = lambda: False, on_external=None):
        super().__init__(should_stop)
        from autojob.jobs.naukri_service import NaukriJobBot

        self.bot = NaukriJobBot()
        self.on_external = on_external
        self._details: dict[str, dict] = {}

    def open(self) -> Optional[dict]:
        from autojob.jobs import naukri_service as ns

        with ns._active_lock:
            if ns._active_bot is not None:
                return {"reason": "error", "detail": "A Naukri apply session is already running."}
            ns._active_bot = self.bot  # stop_naukri_apply / check_naukri_status see this run
        try:
            self.bot.start()
        except Exception as exc:
            return {"reason": "error", "detail": str(exc)}
        if not self.bot.verify_login():
            return {"reason": "login", "detail": "Naukri is not signed in in Veronica's Edge window."}
        return None

    def fetch(self, source: dict, page: int) -> list[BoardJob]:
        if source["kind"] == "recommended":
            rows = self.bot.recommended_jobs() if page == 0 else []
            label = "recommended"
        else:
            rows = self.bot.search_jobs(source["keywords"], source.get("location", ""), page=page + 1)
            label = f"search '{source['keywords']}'"
        jobs = [BoardJob("naukri", _naukri_id(r["url"]), r["url"], r.get("title", ""), r.get("company", ""),
                         None, False, label) for r in rows if r.get("url")]
        return _dedupe_jobs(jobs)

    def inspect(self, job: BoardJob) -> dict:
        d = self.bot.get_job_details(job.url)
        self._details[job.job_id] = d
        text = ""
        try:
            text = self.bot.page.evaluate("() => document.body.innerText.slice(0, 4000)") or ""
        except Exception:
            pass
        return {
            "title": d.get("title") or job.title, "company": d.get("company") or job.company,
            "description": d.get("description", ""), "error": d.get("error"),
            "applied": bool(re.search(r"\bAlready Applied\b", text)),
            "closed": bool(re.search(r"no longer accepting|job (has )?expired|this job is closed", text, re.I)),
            "easy": True,  # Naukri decides on the Apply click (its own form, chatbot or company site)
            "external": False, "login": False, "limit": False,
            "captcha": bool(re.search(r"captcha|are you a robot", text, re.I)),
        }

    def apply(self, job: BoardJob, answerer, dry_run: bool = False) -> FlowResult:
        if dry_run:
            return FlowResult(status="dry_run", reason="Dry run: Naukri's Apply was NOT clicked.")
        res = self.bot.apply_to_job(answerer)
        status, msg = res.get("status"), res.get("message", "")
        if status == "applied":
            return FlowResult(status="applied", confirmation=msg)
        if status == "needs_user":
            q = res.get("question") or ""
            return FlowResult(status="needs_user", questions=[q] if q else [], reason=msg)
        if status == "external":
            return FlowResult(status="external", external_url=res.get("external_url") or "", reason=msg)
        if status == "skipped":
            return FlowResult(status="skipped", reason=msg)
        if status == "unconfirmed":  # clicked through, but Naukri never confirmed: not applied
            return FlowResult(status="unconfirmed", reason=msg)
        if re.search(r"captcha", msg, re.I):
            return FlowResult(status="paused", reason="captcha")
        return FlowResult(status="failed", reason=msg)

    def request_stop(self) -> None:
        self.bot.request_stop()

    def close(self) -> None:
        from autojob.jobs import naukri_service as ns

        try:
            self.bot.stop()
        finally:
            with ns._active_lock:
                if ns._active_bot is self.bot:
                    ns._active_bot = None


def _naukri_id(url: str) -> str:
    m = re.search(r"-(\d{6,})(?:\?|$|/)", url or "")
    return m.group(1) if m else re.sub(r"\W+", "", (url or "").split("?")[0])[-40:]


BOARDS = {"linkedin": LinkedInBoard, "indeed": IndeedBoard, "naukri": NaukriBoard}
NeedsBoss = _NeedsBoss
