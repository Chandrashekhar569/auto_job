"""
Keep Boss's profile in step with the CV file.

When the CV in data/profile/cv/ changes (new file or new content), the next
read of the profile updates both profile files from it before anything is
filled in:

    data/profile/ABOUT_ME.md                         CV-derived sections rewritten
    data/job_applications/candidate_profile.json     CV-derived fields refreshed

Boss's own answers are never touched: CTC, notice period, relocation, work
authorization, address, standard answers and "Learned facts". The ABOUT_ME
rewrite is checked before it is saved (every protected line still there, every
section still there) and both files are backed up first. A rewrite that fails
the check is not saved; the old profile stays and the CV is not retried until
it changes again.

State: data/profile/.cv_sync.json  {"sha256", "file", "status", "at", ...}
"""

import hashlib
import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from autojob.core.paths import ABOUT_ME_FILE, CV_DIR, DATA_DIR, PROFILE_DIR

logger = logging.getLogger(__name__)

STATE_FILE = PROFILE_DIR / ".cv_sync.json"
PROFILE_JSON = DATA_DIR / "job_applications" / "candidate_profile.json"

# ABOUT_ME sections that hold Boss's answers, not CV content: copied through as they are.
PROTECTED_SECTIONS = ("Job search preferences", "Years of experience by skill",
                      "Standard answers for application forms", "Learned facts")
# Lines inside protected sections that do come from the CV and may change.
CV_LINES_IN_PROTECTED = re.compile(r"^- (Target roles|Currently employed)\b", re.I)
# candidate_profile.json fields filled from the CV; everything else is Boss's.
CV_FIELDS = ("name", "email", "phone", "location", "linkedin", "experience_years", "current_title",
             "current_company", "education", "skills", "summary")

_lock = threading.Lock()
_last_stat: Optional[tuple] = None  # (path, size, mtime) seen at the last check


# -- CV file ------------------------------------------------------------------

def find_cv() -> Optional[Path]:
    for pattern in ("*.pdf", "*.PDF", "*.docx", "*.DOCX"):
        files = sorted(CV_DIR.glob(pattern))
        if files:
            return files[0]
    return None


def cv_text(path: Path) -> str:
    try:
        if path.suffix.lower() == ".pdf":
            import pymupdf

            with pymupdf.open(str(path)) as doc:
                return "\n".join(page.get_text("text") for page in doc)
        import docx

        return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    except Exception as exc:
        logger.warning("Could not read CV %s: %s", path, exc)
        return ""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(**state: Any) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({**state, "at": datetime.now().isoformat(timespec="seconds")},
                                     indent=2, ensure_ascii=False), encoding="utf-8")


# -- ABOUT_ME.md ----------------------------------------------------------------

def _sections(text: str) -> dict[str, str]:
    """'## Heading' -> body (without the heading line)."""
    out, name, buf = {}, None, []
    for line in text.splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            if name is not None:
                out[name] = "\n".join(buf)
            name, buf = m.group(1), []
        elif name is not None:
            buf.append(line)
    if name is not None:
        out[name] = "\n".join(buf)
    return out


def _protected_lines(text: str) -> list[str]:
    secs = _sections(text)
    lines = []
    for name in PROTECTED_SECTIONS:
        for line in secs.get(name, "").splitlines():
            if line.strip() and not CV_LINES_IN_PROTECTED.match(line.strip()):
                lines.append(line.rstrip())
    return lines


def check_about_me(old: str, new: str) -> list[str]:
    """Problems that stop a rewritten ABOUT_ME.md from being saved ([] = fine)."""
    problems = []
    if not new.lstrip().startswith("# "):
        problems.append("title line missing")
    if len(new) < 0.5 * len(old):
        problems.append(f"too short ({len(new)} vs {len(old)} chars)")
    missing_secs = [s for s in _sections(old) if s not in _sections(new)]
    if missing_secs:
        problems.append("sections dropped: " + ", ".join(missing_secs))
    new_lines = {line.rstrip() for line in new.splitlines()}
    lost = [line for line in _protected_lines(old) if line not in new_lines]
    if lost:
        problems.append("Boss's answers changed: " + " | ".join(line[:60] for line in lost[:3]))
    return problems


_ABOUT_SYSTEM = """You update a person's profile file from their new CV. Output ONLY the full updated Markdown file.
Rules:
- Keep the exact structure: the same title, the same '## ' sections in the same order, and the rules block at the top.
- Sections that come from the CV (headline, summary, experience in numbers, work history, skills, projects,
  education, certifications, identity/contact details the CV shows) must match the NEW CV: copy its facts and
  wording; add what is new, update what changed.
- Total experience: use the figure the CV states (e.g. '8+ years' -> 'For a whole-number field, use 8').
- A fact in the old file that the new CV no longer mentions (a job, a project detail) is still true: keep it and
  mark it '(not on the current CV)'. Never delete notes that warn agents about something.
- Sections """ + ", ".join(f"'{s}'" for s in PROTECTED_SECTIONS) + """ are the person's own answers: copy every
  line character for character. Only the 'Target roles' and 'Currently employed' lines there may be updated from
  the CV.
- Keep every UNKNOWN that the CV does not answer. Never invent, estimate or round anything.
- No code fences, no comments about what you changed."""


def _rewrite_about_me(old: str, cv: str, cv_path: Path) -> str:
    from autojob.core.llm_router import fast_chat

    prompt = f"CURRENT PROFILE FILE:\n{old}\n\nNEW CV TEXT:\n{cv[:14000]}\n\nReturn the updated profile file."
    new = fast_chat(prompt=prompt, task_type="general", system_prompt=_ABOUT_SYSTEM,
                    temperature=0.0, max_tokens=12000)
    new = re.sub(r"<think>.*?</think>", "", new or "", flags=re.S).strip()
    new = re.sub(r"^```(?:markdown|md)?\s*\n|\n```\s*$", "", new).strip()
    # The resume path is known exactly; never leave it to the model.
    resume_line = f"- Resume file: {cv_path.as_posix()} (use upload_resume on job forms)"
    if re.search(r"(?m)^- Resume file:.*$", new):
        new = re.sub(r"(?m)^- Resume file:.*$", lambda _: resume_line, new)
    return new + "\n"


# -- candidate_profile.json -------------------------------------------------------

def _parse_cv_fields(cv: str) -> dict:
    from autojob.core.llm_router import fast_chat

    prompt = f"""Extract these fields from the CV. Return ONLY one JSON object, null for anything the CV does not state.
{{"name": "", "email": "", "phone": "", "location": "city, state", "linkedin": "full URL or null",
 "experience_years": <whole number: the total the CV states, e.g. '8+ years' -> 8>,
 "current_title": "", "current_company": "", "education": "degree - college, year",
 "skills": ["each skill/tool the CV lists, short names"], "summary": "the CV's professional summary, as written"}}

CV:
{cv[:14000]}"""
    raw = fast_chat(prompt=prompt, task_type="general", temperature=0.0, max_tokens=4096,
                    system_prompt="You extract facts from a CV exactly as written. Output only JSON.")
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S)
    raw = raw[raw.find("{"):raw.rfind("}") + 1]
    return json.loads(raw)


def merge_profile(old: dict, parsed: dict, cv: str) -> dict:
    """CV fields from `parsed` (only values that are really in the CV); Boss's own fields untouched."""
    norm = lambda t: re.sub(r"[^a-z0-9@+.]+", " ", str(t).lower())
    cv_n = norm(cv)
    out = dict(old)
    for key in CV_FIELDS:
        val = parsed.get(key)
        if val in (None, "", []):
            continue
        if key == "experience_years":
            try:
                val = int(float(val))
            except (TypeError, ValueError):
                continue
            if not 0 < val < 60:
                continue
        elif key == "skills":
            val = [s for s in dict.fromkeys(str(s).strip() for s in val) if s]
            if len(val) < 3:
                continue
        elif key in ("email", "phone", "name"):
            # Contact details must appear in the CV text itself.
            probe = re.sub(r"\D", "", str(val))[-10:] if key == "phone" else norm(val).strip()
            if probe not in (re.sub(r"\D", "", cv) if key == "phone" else cv_n):
                continue
        out[key] = val
    return out


# -- the sync ---------------------------------------------------------------------

def _backup(path: Path, stamp: str) -> None:
    if path.exists():
        path.with_name(f"{path.stem}.backup-{stamp}{path.suffix}").write_bytes(path.read_bytes())


def sync_now(cv_path: Path, sha: str) -> dict:
    """Update both profile files from the CV. Caller holds _lock."""
    cv = cv_text(cv_path)
    if len(cv.strip()) < 200:
        _save_state(sha256=sha, file=cv_path.name, status="failed", error="CV text could not be read")
        return {"status": "failed", "error": "Could not read text from the CV (scanned image?)."}

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    done, problems = [], []

    if PROFILE_JSON.exists():
        try:
            old = json.loads(PROFILE_JSON.read_text(encoding="utf-8"))
            new = merge_profile(old, _parse_cv_fields(cv), cv)
            if new != old:
                _backup(PROFILE_JSON, stamp)
                PROFILE_JSON.write_text(json.dumps(new, indent=2, ensure_ascii=False), encoding="utf-8")
            done.append("candidate_profile.json")
        except Exception as exc:
            problems.append(f"candidate_profile.json: {exc}")

    if ABOUT_ME_FILE.exists():
        try:
            from autojob.core.user_profile import _lock as about_lock

            with about_lock:
                old = ABOUT_ME_FILE.read_text(encoding="utf-8")
                new = _rewrite_about_me(old, cv, cv_path)
                issues = check_about_me(old, new)
                if issues:
                    problems.append("ABOUT_ME.md not changed - " + "; ".join(issues))
                else:
                    _backup(ABOUT_ME_FILE, stamp)
                    ABOUT_ME_FILE.write_text(new, encoding="utf-8")
                    done.append("ABOUT_ME.md")
        except Exception as exc:
            problems.append(f"ABOUT_ME.md: {exc}")

    status = "synced" if not problems else ("partial" if done else "failed")
    _save_state(sha256=sha, file=cv_path.name, status=status, updated=done, problems=problems)
    logger.info("CV sync (%s): updated %s %s", status, done, problems or "")
    return {"status": status, "updated": done, "problems": problems, "backups": f"*.backup-{stamp}"}


def sync_if_cv_changed(force: bool = False, wait: bool = False) -> dict:
    """
    Cheap check (file name/size/mtime, then a hash) on every profile read; runs
    the sync only when the CV is new. With no saved state the current CV is
    recorded as the baseline without rewriting anything. Never raises.
    """
    global _last_stat
    try:
        cv = find_cv()
        if cv is None:
            return {"status": "no_cv"}
        st = cv.stat()
        stat = (str(cv), st.st_size, st.st_mtime)
        if not force and stat == _last_stat:
            return {"status": "unchanged"}
        if not _lock.acquire(blocking=wait):
            return {"status": "busy"}  # another thread is syncing; it reads the old profile meanwhile
        try:
            sha = _sha(cv)
            state = _load_state()
            if not force and not state:
                _save_state(sha256=sha, file=cv.name, status="baseline")
                _last_stat = stat
                return {"status": "baseline"}
            if not force and state.get("sha256") == sha:
                _last_stat = stat
                return {"status": "unchanged"}
            logger.info("CV changed (%s) - updating the profile from it", cv.name)
            result = sync_now(cv, sha)
            _last_stat = stat
            return result
        finally:
            _lock.release()
    except Exception as exc:
        logger.warning("CV sync check failed: %s", exc)
        return {"status": "error", "error": str(exc)}
