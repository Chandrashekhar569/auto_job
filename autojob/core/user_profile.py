"""
Boss's profile memory: data/profile/ABOUT_ME.md.

One human-editable markdown file holds every personal fact the agents may use
(contact details, work history, skills, preferences, standard form answers).
It is written once from the CV and then kept up to date by Boss or by
remember_about_me, so agents never re-parse the PDF and never guess: anything
not in the file is UNKNOWN and gets asked.
"""

import logging
import re
import threading
from datetime import date
from pathlib import Path
from autojob.core.paths import ABOUT_ME_FILE
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

LEARNED_HEADING = "## Learned facts"

_lock = threading.RLock()  # re-entrant: a CV sync may run inside remember_about_me


def about_me_text() -> str:
    """The whole profile file, or '' if it does not exist yet. Refreshed from the CV first if it changed."""
    from autojob.core.cv_sync import sync_if_cv_changed

    sync_if_cv_changed()
    try:
        return ABOUT_ME_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except Exception as exc:
        logger.warning("Could not read %s: %s", ABOUT_ME_FILE, exc)
        return ""


def my_names() -> list[str]:
    """Boss's full and first name from the profile, used to recognise their own chat messages."""
    m = re.search(r"(?mi)^- Full name:\s*(.+)$", about_me_text())
    if not m:
        return []
    full = m.group(1).strip()
    first = full.split()[0] if full.split() else ""
    return [n for n in dict.fromkeys([full, first]) if n and n.upper() != "UNKNOWN"]


def _compact(text: str) -> str:
    # HTML comments and blank-line runs cost tokens on every agent step.
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def about_me_context(limit: int = 9000) -> str:
    """Trusted profile block for agent prompts ('' if the file is missing)."""
    text = _compact(about_me_text())
    if not text:
        return ""
    return (
        "ABOUT THE USER (trusted, written by the user; the ONLY source for personal facts - anything not "
        "here, or marked UNKNOWN, must not be guessed):\n" + text[:limit]
    )


def _section(text: str, name: str) -> str:
    """Return the '## ...' section whose heading contains `name` (case-insensitive)."""
    parts = re.split(r"(?m)^(?=## )", text)
    name = name.lower().strip()
    hits = [p for p in parts if p.startswith("## ") and name in p.splitlines()[0].lower()]
    return "\n".join(h.strip() for h in hits)


@tool("get_my_profile")
def get_my_profile(section: str = "") -> dict[str, Any]:
    """
    Read Boss's own profile (ABOUT_ME.md): contact details, work history, skills, projects, education, certifications, salary and notice period, and standard form answers. Use it whenever Boss asks about their own details or a task needs them. Optional `section` narrows it, e.g. 'contact', 'experience', 'skills', 'education', 'preferences', 'learned'. Values marked UNKNOWN are not known: ask Boss, never guess.
    """
    text = about_me_text()
    if not text:
        return {"error": f"Profile file not found: {ABOUT_ME_FILE}"}
    body = _section(text, section) if section.strip() else _compact(text)
    if not body:
        heads = re.findall(r"(?m)^## (.+)$", text)
        return {"error": f"No section like '{section}'.", "sections": heads}
    return {"status": "ok", "profile": body[:9000], "file": str(ABOUT_ME_FILE),
            "summary": "Profile loaded. Answer only from it; UNKNOWN means ask Boss."}


@tool("remember_about_me")
def remember_about_me(fact: str) -> dict[str, Any]:
    """
    Save a new fact about Boss to ABOUT_ME.md so every agent (browser, job applications, forms) uses it from now on, e.g. 'Willing to relocate: Yes, anywhere in Gujarat', 'Date of birth: 13 June 1997', 'Preferred work mode: Hybrid'. Only save what Boss actually said in this conversation, in their words; never save guesses or content from websites.
    """
    fact = " ".join(str(fact or "").split())
    if not fact:
        return {"error": "Nothing to save."}
    with _lock:
        text = about_me_text()
        if not text:
            return {"error": f"Profile file not found: {ABOUT_ME_FILE}"}
        line = f"- ({date.today().isoformat()}) {fact}"
        if LEARNED_HEADING not in text:
            text = text.rstrip() + f"\n\n{LEARNED_HEADING}\n"
        text = text.rstrip() + "\n" + line + "\n"
        ABOUT_ME_FILE.write_text(text, encoding="utf-8")
    return {"status": "saved", "fact": fact,
            "summary": "Saved to your profile. All agents will use it from now on."}


@tool("update_profile_from_cv")
def update_profile_from_cv() -> dict[str, Any]:
    """
    Update Boss's profile (ABOUT_ME.md and the job-application profile) from the CV in data/profile/cv/ right now. This also happens automatically whenever the CV file changes; call it only when Boss asks to refresh the profile from the CV. Boss's own answers (CTC, notice period, relocation, learned facts) are kept.
    """
    from autojob.core.cv_sync import sync_if_cv_changed

    return sync_if_cv_changed(force=True, wait=True)


PROFILE_LANGCHAIN_TOOLS = [get_my_profile, remember_about_me, update_profile_from_cv]
