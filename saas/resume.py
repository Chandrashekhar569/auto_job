"""
Resume -> job-apply skill.

When a user uploads a resume we:
  1. extract its text (PDF / DOCX) and parse the standard CV fields (reusing CVParser),
  2. ask the LLM to turn it into an apply "skill": what to search for on Naukri, which
     titles fit, which jobs to avoid, the skills to match on, and standard answers
     for screening questions. The user can edit every part of it in the dashboard.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
from pathlib import Path

from sqlalchemy import select

from saas.config import resume_dir
from saas.db import Resume, session_scope

logger = logging.getLogger("saas.resume")

ALLOWED = {".pdf", ".docx"}

_SKILL_SYSTEM = "You design job-search strategies from resumes. Output ONLY valid JSON, no markdown."

_SKILL_PROMPT = """Build a job-application skill for this candidate from their resume. It drives a bot that searches
Naukri.com and applies only to jobs that fit. Be specific to THIS resume; do not invent facts.

RESUME:
---
{cv}
---

Return exactly this JSON:
{{
  "headline": "one-line Naukri resume headline, max 200 chars, from the resume",
  "target_titles": ["3-6 job titles this person should apply for, most relevant first"],
  "search_keywords": ["4-8 Naukri search phrases, e.g. 'python developer', 'ml engineer'"],
  "core_skills": ["6-12 skills the resume shows strongly; jobs must ask for some of these"],
  "secondary_skills": ["other skills on the resume"],
  "avoid_keywords": ["title words for jobs NOT to apply to, e.g. 'sales', 'senior director' if too senior"],
  "experience_years": <number, total years of work experience from the resume, 0 for freshers>,
  "seniority": "fresher | junior | mid | senior | lead",
  "preferred_locations": ["cities from the resume if any"],
  "standard_answers": {{
     "notice_period": "from resume or empty",
     "current_ctc": "from resume or empty",
     "expected_ctc": "from resume or empty",
     "willing_to_relocate": "from resume or empty"
  }}
}}"""


def save_upload(user_id: int, filename: str, data: bytes) -> Path:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED:
        raise ValueError("Upload a PDF or DOCX resume.")
    if len(data) > 8 * 1024 * 1024:
        raise ValueError("Resume is larger than 8 MB.")
    folder = resume_dir(user_id)
    shutil.rmtree(folder, ignore_errors=True)   # one resume per user: the parser picks the only file
    folder.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._ -]+", "_", Path(filename).name)[:120] or f"resume{ext}"
    path = folder / safe
    path.write_bytes(data)
    return path


def _json_from(raw: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    if "{" in text:
        text = text[text.find("{"):text.rfind("}") + 1]
    return json.loads(text)


def _fallback_skill(profile: dict) -> dict:
    titles = [t for t in profile.get("job_titles") or [] if t]
    skills = [s for s in profile.get("skills") or [] if s]
    return {
        "headline": profile.get("summary", "")[:200],
        "target_titles": titles[:6],
        "search_keywords": (profile.get("search_keywords") or titles or skills[:3])[:8],
        "core_skills": skills[:12],
        "secondary_skills": skills[12:30],
        "avoid_keywords": [],
        "experience_years": profile.get("experience_years") or 0,
        "seniority": "",
        "preferred_locations": [profile["location"]] if profile.get("location") else [],
        "standard_answers": {
            "notice_period": profile.get("notice_period") or "",
            "current_ctc": profile.get("current_ctc") or "",
            "expected_ctc": profile.get("expected_ctc") or "",
            "willing_to_relocate": "",
        },
    }


def build_skill(raw_text: str, profile: dict) -> dict:
    from autojob.core.llm_router import fast_chat

    try:
        raw = fast_chat(prompt=_SKILL_PROMPT.format(cv=raw_text[:12000]), task_type="general",
                        system_prompt=_SKILL_SYSTEM, temperature=0.1, max_tokens=4096)
        skill = _json_from(raw)
        if not isinstance(skill, dict) or not skill.get("search_keywords"):
            raise ValueError("no search keywords")
    except Exception as exc:
        logger.warning("LLM skill build failed (%s); using parsed CV fields", exc)
        skill = _fallback_skill(profile)
    # Normalise list fields so the UI and the bot can rely on them.
    for key in ("target_titles", "search_keywords", "core_skills", "secondary_skills", "avoid_keywords",
                "preferred_locations"):
        val = skill.get(key) or []
        skill[key] = [str(v).strip() for v in (val if isinstance(val, list) else str(val).split(",")) if str(v).strip()]
    if not isinstance(skill.get("standard_answers"), dict):
        skill["standard_answers"] = {}
    return skill


def process_resume(user_id: int) -> None:
    """Parse the uploaded resume and build the skill. Runs in a background thread."""
    from autojob.jobs.naukri_service import CVParser

    try:
        parser = CVParser(cv_dir=resume_dir(user_id))
        profile = parser.parse()
        if "error" in profile:
            raise RuntimeError(profile["error"])
        skill = build_skill(parser.raw_text, profile)
        with session_scope() as s:
            r = s.scalar(select(Resume).where(Resume.user_id == user_id))
            if r is None:
                return
            r.raw_text = parser.raw_text
            r.profile_json = json.dumps(profile, ensure_ascii=False)
            r.skill_json = json.dumps(skill, ensure_ascii=False)
            r.status, r.error = "ready", ""
        logger.info("Resume processed for user %s: %s", user_id, skill.get("search_keywords"))
    except Exception as exc:
        logger.exception("Resume processing failed for user %s", user_id)
        with session_scope() as s:
            r = s.scalar(select(Resume).where(Resume.user_id == user_id))
            if r is not None:
                r.status, r.error = "error", str(exc)[:500]


def process_resume_async(user_id: int) -> None:
    threading.Thread(target=process_resume, args=(user_id,), daemon=True, name=f"resume-{user_id}").start()
