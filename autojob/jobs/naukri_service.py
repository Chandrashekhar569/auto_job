"""
Naukri.com Auto Job Apply Service for Veronica AI.

Automates job applications on Naukri.com using:
  - Playwright with Edge browser (existing user profile)
  - CV parsing + LLM-powered profile extraction
  - AI job matching & scoring
  - Smart screening form filling
  - Continuous application with anti-detection

Exposed as Gemini Live API function declarations.
"""

import json
import logging
import os
import random
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from autojob.core.paths import CV_DIR, DATA_DIR
from typing import Any, Optional
from urllib.parse import quote

from playwright.sync_api import sync_playwright, Page, BrowserContext, TimeoutError as PlaywrightTimeout
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------

RESULTS_FILE = DATA_DIR / "naukri_results.json"

# Edge profile — use a DEDICATED profile for automation (not the user's real profile)
# This avoids killing the user's actual browser and prevents profile conflicts.
# The user should log into Naukri.com once in this profile, then it persists.
EDGE_USER_DATA = os.environ.get("AUTOJOB_EDGE_PROFILE") or os.path.join(os.environ.get("LOCALAPPDATA", ""), "VeronicaEdgeProfile")

DEFAULT_MAX_APPLICATIONS = 25
MATCH_THRESHOLD = 60  # Minimum ATS score (0-100) to apply
NAUKRI_BASE_URL = "https://www.naukri.com"

# Where to look when Boss names no city: the main tech hubs, then all of India
# ("" = no location filter). The CV's home city is never used as the search
# location - that kept every run inside Surat.
TARGET_CITIES = ["Bengaluru", "Hyderabad", "Pune", "Noida", "New Delhi"]
_PAN_INDIA = re.compile(r"^\s*(pan[\s-]*india|all[\s-]*india|india|anywhere|any|all|remote)\s*$", re.I)


def search_locations(location: str = "") -> list[str]:
    """
    The location filters to search, in order. Empty -> the main cities plus
    all-India; "pan india" / "all india" -> all-India only; "Pune, Noida" ->
    those cities.
    """
    location = (location or "").strip()
    if not location:
        return [*TARGET_CITIES, ""]
    if _PAN_INDIA.match(location):
        return [""]
    parts = [p.strip() for p in re.split(r"[,/;|]|\band\b", location) if p.strip()]
    out: list[str] = []
    for p in parts:
        loc = "" if _PAN_INDIA.match(p) else p
        if loc not in out:
            out.append(loc)
    return out or [""]


def interleave_jobs(lists: list[list[dict]]) -> list[dict]:
    """Round-robin the per-city results (so one city cannot use up the run), dropping repeat URLs."""
    out, seen = [], set()
    for i in range(max((len(x) for x in lists), default=0)):
        for jobs in lists:
            if i < len(jobs):
                url = jobs[i].get("url", "")
                if url and url in seen:
                    continue
                seen.add(url)
                out.append(jobs[i])
    return out

# Human-like delay ranges (seconds)
SHORT_DELAY = (1.5, 3.0)
MEDIUM_DELAY = (3.0, 5.0)
LONG_DELAY = (5.0, 8.0)


def _human_delay(delay_range: tuple = SHORT_DELAY):
    """Add a random human-like delay."""
    time.sleep(random.uniform(*delay_range))


def naukri_search_url(keywords: str, location: str = "", experience: Any = None, page: int = 1) -> str:
    """
    The regular search-results page. (/jobapi/v3/search is Naukri's internal
    JSON API: it needs private appid/systemid headers and just returns "Please
    provide the valid App Id and SystemId in Header" in the browser.)
    Page 2+ adds "-2", "-3"... to the path, the way Naukri's own pager does.
    """
    if experience is not None:
        try:
            experience = int(float(experience))
        except (TypeError, ValueError):
            experience = None
    params = {"k": keywords, "l": location, "experience": experience, "jobAge": 7}  # last 7 days
    params = {k: v for k, v in params.items() if v is not None and v != ""}

    def _slug(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")

    path = f"{_slug(keywords) or 'all'}-jobs"
    if location:
        path += f"-in-{_slug(location)}"
    if page and int(page) > 1:
        path += f"-{int(page)}"
    return f"{NAUKRI_BASE_URL}/{path}?" + "&".join(f"{k}={quote(str(v))}" for k, v in params.items())


# ---------------------------------------------------------------------------
# CV Parser
# ---------------------------------------------------------------------------

class CVParser:
    """Parse CV/Resume files and extract structured candidate profile using LLM."""

    def __init__(self, cv_dir: Path = CV_DIR):
        self.cv_dir = cv_dir
        self._raw_text: str = ""
        self._profile: dict = {}

    def find_cv_file(self) -> Optional[Path]:
        """Find the first PDF or DOCX file in the CV directory."""
        if not self.cv_dir.exists():
            logger.error("CV directory not found: %s", self.cv_dir)
            return None

        for ext in ["*.pdf", "*.PDF", "*.docx", "*.DOCX"]:
            files = list(self.cv_dir.glob(ext))
            if files:
                logger.info("Found CV file: %s", files[0].name)
                return files[0]
        return None

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF using PyMuPDF (primary) or PyPDF2 (fallback)."""
        # Try PyMuPDF first (fast and accurate)
        try:
            import pymupdf as fitz  # legacy top-level `fitz` name is deprecated
            text = ""
            with fitz.open(str(pdf_path)) as doc:
                for page in doc:
                    text += page.get_text("text") + "\n"
            if text.strip():
                logger.info("PDF text extracted via PyMuPDF (%d chars)", len(text))
                return text
        except ImportError:
            logger.info("PyMuPDF not available, trying PyPDF2...")
        except Exception as exc:
            logger.warning("PyMuPDF extraction failed: %s, trying fallback...", exc)

        # Fallback to PyPDF2/pypdf
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(pdf_path))
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            text = "\n".join(text_parts)
            if text.strip():
                logger.info("PDF text extracted via PyPDF2 (%d chars)", len(text))
                return text
        except ImportError:
            logger.warning("PyPDF2 also not available")
        except Exception as exc:
            logger.error("PyPDF2 extraction also failed: %s", exc)

        return ""

    def extract_text_from_docx(self, docx_path: Path) -> str:
        """Extract text from DOCX file."""
        try:
            from docx import Document
            doc = Document(str(docx_path))
            parts = []
            for para in doc.paragraphs:
                if para.text.strip():
                    parts.append(para.text.strip())
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(
                        cell.text.strip() for cell in row.cells if cell.text.strip()
                    )
                    if row_text:
                        parts.append(row_text)
            return "\n".join(parts)
        except Exception as exc:
            logger.error("DOCX extraction failed: %s", exc)
            return ""

    def parse(self) -> dict:
        """
        Parse the CV file and extract a structured profile using LLM.
        Returns a dict with skills, experience, job titles, etc.
        """
        cv_file = self.find_cv_file()
        if cv_file is None:
            return {"error": "No CV file found in CV/ directory. Please add your resume (PDF or DOCX)."}

        # Extract raw text
        if cv_file.suffix.lower() == ".pdf":
            self._raw_text = self.extract_text_from_pdf(cv_file)
        elif cv_file.suffix.lower() == ".docx":
            self._raw_text = self.extract_text_from_docx(cv_file)
        else:
            return {"error": f"Unsupported file format: {cv_file.suffix}"}

        if not self._raw_text.strip():
            return {"error": "Could not extract text from CV file. The file may be image-based."}

        # Use LLM to extract structured profile
        self._profile = self._extract_profile_with_llm(self._raw_text)
        return self._profile

    def _extract_profile_with_llm(self, cv_text: str) -> dict:
        """Send CV text to LLM and get structured profile data."""
        from autojob.core.llm_router import fast_chat

        prompt = f"""Analyze this resume/CV text and extract a structured JSON profile.
Return ONLY valid JSON with NO markdown formatting, no backticks, no explanation.

Resume Text:
---
{cv_text[:4000]}
---

Return this exact JSON structure (fill in from the resume):
{{
    "name": "candidate full name",
    "email": "email if found or null",
    "phone": "phone if found or null",
    "skills": ["skill1", "skill2", "skill3"],
    "experience_years": 0,
    "job_titles": ["most relevant job title 1", "job title 2"],
    "search_keywords": ["keyword1 for job search", "keyword2", "keyword3"],
    "education": "highest degree and field",
    "current_ctc": "if mentioned or null",
    "expected_ctc": "if mentioned or null",
    "notice_period": "if mentioned or 'Immediate'",
    "location": "current city if mentioned or null",
    "summary": "2-3 line professional summary"
}}"""

        try:
            response = fast_chat(
                prompt=prompt,
                task_type="general",
                system_prompt="You are a resume parser. Extract information accurately. Return ONLY valid JSON.",
                temperature=0.1,
                # Reasoning models spend tokens before the JSON; 1024 truncated it.
                max_tokens=4096,
            )

            # Clean the response — remove markdown code blocks if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
                cleaned = re.sub(r"\n?```\s*$", "", cleaned)
            # Reasoning models may wrap the JSON in prose; keep the outermost object.
            cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.S)
            if not cleaned.lstrip().startswith("{") and "{" in cleaned and "}" in cleaned:
                cleaned = cleaned[cleaned.find("{"):cleaned.rfind("}") + 1]

            profile = json.loads(cleaned)
            logger.info("Extracted CV profile: %s", json.dumps(profile, indent=2)[:500])
            return profile
        except json.JSONDecodeError as e:
            logger.error("LLM returned invalid JSON: %s | Response: %s", e, response[:300])
            # Return a basic fallback profile from regex
            return self._fallback_extract(cv_text)
        except Exception as exc:
            logger.error("LLM profile extraction failed: %s", exc)
            return self._fallback_extract(cv_text)

    def _fallback_extract(self, text: str) -> dict:
        """Fallback regex-based extraction if LLM fails."""
        emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
        phones = re.findall(r'(?:\+?91[\s-]?)?[6-9]\d{9}', text)

        # Extract skills by common tech keywords
        common_skills = [
            "Python", "Java", "JavaScript", "React", "Node.js", "Angular", "Vue",
            "Django", "Flask", "Spring", "SQL", "MongoDB", "AWS", "Azure", "Docker",
            "Kubernetes", "Git", "HTML", "CSS", "TypeScript", "C++", "C#", ".NET",
            "Machine Learning", "Deep Learning", "Data Science", "AI", "REST API",
            "Microservices", "Linux", "DevOps", "CI/CD", "Agile", "Scrum",
        ]
        found_skills = [s for s in common_skills if s.lower() in text.lower()]

        return {
            "name": "Candidate",
            "email": emails[0] if emails else None,
            "phone": phones[0] if phones else None,
            "skills": found_skills[:10],
            "experience_years": 0,
            "job_titles": ["Software Developer"],
            "search_keywords": found_skills[:3] if found_skills else ["developer"],
            "education": None,
            "current_ctc": None,
            "expected_ctc": None,
            "notice_period": "Immediate",
            "location": None,
            "summary": "Software professional",
        }

    @property
    def raw_text(self) -> str:
        return self._raw_text

    @property
    def profile(self) -> dict:
        return self._profile


# ---------------------------------------------------------------------------
# Job Matcher (LLM-powered scoring)
# ---------------------------------------------------------------------------

class JobMatcher:
    """Score how well a job listing matches the candidate's CV profile."""

    def __init__(self, cv_profile: dict, cv_raw_text: str = ""):
        self.cv_profile = cv_profile
        self.cv_raw_text = cv_raw_text[:2000]  # Truncate for prompt length

    @staticmethod
    def _norm(text: str) -> str:
        return " " + re.sub(r"[^a-z0-9+#.]+", " ", str(text).lower()).strip() + " "

    def keyword_coverage(self, keywords: list[str]) -> Optional[tuple[int, list[str], list[str]]]:
        """
        ATS keyword match: the share of the job's key skills that literally
        appear in the CV / profile skills. Returns (percent, matched, missing),
        or None when there are no keywords to check.
        """
        cv = self._norm(self.cv_raw_text + " " + " ".join(self.cv_profile.get("skills", []) or []))
        keywords = list(dict.fromkeys(str(k).strip() for k in keywords or [] if str(k).strip()))[:25]
        if not keywords:
            return None
        matched = [k for k in keywords if self._norm(k) in cv]
        missing = [k for k in keywords if k not in matched]
        return round(100 * len(matched) / len(keywords)), matched, missing

    def score_job(self, job_title: str, company: str, job_description: str) -> dict:
        """
        ATS check of a job against the CV: the model's fit score (skills,
        experience, role) blended with a literal keyword match of the job's key
        skills against the CV text. Returns {"score": 0-100, "reasoning",
        "should_apply", "matched_keywords", "missing_keywords"}. If scoring
        fails the job is skipped, never applied to blind.
        """
        from autojob.core.llm_router import fast_chat

        skills_str = ", ".join(self.cv_profile.get("skills", []))
        titles_str = ", ".join(self.cv_profile.get("job_titles", []) or [self.cv_profile.get("current_title") or ""])
        exp = self.cv_profile.get("experience_years", 0)

        prompt = f"""Act as an ATS (applicant tracking system). Score how well this CV fits the job (0-100).
Return ONLY valid JSON with no markdown.

CANDIDATE PROFILE:
- Skills: {skills_str}
- Target Roles: {titles_str}
- Experience: {exp} years
- Summary: {self.cv_profile.get('summary', 'N/A')}

CV TEXT (start):
{self.cv_raw_text[:1500] or '(unavailable)'}

JOB LISTING:
- Title: {job_title}
- Company: {company}
- Description: {job_description[:2500]}

Score like an ATS: must-have skills present, years of experience inside the job's range, role/title fit.
A job asking for clearly more experience than the candidate has, or a different field, scores below 50.
"keywords": the 5-15 key skills/tools the job asks for, each 1-3 words as written in the job.

Return JSON: {{"score": <0-100>, "keywords": ["..."], "reasoning": "<one line why>"}}"""

        try:
            response = fast_chat(
                prompt=prompt,
                task_type="general",
                system_prompt="You are a strict ATS resume screener. Be realistic. Return ONLY valid JSON.",
                temperature=0.1,
                max_tokens=1024,
            )
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
                cleaned = re.sub(r"\n?```\s*$", "", cleaned)
            # Reasoning models may wrap the JSON in prose; keep the outermost object.
            cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.S)
            if not cleaned.lstrip().startswith("{") and "{" in cleaned and "}" in cleaned:
                cleaned = cleaned[cleaned.find("{"):cleaned.rfind("}") + 1]
            result = json.loads(cleaned)
            fit = max(0, min(100, int(float(result.get("score") or 0))))
        except Exception as exc:
            logger.warning("ATS scoring failed, skipping the job: %s", exc)
            return {"score": 0, "reasoning": "ATS check failed - not applied", "should_apply": False}

        score, reasoning = fit, str(result.get("reasoning") or "")
        coverage = self.keyword_coverage(result.get("keywords") or [])
        out: dict = {}
        if coverage:
            pct, matched, missing = coverage
            score = round(0.6 * fit + 0.4 * pct)
            reasoning = f"ATS {score} (fit {fit}, keywords {pct}%). {reasoning}"
            if missing:
                reasoning += " Missing: " + ", ".join(missing[:6])
            out = {"matched_keywords": matched, "missing_keywords": missing}
        return {"score": score, "reasoning": reasoning, "should_apply": score >= MATCH_THRESHOLD, **out}


# ---------------------------------------------------------------------------
# Screening Form Filler (LLM-powered)
# ---------------------------------------------------------------------------

class ScreeningFormFiller:
    """
    Answers screening questions strictly from the candidate's own data: the CV,
    candidate_profile.json and answers the user gave Veronica before
    (data/job_agent/memories/answers.md).

    Every LLM answer must come with evidence — short quotes copied from that
    data — and the quotes are checked against the real text. Anything that is
    not clearly supported returns None: nothing is typed, the question is saved
    to questions_waiting.md, and the job is handed back to the user instead of
    being answered with a guess.
    """

    _NUMERIC_Q = re.compile(r"\b(how many|how much|number of|years?|months?|days?|ctc|salary|lpa|lakhs?|rate)\b", re.I)
    _META_START = re.compile(
        r"^(the user|the candidate wants|based on|as an ai|i (?:think|would|will|need)|okay|let me|we need|"
        r"here is|sure|according to)", re.I)

    def __init__(self, cv_profile: dict, cv_raw_text: str = ""):
        self.cv_profile = cv_profile
        self.cv_raw_text = cv_raw_text or ""
        self.unanswered: list[str] = []
        self._cache: dict[str, Optional[str]] = {}

    # -- the candidate's data ---------------------------------------------

    def _profile_facts(self) -> dict:
        skip = {"search_keywords", "job_titles"}
        return {k: v for k, v in self.cv_profile.items()
                if not str(k).startswith("_") and k not in skip and v not in (None, "", [], {})}

    @staticmethod
    def _user_answers() -> str:
        try:
            from autojob.jobs.job_deep_agent import MEM_ROOT
            path = MEM_ROOT / "answers.md"
            return path.read_text(encoding="utf-8") if path.exists() else ""
        except Exception:
            return ""

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"[^a-z0-9+#.%]+", " ", str(text).lower()).strip()

    def _corpus(self) -> str:
        return self._norm(" ".join([
            self.cv_raw_text,
            json.dumps(self._profile_facts(), ensure_ascii=False),
            self._user_answers(),
        ]))

    def _quote_ok(self, quote: str, corpus: str) -> bool:
        q = self._norm(quote)
        if len(q) < 2:
            return False
        if q in corpus:
            return True
        # Tolerate PDF line breaks / bullet glyphs: most of the quote's words,
        # in a short window of the text.
        words = q.split()
        if len(words) < 3:
            return False
        hits = sum(1 for w in words if f" {w} " in f" {corpus} ")
        return hits / len(words) >= 0.85

    # -- answering --------------------------------------------------------

    def _deterministic(self, question: str) -> Optional[str]:
        q = question.lower()
        exp = self.cv_profile.get("experience_years")
        # Only plain total-experience questions ("How many years of experience do
        # you have?", "Total work experience?"). Anything naming a skill, tool,
        # domain or "relevant" experience goes to the evidence-checked path.
        if exp not in (None, "") and re.search(r"\bexperience\b", q) \
                and re.search(r"\b(total|overall|how many years)\b", q) \
                and not re.search(r"\b(in|with|on|using|of working|as an?|relevant|related|domain|field)\b", q):
            try:
                return str(int(float(exp))) if float(exp).is_integer() else str(exp)
            except (TypeError, ValueError):
                return None
        return None

    def skill_years(self, question: str) -> Optional[str]:
        """
        "Years of experience in <X>?" from Boss's data (skill_years.py): his
        table, 0 for a skill not in the data, his GenAI figure for AI tools on the
        CV, else the dated roles that mention <X>. None = not that kind of question.
        """
        from autojob.core.user_profile import about_me_text
        from autojob.jobs.skill_years import answer_skill_years

        try:
            about = about_me_text()
        except Exception:
            about = ""
        return answer_skill_years(question, about, self.cv_raw_text + " " + json.dumps(self._profile_facts()),
                                  self._user_answers())

    def answer_question(self, question_text: str, options: list[str] = None) -> Optional[str]:
        """
        Answer from the candidate's data only. Returns the answer text (for
        options: the exact option text) or None when the data does not support
        an answer.
        """
        question = " ".join(str(question_text or "").split())
        options = [o for o in (options or []) if o and o.strip()]
        key = question.lower() + "|" + "|".join(options).lower()
        if key in self._cache:
            return self._cache[key]

        if options:
            answer = None
            years = self.skill_years(question)
            if years is not None:
                from autojob.jobs.live_questions import _match_option

                answer = _match_option(years, options)
                if answer is None and float(years) == 0:
                    answer = next((o for o in options if re.search(r"\b(no|none|nil|fresher|0)\b|less than 1|<\s*1",
                                                                     o, re.I)), None)
        else:
            answer = self._deterministic(question) or self.skill_years(question)
        if answer is None:
            answer = self._answer_with_evidence(question, options)
        if answer is None:
            # Not in the data: ask Boss live instead of dropping the job.
            from autojob.jobs.live_questions import ask_boss

            answer = ask_boss(question, options)
        self._cache[key] = answer
        if answer is None:
            self._remember_unanswered(question, options)
        return answer

    # Shared with the batch answerer for LinkedIn / Indeed (screening_answers.py).
    _EVIDENCE_RULES = """- evidence: 1-6 short quotes copied word for word from the CV / profile / answers that prove the answer.
- Numbers (years, months, counts, days, CTC in lakhs): digits only in "answer", e.g. "3".
- Counting questions (e.g. how many projects): count only items the CV explicitly lists; put each counted item's
  name in evidence.
- Years with a specific skill/tool: only if the CV gives it, or the skill is named inside roles whose dates you can
  add up (quote those roles with dates). If the skill is not in the data -> answerable=false.
- Preferences (relocation, shifts, notice period, CTC, joining date) only from the profile or the answers given before.
- "answer" is the text to type or the option to pick: short, first person, no explanations, no reasoning."""

    def _data_block(self) -> str:
        facts = json.dumps(self._profile_facts(), ensure_ascii=False, indent=1)[:3000]
        answers = self._user_answers()[:2500] or "(none)"
        cv = self.cv_raw_text[:16000] or "(CV text unavailable)"
        return (f"CANDIDATE PROFILE (written by the candidate):\n{facts}\n\n"
                f"ANSWERS THE CANDIDATE GAVE BEFORE:\n{answers}\n\nCV TEXT:\n{cv}")

    def _answer_with_evidence(self, question: str, options: list[str]) -> Optional[str]:
        from autojob.core.llm_router import fast_chat

        opts = ("\nOPTIONS (the answer must be exactly one of these):\n"
                + "\n".join(f"- {o}" for o in options)) if options else ""
        prompt = f"""Answer a job-application screening question for the candidate using ONLY the data below.
Never guess, estimate, round up or invent anything. If the data does not clearly state the answer, it is not answerable.

{self._data_block()}

QUESTION: {question}{opts}

Reply with ONLY one JSON object, no other text:
{{"answerable": true or false, "answer": "<the answer>", "evidence": ["<short exact quote copied from the data above>", ...]}}

Rules:
{self._EVIDENCE_RULES}"""

        try:
            raw = fast_chat(
                prompt=prompt,
                task_type="general",
                system_prompt="You fill job applications truthfully from the candidate's own data. Output only JSON.",
                temperature=0.0,
                max_tokens=1500,  # reasoning models think before the JSON
            )
        except Exception as exc:
            logger.error("Screening answer failed for '%s': %s", question[:80], exc)
            return None

        return self._verify_answer(question, options, self._parse_json(raw))

    def _verify_answer(self, question: str, options: list[str], data: Optional[dict],
                       max_len: int = 300) -> Optional[str]:
        """
        Accept a model answer only when its evidence quotes are really in the
        candidate's data (and, for options, it is one of them). Else None.
        """
        if not data or not data.get("answerable"):
            logger.info("Not answerable from CV/profile: %s", question[:100])
            return None

        answer = " ".join(str(data.get("answer") or "").split())
        evidence = [str(e) for e in (data.get("evidence") or []) if str(e).strip()]
        if not answer or len(answer) > max_len or self._META_START.match(answer):
            logger.warning("Rejected malformed answer for '%s': %s", question[:80], answer[:100])
            return None

        corpus = self._corpus()
        verified = [e for e in evidence if self._quote_ok(e, corpus)]
        if not verified or len(verified) < max(1, len(evidence) // 2 + 1):
            logger.warning("Rejected unverified answer for '%s': %r (evidence %r)", question[:80], answer, evidence[:3])
            return None

        if options:
            match = next((o for o in options if o.strip().lower() == answer.lower()), None) or \
                next((o for o in options if answer.lower() in o.lower() or o.lower() in answer.lower()), None)
            if match is None:
                logger.warning("Answer %r is not one of the options for '%s'", answer, question[:80])
                return None
            answer = match
        elif self._NUMERIC_Q.search(question) and re.fullmatch(r"\d+(\.\d+)?", answer):
            # A number must be traceable: stated in the evidence, the count of
            # listed items, or computed from dated roles.
            ev_text = " ".join(verified)
            if not (re.search(rf"\b{re.escape(answer)}\b", ev_text)
                    or int(float(answer)) == len(verified)
                    or len(re.findall(r"\b(19|20)\d{2}\b", ev_text)) >= 2):
                logger.warning("Rejected untraceable number %s for '%s'", answer, question[:80])
                return None

        logger.info("Screening answer (verified): '%s' -> '%s' | evidence: %s",
                    question[:60], answer[:60], "; ".join(v[:50] for v in verified[:3]))
        return answer

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict]:
        text = re.sub(r"<think>.*?(</think>|$)", "", raw or "", flags=re.S)
        text = re.sub(r"```(?:json)?", "", text)
        # The JSON is the last object in the reply (reasoning text may come first).
        for m in reversed(list(re.finditer(r"\{", text))):
            end = text.rfind("}")
            if end <= m.start():
                continue
            try:
                obj = json.loads(text[m.start():end + 1])
                if isinstance(obj, dict) and "answerable" in obj:
                    return obj
            except Exception:
                continue
        return None

    def _remember_unanswered(self, question: str, options: list[str]) -> None:
        if question and question not in self.unanswered:
            self.unanswered.append(question)
        try:
            from autojob.jobs.job_deep_agent import note_unanswered
            note_unanswered(question, options)
        except Exception as exc:
            logger.debug("Could not save unanswered question: %s", exc)


# ---------------------------------------------------------------------------
# Naukri Job Bot (Playwright + Edge)
# ---------------------------------------------------------------------------

class NaukriJobBot:
    """Automates job search and application on Naukri.com using Playwright + Edge."""

    def __init__(self):
        self.playwright = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self._is_running = False
        self._should_stop = False
        self._session_results: list[dict] = []
        self._status = "idle"
        self._lock = threading.Lock()
        self._unanswered_question = ""

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @status.setter
    def status(self, value: str):
        with self._lock:
            self._status = value

    @property
    def should_stop(self) -> bool:
        with self._lock:
            return self._should_stop

    def request_stop(self):
        with self._lock:
            self._should_stop = True

    def _close_edge_processes(self):
        """Kill only automation Edge processes to release the profile lock.

        Note: Since we use a dedicated profile (VeronicaEdgeProfile), we only
        need to close processes using THAT specific profile, not all Edge instances.
        """
        # The general browser tools (browser_service) share this profile. Ask
        # that session to shut down cleanly instead of killing its Edge.
        try:
            from autojob.integrations import browser_service
            browser_service._get_session().stop()
        except Exception as exc:
            logger.warning("Could not stop the shared browser session: %s", exc)

        # Kill only leftover Edge processes launched with our profile (e.g. from
        # a crashed run). Matching by window title used to kill the user's own
        # Edge windows — and the previous run's browser — whenever "Naukri"
        # appeared in a title.
        from autojob.integrations.browser_service import kill_profile_edge

        kill_profile_edge(EDGE_USER_DATA)

    def _validate_edge_profile(self):
        """Verify the Edge User Data directory exists and has the Default profile."""
        edge_path = Path(EDGE_USER_DATA)

        if not edge_path.exists():
            raise FileNotFoundError(
                f"Edge User Data not found at {EDGE_USER_DATA}. "
                "Make sure Microsoft Edge is installed and you've logged into Naukri.com."
            )

        default_profile = edge_path / "Default"
        if not default_profile.exists():
            raise FileNotFoundError(
                f"Edge Default profile not found at {default_profile}. "
                "Open Edge, log into Naukri.com, then try again."
            )

        # Check for Cookies file (confirms login data exists)
        cookies_file = default_profile / "Cookies"
        if not cookies_file.exists():
            # Also check Network/Cookies (newer Edge versions)
            cookies_file = default_profile / "Network" / "Cookies"

        if cookies_file.exists():
            logger.info("Edge profile validated. Cookies file found at: %s", cookies_file)
        else:
            logger.warning("Cookies file not found in Edge profile — login may not persist.")

    def start(self):
        """Launch Edge browser using the REAL Edge profile (preserving login sessions)."""
        if self._is_running:
            return

        self.status = "starting"
        self._should_stop = False
        self._session_results = []

        # Validate Edge profile exists
        try:
            self._validate_edge_profile()
        except Exception as exc:
            self.status = "error"
            raise RuntimeError(f"Profile validation failed: {exc}") from exc

        # Close running Edge instances to release the profile lock
        self._close_edge_processes()

        try:
            self.playwright = sync_playwright().start()
            self.context = self.playwright.chromium.launch_persistent_context(
                EDGE_USER_DATA,  # Use the REAL Edge User Data directory
                channel="msedge",
                headless=False,
                chromium_sandbox=True,  # otherwise Playwright adds --no-sandbox
                viewport={"width": 1366, "height": 768},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/128.0.0.0 Safari/537.36 Edg/128.0.0.0"
                ),
                args=[
                    "--profile-directory=Default",
                    "--disable-blink-features=AutomationControlled",
                    "--no-first-run",
                    "--no-default-browser-check",
                ],
            )
            self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
            self._is_running = True
            self.status = "running"
            logger.info("NaukriJobBot browser started with real Edge profile.")
        except Exception as exc:
            self.status = "error"
            self.stop()
            raise RuntimeError(f"Failed to launch Edge browser: {exc}") from exc

    def stop(self):
        """Close the browser and cleanup."""
        self._is_running = False
        self.status = "stopping"
        try:
            if self.context:
                self.context.close()
                self.context = None
            if self.playwright:
                self.playwright.stop()
                self.playwright = None
        except Exception as exc:
            logger.error("Error stopping NaukriJobBot: %s", exc)
        # Veronica's browser opens this same profile next (company-site jobs);
        # an Edge left behind by a failed close would keep it locked.
        try:
            from autojob.integrations.browser_service import kill_profile_edge

            kill_profile_edge(EDGE_USER_DATA)
        except Exception as exc:
            logger.warning("Could not release the Edge profile: %s", exc)
        self.page = None
        self.status = "idle"

    def verify_login(self) -> bool:
        """Check if we're logged into Naukri.com."""
        try:
            self.page.goto(NAUKRI_BASE_URL, wait_until="domcontentloaded", timeout=20000)
            _human_delay(MEDIUM_DELAY)

            # Check for logged-in indicators
            logged_in_selectors = [
                "a[href*='/mnjuser/profile']",
                ".nI-gNb-drawer__icon",
                ".view-profile-node",
                "div.nI-gNb-header__right-wrapper a[title]",
                ".nI-gNb-sb__icon",
            ]

            for selector in logged_in_selectors:
                try:
                    if self.page.locator(selector).first.is_visible(timeout=3000):
                        logger.info("Naukri login verified via selector: %s", selector)
                        return True
                except Exception:
                    continue

            # Fallback: check if login/register buttons are absent
            try:
                login_btn = self.page.locator("a[href*='/login'], a:has-text('Login')")
                if login_btn.count() == 0:
                    logger.info("Naukri login verified (no login button found)")
                    return True
            except Exception:
                pass

            logger.warning("Naukri login verification failed — may not be logged in")
            return False
        except Exception as exc:
            logger.error("Login verification failed: %s", exc)
            return False

    def recommended_jobs(self) -> list[dict]:
        """Jobs Naukri recommends for this profile (the 'Recommended jobs for you' page)."""
        try:
            self.page.goto(f"{NAUKRI_BASE_URL}/mnjuser/recommendedjobs", wait_until="domcontentloaded",
                           timeout=30000)
            _human_delay(MEDIUM_DELAY)
            for _ in range(4):
                self.page.mouse.wheel(0, random.randint(700, 1100))
                _human_delay(SHORT_DELAY)
            jobs = self._job_links_on_page()
            logger.info("Naukri recommended jobs: %d", len(jobs))
            return jobs
        except Exception as exc:
            logger.error("Could not read Naukri recommended jobs: %s", exc)
            return []

    def _job_links_on_page(self) -> list[dict]:
        """Every job on the page, from its /job-listings- link (works on any Naukri list layout)."""
        try:
            found = self.page.evaluate("""() => {
              const out = [], seen = new Set();
              for (const a of document.querySelectorAll('a[href*="/job-listings-"]')) {
                const url = a.href.split('?')[0];
                if (seen.has(url)) continue;
                const title = (a.getAttribute('title') || a.innerText || '').trim().replace(/\\s+/g, ' ');
                if (!title || title.length > 150) continue;
                seen.add(url);
                const card = a.closest('article, [class*="jobTuple"], [class*="tuple"], [class*="Card"], [class*="card"], li') || a.parentElement;
                const comp = card && card.querySelector('a[class*="comp"], [class*="comp-name"], [class*="company"], a.subTitle');
                out.push({title, url, company: comp ? comp.innerText.trim().slice(0, 100) : 'Unknown',
                          experience_range: '', description_snippet: (card ? card.innerText : '').slice(0, 500)});
              }
              return out;
            }""") or []
        except Exception as exc:
            logger.debug("Job link scan failed: %s", exc)
            return []
        return [j for j in found if j.get("title") and j.get("url")]

    def search_jobs(self, keywords: str, location: str = "", experience: int = None, page: int = 1) -> list[dict]:
        """
        Search for jobs on Naukri and return a list of job listings.
        Each listing has: title, company, url, description_snippet, experience_range.
        page > 1 opens that results page (Naukri adds "-2", "-3"... to the path).
        """
        jobs = []
        try:
            search_url = naukri_search_url(keywords, location, experience, page)
            logger.info("Searching Naukri: %s", search_url)
            self.page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
            _human_delay(LONG_DELAY)

            # Scroll down to load more jobs
            for _ in range(3):
                self.page.mouse.wheel(0, random.randint(500, 1000))
                _human_delay(SHORT_DELAY)

            # Extract job cards with more specific selectors
            job_card_selectors = [
                "article.jobTuple",
                "div.srp-jobtuple-wrapper",
                "div.cust-job-tuple",
                "div[data-job-id]",
                "div[class*='jobTuple']",
            ]

            job_cards = None
            count = 0
            for selector in job_card_selectors:
                try:
                    job_cards = self.page.locator(selector)
                    count = job_cards.count()
                    if count > 0:
                        logger.info("Found %d job cards using selector: %s", count, selector)
                        break
                except Exception:
                    continue

            if count == 0:
                # New layouts: fall back to the job links themselves.
                jobs = self._job_links_on_page()
                logger.warning("No job cards found with any selector; %d job link(s) on the page", len(jobs))
                return jobs

            max_cards = min(count, 30)  # Cap at 30 jobs per page
            for i in range(max_cards):
                try:
                    card = job_cards.nth(i)

                    # Extract title with multiple fallback selectors
                    title = ""
                    title_selectors = [
                        "a.title",
                        "a[class*='title']",
                        "h2 a",
                        "a[class*='jobTitle']",
                        "a[data-ga-track*='jobTitle']",
                    ]
                    for sel in title_selectors:
                        try:
                            title_el = card.locator(sel).first
                            if title_el.count() > 0:
                                title = title_el.inner_text().strip()
                                if title:
                                    break
                        except Exception:
                            continue

                    # Extract URL
                    url = ""
                    for sel in title_selectors:
                        try:
                            title_el = card.locator(sel).first
                            if title_el.count() > 0:
                                url = title_el.get_attribute("href") or ""
                                if url:
                                    break
                        except Exception:
                            continue
                    if url and not url.startswith("http"):
                        url = f"{NAUKRI_BASE_URL}{url}"

                    # Extract company
                    company = "Unknown"
                    company_selectors = [
                        "a.subTitle",
                        "a[class*='comp']",
                        "span[class*='comp']",
                        "a[class*='company']",
                        "a[data-ga-track*='companyName']",
                    ]
                    for sel in company_selectors:
                        try:
                            company_el = card.locator(sel).first
                            if company_el.count() > 0:
                                company = company_el.inner_text().strip()
                                if company:
                                    break
                        except Exception:
                            continue

                    # Extract experience
                    exp_range = ""
                    exp_selectors = [
                        "span.expwdth",
                        "span[class*='exp']",
                        "li[class*='exp']",
                        "span[class*='expwdth']",
                    ]
                    for sel in exp_selectors:
                        try:
                            exp_el = card.locator(sel).first
                            if exp_el.count() > 0:
                                exp_range = exp_el.inner_text().strip()
                                if exp_range:
                                    break
                        except Exception:
                            continue

                    # Extract snippet
                    snippet = ""
                    snippet_selectors = [
                        "span.job-desc",
                        "div[class*='desc']",
                        "span[class*='desc']",
                    ]
                    for sel in snippet_selectors:
                        try:
                            snippet_el = card.locator(sel).first
                            if snippet_el.count() > 0:
                                snippet = snippet_el.inner_text().strip()
                                if snippet:
                                    break
                        except Exception:
                            continue

                    if title and url:
                        jobs.append({
                            "title": title,
                            "company": company,
                            "url": url,
                            "experience_range": exp_range,
                            "description_snippet": snippet[:500],
                        })
                except Exception as card_exc:
                    logger.debug("Error extracting job card %d: %s", i, card_exc)
                    continue

        except Exception as exc:
            logger.error("Job search failed: %s", exc)

        logger.info("Extracted %d jobs from search results", len(jobs))
        return jobs

    def get_job_details(self, job_url: str) -> dict:
        """Navigate to a job page and extract full details."""
        try:
            self.page.goto(job_url, wait_until="domcontentloaded", timeout=20000)
            _human_delay(MEDIUM_DELAY)

            details = {"url": job_url}

            # Title
            title_el = self.page.locator("h1, h1[class*='title'], .jd-header-title").first
            details["title"] = title_el.inner_text().strip() if title_el.count() > 0 else ""

            # Company
            comp_el = self.page.locator("a[class*='comp'], div[class*='comp'] a, .jd-header-comp-name a").first
            details["company"] = comp_el.inner_text().strip() if comp_el.count() > 0 else ""

            # Full description
            desc_el = self.page.locator(
                "div.job-desc, div[class*='description'], "
                "section[class*='desc'], div.dang-inner-html, "
                "div[class*='jobDesc'] div[class*='dang']"
            ).first
            details["description"] = desc_el.inner_text().strip()[:3000] if desc_el.count() > 0 else ""

            # Experience & Salary
            info_els = self.page.locator("div.salary, span[class*='exp'], div[class*='details'] span, div.loc span").all()
            details["meta_info"] = " | ".join(
                el.inner_text().strip() for el in info_els[:5] if el.inner_text().strip()
            )

            return details
        except Exception as exc:
            logger.error("Failed to get job details for %s: %s", job_url, exc)
            return {"url": job_url, "title": "", "company": "", "description": "", "error": str(exc)}

    def _wait_for_portal_url(self, page: Page, timeout_s: float = 25) -> str:
        """Wait until `page` has left about:blank / Naukri's redirect pages and return the portal URL."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            url = page.url or ""
            if url.startswith("http") and "naukri.com" not in url:
                try:
                    page.wait_for_load_state("load", timeout=10000)
                except Exception:
                    pass
                return page.url
            try:
                page.wait_for_timeout(500)
            except Exception:
                return ""
        return ""

    def _capture_portal_url(self, apply_btn) -> str:
        """
        Learn the company-portal URL behind "Apply on company site".

        The application itself happens later in job_apply_service (in Veronica's
        own browser, with the user's confirmation before any account or submit),
        so the tab opened here is closed once its real URL is known. Reading
        the URL at domcontentloaded used to return about:blank or Naukri's
        redirect page, so the portal step later opened nothing useful.
        """
        try:
            href = apply_btn.get_attribute("href") or ""
            if href.startswith("http") and "naukri.com" not in href:
                return href
        except Exception:
            pass

        try:
            with self.context.expect_page(timeout=15000) as new_page_info:
                apply_btn.click()
            portal = new_page_info.value
            url = self._wait_for_portal_url(portal)
            if not url:
                logger.warning("Company site tab never left %s", portal.url)
            try:
                portal.close()
            except Exception:
                pass
            return url
        except PlaywrightTimeout:
            # Some jobs navigate in the same tab instead of opening a new one.
            url = self._wait_for_portal_url(self.page, timeout_s=10)
            if url:
                try:
                    self.page.go_back(timeout=15000)
                except Exception:
                    pass
            return url
        except Exception as exc:
            logger.warning("Could not open the company site: %s", exc)
            return ""

    def _portal_opened_by_click(self, pages_before: list) -> Optional[str]:
        """
        After an Apply click: if it opened the company site (new tab or this
        tab), return its URL ("" if it could not be read), else None.
        """
        new_pages = []
        for _ in range(6):
            new_pages = [p for p in self.context.pages if not p.is_closed() and p not in pages_before]
            if new_pages:
                break
            self.page.wait_for_timeout(250)
        if new_pages:
            portal = new_pages[-1]
            url = self._wait_for_portal_url(portal)
            try:
                portal.close()
            except Exception:
                pass
            return url
        url = self.page.url or ""
        if url.startswith("http") and "naukri.com" not in url:
            try:
                self.page.go_back(timeout=15000)
            except Exception:
                pass
            return url
        return None

    def apply_to_job(self, form_filler: ScreeningFormFiller) -> dict:
        """
        Attempt to apply to the current job page.
        Returns {"status": "applied"|"skipped"|"external"|"error", "message": "..."}
        """
        try:
            # Check if already applied
            applied_selectors = [
                "button:has-text('Already Applied')",
                "span:has-text('Already Applied')",
                "div[class*='applied']",
                "div:has-text('Already Applied')",
            ]
            for selector in applied_selectors:
                try:
                    if self.page.locator(selector).first.is_visible(timeout=2000):
                        return {"status": "skipped", "message": "Already applied to this job"}
                except Exception:
                    continue

            # Check for CAPTCHA
            captcha_selectors = [
                "iframe[src*='recaptcha']",
                "div[id*='captcha']",
                "div[class*='captcha']",
                "img[src*='captcha']",
            ]
            def _captcha_shown() -> bool:
                for selector in captcha_selectors:
                    try:
                        if self.page.locator(selector).first.is_visible(timeout=1000):
                            return True
                    except Exception:
                        continue
                return False

            if _captcha_shown():
                # Leave it on screen; Boss completes it in the Edge window and this job carries on.
                from autojob.jobs.easy_apply import wait_for_human_check

                if not wait_for_human_check(_captcha_shown, lambda: self._should_stop, "Naukri",
                                            bring_front=self.page.bring_to_front):
                    return {"status": "error", "message": "CAPTCHA detected - manual intervention required"}

            # Find the Apply button with more specific selectors
            apply_selectors = [
                "button#company-site-button",
                "button#apply-button",
                "button[data-ga-track*='apply']",
                "button[id*='applyBtn']",
                "button:has-text('Apply')",
                "a:has-text('Apply on Naukri')",
                "button:has-text('Apply on company site')",
                "div[class*='apply-button'] button",
                "div[class*='apply'] button",
            ]

            apply_btn = None
            is_external = False

            for selector in apply_selectors:
                try:
                    btn = self.page.locator(selector).first
                    if btn.count() > 0 and btn.is_visible(timeout=2000):
                        btn_text = btn.inner_text().strip().lower()
                        btn_id = (btn.get_attribute("id") or "").lower()
                        if ("company-site" in btn_id
                                or re.search(r"company|website|career|external|redirect", btn_text)):
                            is_external = True
                        apply_btn = btn
                        break
                except Exception:
                    continue

            if apply_btn is None:
                return {"status": "skipped", "message": "No Apply button found on page"}

            if is_external:
                # Follow the button only to learn the company portal URL; the
                # application itself is done later by job_apply_service with
                # the user's confirmation at each account/submit step.
                external_url = self._capture_portal_url(apply_btn)
                return {
                    "status": "external",
                    "external_url": external_url,
                    "message": "Applies on the company site" + ("" if external_url else " (portal URL not captured)"),
                }

            # Scroll to the button and click with human-like behavior
            pages_before = [p for p in self.context.pages if not p.is_closed()]
            try:
                apply_btn.scroll_into_view_if_needed()
                _human_delay(SHORT_DELAY)
                apply_btn.click()
                _human_delay(MEDIUM_DELAY)
            except Exception as exc:
                logger.error("Failed to click apply button: %s", exc)
                return {"status": "error", "message": f"Failed to click apply button: {exc}"}

            # A plain-looking "Apply" can still lead to the company's site, in a
            # new tab or this one. That used to be reported as "applied" while
            # nothing was submitted; hand it to the company-site agent instead.
            portal_url = self._portal_opened_by_click(pages_before)
            if portal_url is not None:
                return {
                    "status": "external",
                    "external_url": portal_url,
                    "message": "Applies on the company site" + ("" if portal_url else " (portal URL not captured)"),
                }

            # Answer the screening chatbot / questionnaire if one opened
            form_state = self._handle_screening_form(form_filler)
            if form_state == "needs_user":
                return {"status": "needs_user",
                        "question": self._unanswered_question,
                        "message": (f"Not applied: the form asks \"{self._unanswered_question[:150]}\", which your CV "
                                    "and profile do not answer, so nothing was guessed. Tell Veronica the answer "
                                    "and it will be used next time.")}
            if form_state == "stuck":
                return {"status": "error",
                        "message": "Screening questions could not be completed — please finish this one manually"}

            # Only Naukri's own confirmation counts. A questionnaire that closed
            # half-way (or a drawer reference that went stale) used to be
            # reported as "applied" while the application was never sent.
            if self._wait_for_confirmation(8):
                return {"status": "applied", "message": "Applied - Naukri confirmed it"}
            if form_state == "closed":
                return {"status": "unconfirmed",
                        "message": ("The screening questions closed before Naukri confirmed the application - "
                                    "NOT counted as applied. Check this job on Naukri.")}
            return {"status": "unconfirmed",
                    "message": "Clicked Apply but Naukri showed no confirmation - NOT counted as applied, check it"}

        except PlaywrightTimeout:
            return {"status": "error", "message": "Timeout during application"}
        except Exception as exc:
            logger.error("Apply failed: %s", exc)
            return {"status": "error", "message": str(exc)}

    # Naukri's screening questions come in a chat drawer ("chatbot"): a bot
    # bubble asks one question at a time and the answer goes into a
    # contenteditable div ("Type message here..."), radio/checkbox chips, or
    # suggestion chips, followed by a Save/Submit control that is often a
    # <div>, not a <button>. Everything below is scoped to that drawer so we
    # never type into the page header's search box.
    _DRAWER_SELECTORS = [
        "div[class*='chatbot_Drawer']",
        "div[class*='chatbot_drawer']",
        "div[class*='chatbot']",
        "div[id*='chatbot']",
        "div[class*='screening']",
        "div[class*='questionnaire']",
        "div[class*='apply-modal']",
        "div[role='dialog']",
        "div[class*='drawer']",
        "div[class*='modal']",
    ]
    _QUESTION_SELECTORS = [
        "li[class*='botItem'] div[class*='botMsg']",
        "div[class*='botMsg']",
        "li[class*='botItem']",
        "[class*='bot'][class*='msg']",
        "[class*='question']",
    ]
    _TEXT_INPUT_SELECTORS = (
        "[contenteditable='true'], textarea, input[type='text'], "
        "input[type='number'], input[type='tel'], input:not([type])"
    )
    _SEND_SELECTORS = [
        "[class*='sendMsg']",
        "button:has-text('Save')",
        "button:has-text('Submit')",
        "button:has-text('Next')",
        "button:has-text('Continue')",
        "div:text-is('Save')",
        "div:text-is('Submit')",
        "span:text-is('Save')",
        "button[type='submit']",
    ]

    def _find_drawer(self, timeout: int = 4000):
        deadline = time.time() + timeout / 1000
        while time.time() < deadline:
            for selector in self._DRAWER_SELECTORS:
                try:
                    loc = self.page.locator(selector)
                    for i in range(min(loc.count(), 5)):
                        el = loc.nth(i)
                        # A real questionnaire has somewhere to answer or a send control.
                        if el.is_visible() and (
                            el.locator(self._TEXT_INPUT_SELECTORS).count() > 0
                            or el.locator("input[type='radio'], input[type='checkbox']").count() > 0
                            or el.locator("[class*='sendMsg'], [class*='chip'], [class*='Chip']").count() > 0
                        ):
                            return el
                except Exception:
                    continue
            time.sleep(0.4)
        return None

    def _current_question(self, drawer) -> str:
        for selector in self._QUESTION_SELECTORS:
            try:
                loc = drawer.locator(selector)
                if loc.count() > 0:
                    text = loc.last.inner_text().strip()
                    if text:
                        return text
            except Exception:
                continue
        try:
            lines = [l.strip() for l in drawer.inner_text().splitlines() if l.strip()]
            lines = [l for l in lines if l.lower() not in ("save", "submit", "skip", "type message here...")]
            return lines[-1] if lines else ""
        except Exception:
            return ""

    @staticmethod
    def _pick(answer: str, labels: list[str]) -> int:
        a = answer.strip().lower()
        for i, l in enumerate(labels):
            if l.lower() == a:
                return i
        for i, l in enumerate(labels):
            if l and (a in l.lower() or l.lower() in a):
                return i
        return 0

    def _choice_options(self, drawer, input_type: str) -> list[tuple[str, Any]]:
        """Return (label text, clickable element) for visible radio/checkbox options."""
        out = []
        inputs = drawer.locator(f"input[type='{input_type}']")
        for i in range(inputs.count()):
            inp = inputs.nth(i)
            label = None
            inp_id = inp.get_attribute("id")
            if inp_id:
                cand = drawer.locator(f"label[for='{inp_id}']")
                if cand.count() > 0:
                    label = cand.first
            if label is None:
                cand = inp.locator("xpath=ancestor::label[1] | following-sibling::label[1] | ..")
                label = cand.first if cand.count() > 0 else None
            text = (label.inner_text().strip() if label is not None else "") or (inp.get_attribute("value") or "")
            target = label if label is not None and label.is_visible() else inp
            if text:
                out.append((text, target))
        return out

    _SKIP_RE = re.compile(r"^\s*(skip|skip this( question)?|i'?ll skip)\s*$", re.I)

    def _click_skip(self, drawer) -> bool:
        """Use Naukri's own 'Skip' choice when a question offers one."""
        cands = drawer.locator("label, button, span, div[class*='chip' i], div[class*='Chip']")
        for i in range(min(cands.count(), 40)):
            el = cands.nth(i)
            try:
                if el.is_visible() and self._SKIP_RE.match(el.inner_text() or ""):
                    el.click()
                    return True
            except Exception:
                continue
        return False

    def _answer_current(self, drawer, question: str, form_filler: ScreeningFormFiller) -> str:
        """
        Answer the question on screen from the candidate's data.
        Returns "answered", "skipped" (used the form's Skip), "unanswerable"
        (no verified answer, nothing typed) or "none" (no answer field found).
        """
        def unanswerable() -> str:
            if self._click_skip(drawer):
                logger.info("Chatbot: skipped '%s' (not in CV/profile)", question[:80])
                return "skipped"
            self._unanswered_question = question
            return "unanswerable"

        for kind in ("radio", "checkbox"):
            options = [(t, el) for t, el in self._choice_options(drawer, kind) if not self._SKIP_RE.match(t)]
            if options:
                labels = [t for t, _ in options]
                best = form_filler.answer_question(question, labels)
                if best is None:
                    return unanswerable()
                text, target = options[self._pick(best, labels)]
                target.click(force=True)
                logger.info("Chatbot %s: '%s' -> '%s'", kind, question[:60], text[:60])
                return "answered"

        chips = drawer.locator("[class*='chatbot_Chip'], [class*='chip']:visible, [class*='Chip']:visible")
        chip_list = [chips.nth(i) for i in range(min(chips.count(), 15))]
        chip_list = [c for c in chip_list if c.is_visible() and c.inner_text().strip()
                     and not self._SKIP_RE.match(c.inner_text())]
        if chip_list:
            labels = [c.inner_text().strip() for c in chip_list]
            best = form_filler.answer_question(question, labels)
            if best is None:
                return unanswerable()
            idx = self._pick(best, labels)
            chip_list[idx].click()
            logger.info("Chatbot chip: '%s' -> '%s'", question[:60], labels[idx][:60])
            return "answered"

        fields = drawer.locator(self._TEXT_INPUT_SELECTORS)
        for i in range(fields.count()):
            field = fields.nth(i)
            try:
                if not field.is_visible() or field.is_disabled():
                    continue
            except Exception:
                continue
            answer = form_filler.answer_question(question)
            if answer is None:
                return unanswerable()
            field.click()
            # contenteditable divs have no value, and React only notices real
            # key events, so type instead of fill().
            self.page.keyboard.press("Control+A")
            self.page.keyboard.press("Backspace")
            field.press_sequentially(answer, delay=random.randint(40, 90))
            logger.info("Chatbot text: '%s' -> '%s'", question[:60], answer[:60])
            return "answered"

        selects = drawer.locator("select")
        for i in range(selects.count()):
            sel = selects.nth(i)
            if not sel.is_visible():
                continue
            opts = [o.strip() for o in sel.locator("option").all_inner_texts() if o.strip()]
            if opts:
                best = form_filler.answer_question(question, opts)
                if best is None:
                    return unanswerable()
                best = opts[self._pick(best, opts)]
                sel.select_option(label=best)
                logger.info("Chatbot select: '%s' -> '%s'", question[:60], best[:60])
                return "answered"
        return "none"

    def _click_send(self, drawer) -> bool:
        # Naukri's chatbot (seen live): <div class="send disabled"><div class="sendMsg">Save</div></div>.
        # "disabled" sits on the OUTER div.send, so waiting on .sendMsg never saw it and
        # Save was clicked before the typed answer registered (the answer was lost).
        send = drawer.locator("div.send")
        if send.count() > 0:
            try:
                for _ in range(20):
                    if "disabled" not in (send.last.get_attribute("class") or "").lower():
                        break
                    time.sleep(0.3)
                else:
                    logger.warning("Chatbot Save stayed disabled - the answer did not register")
                    return False
                inner = send.last.locator(".sendMsg")
                (inner.first if inner.count() > 0 else send.last).click()
                return True
            except Exception as exc:
                logger.debug("div.send click failed: %s", exc)
        for selector in self._SEND_SELECTORS:
            try:
                btn = drawer.locator(selector).last
                if btn.count() == 0 or not btn.is_visible():
                    continue
                # Save stays disabled until the answer registers; give it a moment.
                for _ in range(10):
                    cls = (btn.get_attribute("class") or "").lower()
                    if "disabled" not in cls and btn.is_enabled():
                        break
                    time.sleep(0.3)
                btn.click()
                return True
            except Exception:
                continue
        try:
            self.page.keyboard.press("Enter")
            return True
        except Exception:
            return False

    def _handle_screening_form(self, form_filler: ScreeningFormFiller) -> str:
        """
        Answer Naukri's screening chatbot / questionnaire, one question at a time.
        Returns "none" (no form appeared), "done" (Naukri confirmed), "closed"
        (the drawer went away without a confirmation), "stuck", or "needs_user"
        (a question the CV/profile cannot answer; nothing was guessed).
        Questions differ per company, so each one is read fresh from the drawer.
        """
        self._unanswered_question = ""
        try:
            drawer = self._find_drawer()
            if drawer is None:
                return "none"
            logger.info("Screening questionnaire detected")

            last_question, repeats = "", 0
            for _ in range(25):  # safety cap on number of questions
                if self.should_stop:
                    return "stuck"
                # Re-find the drawer: Naukri re-renders it between questions and
                # an old locator can point at a hidden copy.
                drawer = self._find_drawer(timeout=3000)
                if drawer is None:
                    return "done" if self._confirmed() else "closed"
                question = self._current_question(drawer)
                if question == last_question:
                    repeats += 1
                    if repeats >= 2:
                        logger.warning("Screening form stuck on: %s", question[:120])
                        return "stuck"
                else:
                    repeats = 0
                last_question = question

                _human_delay(SHORT_DELAY)
                outcome = self._answer_current(drawer, question, form_filler)
                if outcome == "unanswerable":
                    logger.warning("Handing job to the user; not in CV/profile: %s", question[:120])
                    return "needs_user"
                _human_delay(SHORT_DELAY)
                if outcome != "skipped" and not self._click_send(drawer) and outcome == "none":
                    return "stuck"

                # Wait for the next question, a success message, or the drawer to close.
                deadline = time.time() + 12
                while time.time() < deadline:
                    time.sleep(0.5)
                    if self._confirmed():
                        return "done"
                    fresh = self._find_drawer(timeout=500)
                    if fresh is None:
                        break  # gone: the top of the loop decides done / closed
                    if self._current_question(fresh) != question:
                        break
            return "stuck"
        except Exception as exc:
            logger.error("Screening form handling failed: %s", exc)
            return "stuck"

    # Seen live (Nov 2026): a job with no questions jumps to
    # /myapply/saveApply?...multiApplyResp={"<id>":200} titled "Apply Confirmation"
    # with the text: Applied to "<job title>".
    _CONFIRM_JS = r"""
    () => {
      if (/\/myapply\/saveApply/i.test(location.pathname)
          && (/%22%3A200|":200/.test(location.search) || /apply confirmation/i.test(document.title))) return true;
      if (/^\s*applied to\s+["“]/im.test((document.body && document.body.innerText || '').slice(0, 3000))) return true;
      const vis = el => { const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
      for (const el of document.querySelectorAll('#apply-button, button[id*=apply], button[class*=apply]'))
        if (vis(el) && /^\s*(already\s+)?applied\s*$/i.test(el.innerText || '')) return true;
      const rx = /applied successfully|successfully applied|application (was )?submitted|you have (successfully )?applied/i;
      for (const el of document.querySelectorAll('div, span, p'))
        if (el.children.length < 4 && vis(el) && rx.test(el.innerText || '')) return true;
      return false;
    }
    """

    def _confirmed(self) -> bool:
        """Naukri says it went through: a success message, or the job's Apply button now reads Applied."""
        try:
            return bool(self.page.evaluate(self._CONFIRM_JS))
        except Exception:
            return False

    def _wait_for_confirmation(self, seconds: float) -> bool:
        deadline = time.time() + seconds
        while True:
            if self._confirmed():
                return True
            if time.time() >= deadline:
                return False
            time.sleep(0.5)

    def _visible(self, selector: str) -> bool:
        try:
            return self.page.locator(selector).first.is_visible()
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Session Results Storage
# ---------------------------------------------------------------------------

def _save_results(results: list[dict]):
    """Save application results to JSON file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "last_session": datetime.now().isoformat(),
        "total_applied": sum(1 for r in results if r.get("status") == "applied"),
        "total_skipped": sum(1 for r in results if r.get("status") in ("skipped", "external")),
        "total_errors": sum(1 for r in results if r.get("status") == "error"),
        "applications": results,
    }
    try:
        RESULTS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Results saved to %s", RESULTS_FILE)
    except Exception as exc:
        logger.error("Failed to save results: %s", exc)


def _load_previous_results() -> list[dict]:
    """Load previously applied job URLs to avoid re-applying."""
    if RESULTS_FILE.exists():
        try:
            data = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
            return data.get("applications", [])
        except Exception:
            pass
    return []


# ---------------------------------------------------------------------------
# Main Orchestration Functions (exposed as tools)
# ---------------------------------------------------------------------------

# Global bot instance for session management
_active_bot: Optional[NaukriJobBot] = None
_active_lock = threading.Lock()

@tool
def apply_jobs_naukri(
    keywords: str = "",
    location: str = "",
    max_applications: int = DEFAULT_MAX_APPLICATIONS,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Main function: Parse CV → Search Jobs → Match & Apply continuously. Jobs that apply on a company career site are then applied to there as well, pausing with 'needs_confirmation' before any account creation or final submit (answer with confirm_job_step).

    Args:
        keywords: Job search keywords (e.g., "Python Developer"). Auto-derived from CV if empty.
        location: Preferred job location (e.g., "Bangalore"). Optional.
        max_applications: Maximum number of applications per session (default 25).
        dry_run: If True, don't actually click Apply (for testing).

    Returns:
        Summary dict with applied/skipped/error counts and details.
    """
    if dry_run:
        return run_naukri_apply(keywords, location, max_applications, dry_run=True)
    # Same pipeline as auto_apply_jobs: previously this tool only collected
    # "Apply on company site" jobs and nothing ever applied to them.
    from autojob.jobs.job_apply_service import auto_apply_jobs
    return auto_apply_jobs.invoke({
        "keywords": keywords, "location": location, "max_applications": max_applications,
    })


def run_naukri_apply(
    keywords: str = "",
    location: str = "",
    max_applications: int = DEFAULT_MAX_APPLICATIONS,
    dry_run: bool = False,
    on_external=None,
) -> dict[str, Any]:
    """
    Naukri-only part of the pipeline: search, match and apply on Naukri itself.
    on_external(job) is called as soon as a company-site job is found, so it is
    saved even if this run later fails or is stopped.
    """
    global _active_bot

    # Claim the single session slot atomically. Checking only for status
    # "running" let a second call through while the first was "searching" or
    # "applying"; it then killed the first run's browser and relaunched.
    bot = NaukriJobBot()
    with _active_lock:
        if _active_bot is not None:
            return {
                "error": f"A Naukri apply session is already in progress ({_active_bot.status}). "
                         "Use check_naukri_status to follow it or stop_naukri_apply to stop it."
            }
        _active_bot = bot

    try:
        return _run_naukri_session(bot, keywords, location, max_applications, dry_run,
                                   on_external)
    finally:
        with _active_lock:
            _active_bot = None


def _run_naukri_session(
    bot: "NaukriJobBot", keywords: str, location: str, max_applications: int, dry_run: bool,
    on_external=None,
) -> dict[str, Any]:

    logger.info("=== Starting Naukri Auto-Apply Session ===")
    logger.info("Keywords: %s | Location: %s | Max: %d | Dry run: %s",
                keywords, location, max_applications, dry_run)

    # Step 1: Parse CV (and bring the saved profile up to date if the CV changed)
    from autojob.core.cv_sync import sync_if_cv_changed

    sync_if_cv_changed(wait=True)
    parser = CVParser()
    profile = parser.parse()
    if "error" in profile:
        return {"error": f"CV parsing failed: {profile['error']}"}

    # The user's edited profile (CTC, notice period, ...) wins over CV parsing.
    profile_file = DATA_DIR / "job_applications" / "candidate_profile.json"
    if profile_file.exists():
        try:
            overrides = json.loads(profile_file.read_text(encoding="utf-8"))
            profile.update({k: v for k, v in overrides.items()
                            if not k.startswith("_") and v not in (None, "", [])})
        except Exception as exc:
            logger.warning("Could not read %s: %s", profile_file, exc)

    logger.info("CV parsed. Name: %s, Skills: %s",
                profile.get("name"), profile.get("skills", [])[:5])

    # Auto-derive keywords from CV if not provided
    if not keywords:
        search_kw = profile.get("search_keywords", [])
        job_titles = profile.get("job_titles", [])
        keywords = search_kw[0] if search_kw else (job_titles[0] if job_titles else "developer")
        logger.info("Auto-derived search keywords from CV: %s", keywords)

    # Pan-India: the main cities + all-India unless Boss named a place.
    locations = search_locations(location)
    location = ", ".join(loc or "All India" for loc in locations)

    # Step 2: Initialize browser
    try:
        bot.start()

        # Step 3: Verify login
        if not bot.verify_login():
            bot.stop()
            return {
                "error": "Not logged into Naukri.com. Please open Edge, go to naukri.com, "
                         "log in, then try again. The bot will use your saved session."
            }

        logger.info("✅ Naukri login verified!")

        # Step 4: Initialize matchers
        matcher = JobMatcher(cv_profile=profile, cv_raw_text=parser.raw_text)
        # Screening answers may also come from ABOUT_ME.md (e.g. years per skill),
        # not just the CV - each company asks different questions.
        from autojob.core.user_profile import about_me_text
        form_filler = ScreeningFormFiller(cv_profile=profile,
                                          cv_raw_text=about_me_text() + "\n\n" + parser.raw_text)

        # Load previously applied URLs
        prev_results = _load_previous_results()
        applied_urls = set(r.get("url", "") for r in prev_results if r.get("status") == "applied")

        # Step 5: Search for jobs
        bot.status = "searching"
        experience = profile.get("experience_years")
        per_city = [bot.search_jobs(keywords=keywords, location=loc, experience=experience)
                    for loc in locations if not bot.should_stop]
        jobs = interleave_jobs(per_city)

        if not jobs:
            bot.stop()
            return {
                "summary": "No jobs found matching your search criteria.",
                "keywords": keywords,
                "location": location,
                "total_applied": 0,
            }

        # Step 6: Apply to jobs one by one
        bot.status = "applying"
        results = []
        external_jobs: list[dict] = []
        applied_count = 0
        skipped_count = 0
        error_count = 0

        for idx, job in enumerate(jobs):
            if bot.should_stop:
                logger.info("Stop requested by user. Ending session.")
                break

            if applied_count >= max_applications:
                logger.info("Reached max applications limit (%d). Stopping.", max_applications)
                break

            job_url = job.get("url", "")

            # Skip already applied
            if job_url in applied_urls:
                logger.info("Skipping already-applied: %s", job.get("title", ""))
                skipped_count += 1
                continue

            logger.info("[%d/%d] Processing: %s at %s",
                        idx + 1, len(jobs), job.get("title", ""), job.get("company", ""))

            # Get full job details
            details = bot.get_job_details(job_url)
            if details.get("error"):
                results.append({
                    "job_title": job.get("title", ""),
                    "company": job.get("company", ""),
                    "url": job_url,
                    "match_score": 0,
                    "status": "error",
                    "message": details["error"],
                    "timestamp": datetime.now().isoformat(),
                })
                error_count += 1
                continue

            # Score the job match
            score_result = matcher.score_job(
                job_title=details.get("title", job.get("title", "")),
                company=details.get("company", job.get("company", "")),
                job_description=details.get("description", job.get("description_snippet", "")),
            )

            match_score = score_result.get("score", 0)
            logger.info("Match score: %d/100 — %s", match_score, score_result.get("reasoning", ""))

            if not score_result.get("should_apply", False):
                results.append({
                    "job_title": details.get("title", ""),
                    "company": details.get("company", ""),
                    "url": job_url,
                    "match_score": match_score,
                    "status": "skipped",
                    "message": f"Low match ({match_score}): {score_result.get('reasoning', '')}",
                    "timestamp": datetime.now().isoformat(),
                })
                skipped_count += 1
                continue

            # Apply!
            if dry_run:
                apply_result = {"status": "dry_run", "message": "Dry run — would have applied"}
            else:
                apply_result = bot.apply_to_job(form_filler)

            result_entry = {
                "job_title": details.get("title", ""),
                "company": details.get("company", ""),
                "url": job_url,
                "match_score": match_score,
                "status": apply_result["status"],
                "message": apply_result["message"],
                "timestamp": datetime.now().isoformat(),
            }
            results.append(result_entry)

            if apply_result["status"] == "applied":
                applied_count += 1
                applied_urls.add(job_url)
                logger.info("✅ Applied to %s at %s (score: %d)",
                            details.get("title", ""), details.get("company", ""), match_score)
            elif apply_result["status"] == "external":
                # Skipped for now so the Naukri run keeps going; saved for the
                # company-site agent. Without a captured portal URL the agent
                # starts from the Naukri page and clicks through itself.
                skipped_count += 1
                portal_url = apply_result.get("external_url") or ""
                if portal_url:
                    result_entry["external_url"] = portal_url
                external = {
                    "url": portal_url or job_url,
                    "via_naukri": not portal_url,
                    "company": details.get("company", job.get("company", "")),
                    "title": details.get("title", job.get("title", "")),
                    "source": "naukri",
                    "naukri_url": job_url,
                    "match_score": match_score,
                }
                external_jobs.append(external)
                if on_external:
                    try:
                        on_external(external)
                    except Exception as exc:
                        logger.warning("Could not save company-site job: %s", exc)
                logger.info("Saved company-site job for the portal agent: %s at %s -> %s",
                            external["title"], external["company"], external["url"])
            elif apply_result["status"] in ("error", "unconfirmed"):
                error_count += 1
            elif apply_result["status"] == "needs_user":
                skipped_count += 1

            # Human-like delay between applications
            _human_delay(LONG_DELAY)

        # Save results
        _save_results(results)

        summary = {
            "summary": (f"Naukri pass finished: applied on Naukri to {applied_count} job(s), skipped "
                        f"{skipped_count - len(external_jobs)}, errors {error_count}. "
                        f"{len(external_jobs)} job(s) apply on the company's own site and are NOT applied yet - "
                        "they are handed to the company-site agent next."),
            "keywords": keywords,
            "location": location or "Any",
            "total_jobs_found": len(jobs),
            "total_applied": applied_count,
            "total_skipped": skipped_count,
            "total_errors": error_count,
            "applications": results[:10],  # First 10 for brevity
            # Jobs that apply on a company career portal; job_apply_service
            # works through these after the Naukri browser has closed.
            "external_jobs": external_jobs,
            "all_results": results,
        }

        logger.info("=== Session Complete: Applied=%d, Skipped=%d, Errors=%d ===",
                     applied_count, skipped_count, error_count)
        return summary

    except Exception as exc:
        logger.exception("Naukri auto-apply session failed")
        return {"error": f"Session failed: {exc}"}
    finally:
        try:
            bot.stop()
        except Exception:
            pass


@tool
def check_naukri_status() -> dict[str, Any]:
    """Check the status of the current Naukri auto-apply session."""
    with _active_lock:
        if _active_bot is None:
            # Check for last session results
            prev = _load_previous_results()
            if prev:
                applied = sum(1 for r in prev if r.get("status") == "applied")
                return {
                    "status": "idle",
                    "message": f"No active session. Last session: applied to {applied} jobs.",
                    "last_results": prev[:5],
                }
            return {"status": "idle", "message": "No active session and no previous results."}

        return {
            "status": _active_bot.status,
            "message": f"Session is {_active_bot.status}.",
            "results_so_far": _active_bot._session_results[:5] if _active_bot._session_results else [],
        }


@tool
def stop_naukri_apply() -> dict[str, Any]:
    """Stop the currently running Naukri auto-apply session."""
    with _active_lock:
        if _active_bot is None:
            return {"status": "idle", "message": "No active session to stop."}
        _active_bot.request_stop()
        return {"status": "stopping", "message": "Stop signal sent. Session will end after current job."}


NAUKRI_LANGCHAIN_TOOLS = [
    apply_jobs_naukri,
    check_naukri_status,
    stop_naukri_apply
]

def get_naukri_tools() -> list[dict]:
    """Return the tools list ready to merge into LiveConnectConfig."""
    from autojob.core.langchain_bridge import get_gemini_tools_from_langchain
    return get_gemini_tools_from_langchain(NAUKRI_LANGCHAIN_TOOLS)

def execute_naukri_function(name: str, args: dict) -> dict[str, Any]:
    """Dispatch a function call to the right Naukri tool handler."""
    for t in NAUKRI_LANGCHAIN_TOOLS:
        if t.name == name:
            try:
                return t.invoke(args)
            except Exception as exc:
                return {"error": f"Invalid arguments or execution failed for {name}: {exc}"}
    return {"error": f"Unknown naukri function: {name}"}
