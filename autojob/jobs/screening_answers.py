"""
Answers for the questions job boards ask while applying (LinkedIn Easy Apply,
Indeed Apply, ...), one whole form step at a time.

    1. rules   contact details, total experience, notice period, CTC, consent
               boxes and the like come straight from candidate_profile.json,
               without a model call;
    2. cache   a question already answered in this run gets the same answer;
    3. model   ONE call answers the rest, each answer with quotes from the CV /
               profile / answers Boss gave before, checked against the real
               text exactly like the Naukri chatbot answers (ScreeningFormFiller).

Nothing unsupported is ever typed. Such a question comes back as None, is
saved to questions_waiting.md, and the caller skips the job, so Boss answers it
once (teach_job_agent) and every later run fills it in.

Field dicts come from easy_apply.collect_fields:
    {"id": "3", "kind": "text|number|textarea|select|radio|checkbox|checkboxes|combobox|date|email|tel|file",
     "question": "...", "options": [...], "required": bool, "value": "...", "error": "..."}
Answers: str for text-like fields, select, radio and combobox (an exact option
for choices), bool for a single checkbox, list[str] for a checkbox group, and
UPLOAD_CV for a resume file input.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from autojob.jobs.naukri_service import ScreeningFormFiller

logger = logging.getLogger(__name__)

UPLOAD_CV = "__upload_cv__"

_TEXT_KINDS = ("text", "number", "textarea", "email", "tel", "combobox", "date")
_CHOICE_KINDS = ("select", "radio")

_EMAIL_Q = re.compile(r"\be-?mail( address| id)?\b", re.I)
_COUNTRY_CODE_Q = re.compile(r"country( calling)? code|phone country|\bdial(ing)? code", re.I)
_PHONE_Q = re.compile(r"\b(phone|mobile|contact|cell|whats ?app)\s*(phone\s*)?(number|no\b|no\.|#)"
                      r"|^\W*(mobile|phone|telephone)\W*$|\bmobile phone\b", re.I)
# Questions that mention email / phone / location but ask about skills, not contact details.
_NOT_CONTACT_Q = re.compile(r"experience|\byears?\b|how many|develop|skill|marketing|campaign|\bapps?\b|"
                            r"testing|screening|support", re.I)
_FIRST_NAME_Q = re.compile(r"\bfirst\s*name\b|\bgiven name\b", re.I)
_LAST_NAME_Q = re.compile(r"\b(last|sur|family)\s*name\b|\bsurname\b", re.I)
_FULL_NAME_Q = re.compile(r"^\W*(your\s+)?(full\s+|legal\s+)?name\W*$", re.I)
_CITY_Q = re.compile(r"\b(city|current location|location \(city\)|where are you (based|located)|your location)\b"
                     r"|^\W*location\W*$", re.I)
_NOT_MY_CITY_Q = re.compile(r"prefer|willing|relocat|commut|job location|work location|open to", re.I)
_NOTICE_Q = re.compile(r"notice period|how soon can you (join|start)|when can you (join|start)|earliest "
                       r"(start|joining)|joining (date|time|period)|available to (join|start)|start date", re.I)
_CTC_Q = re.compile(r"\b(ctc|salary|compensation|package|remuneration)\b", re.I)
_EXPECTED_Q = re.compile(r"expect|desired|asking|looking for", re.I)
_CURRENT_Q = re.compile(r"\bcurrent\b|\bpresent\b|\bexisting\b|last drawn", re.I)
_EEO_Q = re.compile(r"\b(gender|race|racial|ethnic\w*|veteran|disab\w*|sexual orientation|pronouns?|religion|"
                    r"caste|marital)\b", re.I)
_DECLINE_OPT = re.compile(r"prefer not|decline|don'?t wish|do not wish|not to (say|disclose|answer|identify)|"
                          r"rather not|choose not", re.I)
_CONSENT_Q = re.compile(r"\b(i agree|agree to|consent|terms (of|and)|privacy (policy|notice)|i accept|"
                        r"acknowledge|i certify|i declare|i confirm|data processing)\b", re.I)
_NO_TICK_Q = re.compile(r"\bfollow\b|newsletter|marketing|promotion|job alerts?|notify me|subscribe|"
                        r"share my (profile|information)|text messages|\bsms\b|similar jobs|talent community", re.I)
_RESUME_Q = re.compile(r"resume|\bcv\b|curriculum", re.I)
_LINKEDIN_Q = re.compile(r"linkedin", re.I)
_PLACEHOLDER_OPT = re.compile(r"^\W*(select( an)?( option)?|choose( one| an option)?|please select|--+|none selected)?\W*$",
                              re.I)


# ---------------------------------------------------------------------------
# Profile facts -> form values (pure helpers, unit tested)
# ---------------------------------------------------------------------------

def phone_parts(phone: str) -> tuple[str, str]:
    """('91', '8083188354') from '+91-8083188354'; ('', digits) when no country code is given."""
    raw = str(phone or "").strip()
    digits = re.sub(r"\D", "", raw)
    if not digits:
        return "", ""
    if raw.startswith("+") or raw.startswith("00"):
        digits = digits[2:] if raw.startswith("00") else digits
        # Indian numbers are 10 digits; keep whatever precedes them as the code.
        if len(digits) > 10:
            return digits[:-10], digits[-10:]
        return "", digits
    if len(digits) == 12 and digits.startswith("91"):
        return "91", digits[2:]
    if len(digits) == 11 and digits.startswith("0"):
        return "", digits[1:]
    return "", digits


def lakhs_of(ctc: str) -> Optional[float]:
    """CTC text -> lakhs per year: '12 LPA' -> 12.0, '12,00,000' -> 12.0, '1.2 Cr' -> 120.0."""
    text = str(ctc or "").lower().replace(",", "")
    if not text.strip():
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(lpa|lakhs?|lacs?|l\b|l\.p\.a)", text)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(cr|crores?)\b", text)
    if m:
        return float(m.group(1)) * 100
    m = re.search(r"\d+(?:\.\d+)?", text)
    if not m:
        return None
    value = float(m.group(0))
    if value >= 10000:  # a full rupee amount
        if re.search(r"month|/\s*m\b|\bpm\b", text):
            value *= 12
        return value / 100000
    return None  # a bare small number: unit unknown


def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.2f}".rstrip("0").rstrip(".")


def ctc_answer(question: str, ctc: str, kind: str) -> Optional[str]:
    """The CTC in the unit the field asks for. Free-text fields get the profile's own wording."""
    lakhs = lakhs_of(ctc)
    q = question.lower()
    if lakhs is None:
        return None
    if re.search(r"lakh|\blacs?\b|\blpa\b|l\.p\.a|in lakhs", q):
        return _fmt(lakhs)
    if re.search(r"per month|monthly|/\s*month|\bpm\b|in hand", q):
        return _fmt(round(lakhs * 100000 / 12))
    if kind == "number" or re.search(r"\binr\b|rupees|₹|\brs\b|per annum|annual|yearly|in numbers|digits", q):
        return _fmt(round(lakhs * 100000))
    return str(ctc).strip()


def notice_days(notice: str) -> Optional[int]:
    """'Immediate' -> 0, '15 days' -> 15, '1 month' -> 30, '2 months' -> 60. None when unclear."""
    t = str(notice or "").lower().strip()
    if not t:
        return None
    if re.search(r"serving|last working|lwd", t):
        return None
    if re.search(r"immediate|immediately|\bnone\b|\bno notice\b|^0\b", t):
        return 0
    m = re.search(r"(\d+)\s*(day|week|month)", t)
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2)
    return n * {"day": 1, "week": 7, "month": 30}[unit]


def _option_days(option: str) -> Optional[tuple[int, int]]:
    """Day range an option stands for: 'Immediate' (0,0), '15 days or less' (0,15), '1 month' (30,30)..."""
    t = option.lower()
    if re.search(r"immediate|\bimmediately\b|\b0 days?\b|right away", t):
        return 0, 0
    nums = [(int(n), u) for n, u in re.findall(r"(\d+)\s*(?:-|to)?\s*(?:\d+\s*)?(day|week|month)", t)]
    if not nums:
        return None
    days = [n * {"day": 1, "week": 7, "month": 30}[u] for n, u in nums]
    lo, hi = min(days), max(days)
    if re.search(r"less than|or less|within|up ?to|under|maximum|max\b", t):
        return 0, hi
    if re.search(r"more than|or more|above|over|\+", t):
        return lo, 10_000
    m = re.search(r"(\d+)\s*(?:-|to)\s*(\d+)\s*(day|week|month)", t)
    if m:
        mult = {"day": 1, "week": 7, "month": 30}[m.group(3)]
        return int(m.group(1)) * mult, int(m.group(2)) * mult
    return lo, hi


def notice_answer(question: str, notice: str, kind: str, options: list[str]) -> Optional[str]:
    days = notice_days(notice)
    if days is None:
        return None
    if options:
        for opt in options:
            span = _option_days(opt)
            if span and span[0] <= days <= span[1]:
                return opt
        return None
    q = question.lower()
    if kind == "number":
        if "month" in q:
            return _fmt(days / 30) if days % 30 == 0 or days == 0 else None
        if "week" in q:
            return _fmt(days / 7) if days % 7 == 0 else None
        return str(days)
    if kind == "date":
        return None  # a joining date is Boss's call, not a conversion
    return str(notice).strip()


def pick_option(answer: str, options: list[str]) -> Optional[str]:
    """The option that is `answer` (exact, then contained, then fuzzy)."""
    from autojob.integrations.browser_service import _best_option

    a = str(answer or "").strip().lower()
    if not a or not options:
        return None
    for o in options:
        if o.strip().lower() == a:
            return o
    idx = _best_option(answer, options)
    return options[idx] if idx is not None else None


def real_options(options: list[str]) -> list[str]:
    """Drop placeholder entries such as 'Select an option'."""
    return [o for o in (options or []) if o and not _PLACEHOLDER_OPT.match(o)]


# ---------------------------------------------------------------------------
# The answerer
# ---------------------------------------------------------------------------

class BoardAnswerer(ScreeningFormFiller):
    """ScreeningFormFiller plus profile rules and one batched, evidence-checked call per form step."""

    def __init__(self, cv_profile: dict, cv_raw_text: str = "", cv_file: Optional[str] = None):
        super().__init__(cv_profile, cv_raw_text)
        self.cv_file = cv_file

    # -- rules -----------------------------------------------------------

    def _p(self, key: str) -> str:
        value = self.cv_profile.get(key)
        return "" if value in (None, [], {}) else str(value).strip()

    def rule_answer(self, field: dict, step_fields: Optional[list[dict]] = None) -> Optional[Any]:
        """A value straight from the profile, or None when no rule covers this field."""
        kind = field.get("kind", "text")
        q = " ".join(str(field.get("question") or "").split())
        options = real_options(field.get("options") or [])
        if not q:
            return None

        if kind == "file":
            if _RESUME_Q.search(q) and self.cv_file:
                return UPLOAD_CV
            return None

        if kind == "checkbox":
            if _NO_TICK_Q.search(q):
                return False
            if _CONSENT_Q.search(q):
                return True
            return None

        if _EEO_Q.search(q) and options:
            return next((o for o in options if _DECLINE_OPT.search(o)), None)

        if _COUNTRY_CODE_Q.search(q):
            code, _ = phone_parts(self._p("phone"))
            if not code or not options:
                return None
            return next((o for o in options if re.search(rf"\+\s?{code}\b|\(\+?{code}\)", o)), None)

        if not _NOT_CONTACT_Q.search(q):
            value = self._contact_answer(q, kind, options, step_fields or [])
            if value is not None:
                return value

        if _NOTICE_Q.search(q) and kind in (*_TEXT_KINDS, *_CHOICE_KINDS):
            return notice_answer(q, self._p("notice_period"), kind, options)

        if _CTC_Q.search(q) and kind in ("text", "number", "textarea"):
            if _EXPECTED_Q.search(q):
                return ctc_answer(q, self._p("expected_ctc"), kind)
            if _CURRENT_Q.search(q):
                return ctc_answer(q, self._p("current_ctc"), kind)
            return None

        if kind in ("text", "number"):
            total = self._deterministic(q)
            if total is not None:
                return total
        return None

    def _contact_answer(self, q: str, kind: str, options: list[str], step_fields: list[dict]) -> Optional[str]:
        """Email, phone, name, city and LinkedIn URL from the profile."""
        if _EMAIL_Q.search(q) and kind in (*_TEXT_KINDS, "select"):
            email = self._p("email")
            if not email:
                return None
            if options:
                return next((o for o in options if email.lower() in o.lower()), None)
            return email

        if _PHONE_Q.search(q) and kind in ("text", "tel", "number"):
            code, national = phone_parts(self._p("phone"))
            if not national:
                return None
            has_code_field = any(_COUNTRY_CODE_Q.search(str(f.get("question") or "")) for f in step_fields)
            if kind == "number" or has_code_field or not code:
                return national
            return f"+{code} {national}"

        name = self._p("name")
        if name and kind in ("text", "combobox"):
            parts = name.split()
            if _FIRST_NAME_Q.search(q):
                return parts[0]
            if _LAST_NAME_Q.search(q):
                return parts[-1] if len(parts) > 1 else None
            if _FULL_NAME_Q.search(q):
                return name

        if _CITY_Q.search(q) and not _NOT_MY_CITY_Q.search(q) and kind in ("text", "combobox"):
            loc = self._p("location")
            if not loc:
                return None
            # Typeaheads match the city name best ("Surat" -> "Surat, Gujarat, India").
            return loc.split(",")[0].strip() if kind == "combobox" else loc

        if _LINKEDIN_Q.search(q) and kind == "text" and re.search(r"url|profile|link", q, re.I):
            return self._p("linkedin") or None
        return None

    # -- the whole step --------------------------------------------------

    def _cache_key(self, field: dict) -> str:
        return "|".join([
            field.get("kind", ""),
            " ".join(str(field.get("question") or "").lower().split()),
            "|".join(o.lower() for o in real_options(field.get("options") or [])),
        ])

    def answer_fields(self, fields: list[dict], step_fields: Optional[list[dict]] = None) -> dict[str, Any]:
        """
        {field id: value or None} for `fields` (rules, cache, then one model call).
        step_fields: every field on the page, for context (e.g. a separate country-code box).
        """
        out: dict[str, Any] = {}
        todo: list[dict] = []
        for f in fields:
            fid = str(f["id"])
            value = self.rule_answer(f, step_fields or fields)
            if value is None and f.get("kind") in ("text", "number"):
                value = self.skill_years(str(f.get("question") or ""))
            if value is not None:
                out[fid] = value
                continue
            key = self._cache_key(f)
            if key in self._cache:
                out[fid] = self._cache[key]
                continue
            if f.get("kind") == "file":
                out[fid] = None
                continue
            todo.append(f)

        if todo:
            for fid, value in self._answer_batch(todo).items():
                out[fid] = value
            for f in todo:
                self._cache[self._cache_key(f)] = out.get(str(f["id"]))

        for f in fields:
            if out.get(str(f["id"])) is None and f.get("required"):
                value = self._ask_boss_live(f)
                if value is not None:
                    out[str(f["id"])] = value
                    self._cache[self._cache_key(f)] = value
                    continue
                self._remember_unanswered(str(f.get("question") or ""), real_options(f.get("options") or []))
        return out

    def _ask_boss_live(self, f: dict) -> Any:
        """A required field the data does not answer: ask Boss now (None if no answer in time)."""
        from autojob.jobs.live_questions import ask_boss

        kind = f.get("kind", "text")
        if kind in ("file", "checkboxes"):
            return None
        question = " ".join(str(f.get("question") or "").split())
        options = ["Yes", "No"] if kind == "checkbox" else real_options(f.get("options") or [])
        if kind in _CHOICE_KINDS and not options:
            return None
        answer = ask_boss(question, options)
        if answer is None:
            return None
        if kind == "checkbox":
            return answer == "Yes"
        if kind == "number":
            nums = re.findall(r"\d+(?:\.\d+)?", answer)
            return nums[0] if len(nums) == 1 else None
        return answer

    def _answer_batch(self, fields: list[dict]) -> dict[str, Any]:
        from autojob.core.llm_router import fast_chat

        items = []
        for f in fields:
            kind = f.get("kind", "text")
            item = {"id": str(f["id"]), "question": str(f.get("question") or "")[:300], "type": kind}
            opts = real_options(f.get("options") or [])
            if kind == "checkbox":
                item["options"] = ["Yes", "No"]
            elif opts:
                item["options"] = opts[:40]
            if f.get("error"):
                item["form_error"] = str(f["error"])[:160]
            items.append(item)

        prompt = f"""Answer job-application form questions for the candidate using ONLY the data below.
Never guess, estimate, round up or invent anything. If the data does not clearly state an answer, it is not answerable.

{self._data_block()}

QUESTIONS (JSON; when "options" is given the answer must be exactly one of them):
{json.dumps(items, ensure_ascii=False, indent=1)}

Reply with ONLY one JSON object, no other text, with every question id:
{{"answers": {{"<id>": {{"answerable": true or false, "answer": "<the answer>", "evidence": ["<short exact quote copied from the data above>", ...]}}}}}}

Rules:
{self._EVIDENCE_RULES}
- type "number": digits only (a decimal point is allowed), no units.
- type "checkbox": "Yes" to tick it, "No" to leave it unticked.
- type "checkboxes": every option that applies, separated by " | ", each one an exact option.
- type "date": YYYY-MM-DD.
- Open questions (summary, cover letter, why this role, about yourself): 2-3 short first-person sentences made
  only of facts in the data; quote those facts as evidence.
- form_error is the website's complaint about an earlier answer: give an answer that fixes it, or mark it not
  answerable if the true answer cannot satisfy it."""

        try:
            raw = fast_chat(
                prompt=prompt,
                task_type="general",
                system_prompt="You fill job applications truthfully from the candidate's own data. Output only JSON.",
                temperature=0.0,
                max_tokens=3000,  # reasoning models think before the JSON
            )
        except Exception as exc:
            logger.error("Batch screening answer failed: %s", exc)
            return {str(f["id"]): None for f in fields}

        answers = self._parse_batch(raw)
        out: dict[str, Any] = {}
        for f in fields:
            fid = str(f["id"])
            out[fid] = self._accept(f, answers.get(fid))
        return out

    @staticmethod
    def _parse_batch(raw: str) -> dict[str, dict]:
        text = re.sub(r"<think>.*?(</think>|$)", "", raw or "", flags=re.S)
        text = re.sub(r"```(?:json)?", "", text)
        for m in re.finditer(r"\{", text):
            end = text.rfind("}")
            if end <= m.start():
                break
            try:
                obj = json.loads(text[m.start():end + 1])
            except Exception:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("answers"), dict):
                return {str(k): v for k, v in obj["answers"].items() if isinstance(v, dict)}
        return {}

    def _accept(self, field: dict, data: Optional[dict]) -> Any:
        """Verify one model answer and convert it to the field's value type."""
        kind = field.get("kind", "text")
        question = " ".join(str(field.get("question") or "").split())
        options = real_options(field.get("options") or [])

        if kind == "checkbox":
            value = self._verify_answer(question, ["Yes", "No"], data)
            return None if value is None else value == "Yes"

        if kind == "checkboxes":
            raw = dict(data or {})
            parts = [p.strip() for p in str(raw.get("answer") or "").split("|") if p.strip()]
            chosen = []
            for part in parts:
                value = self._verify_answer(question, options, {**raw, "answer": part})
                if value is not None and value not in chosen:
                    chosen.append(value)
            return chosen or None

        if kind in _CHOICE_KINDS:
            return self._verify_answer(question, options, data) if options else None

        value = self._verify_answer(question, [], data, max_len=600 if kind == "textarea" else 300)
        if value is None:
            return None
        if kind == "number":
            nums = re.findall(r"\d+(?:\.\d+)?", value)
            if len(nums) != 1:
                logger.warning("Rejected non-numeric answer %r for '%s'", value, question[:80])
                return None
            value = nums[0]
        if kind == "date" and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return None
        return value

    # -- after the site rejected an answer ---------------------------------

    def fix_for_error(self, field: dict, previous: Any) -> Optional[Any]:
        """
        A simple fix the site's error message asks for ("Enter a whole number"),
        or None so the caller re-asks the model with the error attached.
        """
        error = str(field.get("error") or "").lower()
        prev = str(previous if previous is not None else "").strip()
        if not error:
            return None
        if re.search(r"whole number|integer", error) and re.fullmatch(r"\d+\.\d+", prev):
            return str(int(float(prev)))  # completed years: never round up
        if _CTC_Q.search(str(field.get("question") or "")) and re.fullmatch(r"\d+(\.\d+)?", prev):
            m = re.search(r"between\s+[\d.]+\s+and\s+([\d.]+)|less than\s+([\d.]+)|at most\s+([\d.]+)", error)
            limit = float(next(g for g in m.groups() if g)) if m else None
            if limit is not None and float(prev) > limit and float(prev) >= 100000:
                return _fmt(float(prev) / 100000)  # the field wants lakhs
        if re.search(r"valid (email|e-mail)", error):
            return self._p("email") or None
        if re.search(r"valid (phone|mobile)|phone number", error):
            return phone_parts(self._p("phone"))[1] or None
        return None
