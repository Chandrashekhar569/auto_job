"""
One-shot application-form filler for the Playwright browser session.

The step-by-step vision agent in browser_service spends one model call per
click and often stalls on long company forms (it re-read the same Zoho Recruit
form several times without typing anything). Research on web agents points the
same way: Agent-E and Browser Use hand the model a distilled DOM/accessibility
view instead of raw pixels, and WebVoyager/OpenWebVoyager show that pairing it
with a screenshot works best (see https://arxiv.org/pdf/2410.19609). So here:

    1. collect   every visible form field on the page and in its iframes, with
                 its real question text, required flag and options;
    2. decide    ONE model call maps the candidate profile / CV / saved answers
                 to those fields (null where the facts do not cover it);
    3. fill      Playwright types, selects and ticks the answers.

It never clicks a button, never touches password or file inputs, and never
guesses: required fields it cannot answer are returned so the caller can hand
them to the user. Submit / account-creation clicks stay behind the usual
confirmation in browser_service._run_actions.
"""

import json
import logging
import re
from typing import Any

from autojob.integrations import browser_service as bs

logger = logging.getLogger(__name__)

_FIELD_SELECTOR = (
    "input:not([type=hidden]):not([type=password]):not([type=file]):not([type=submit])"
    ":not([type=button]):not([type=image]):not([type=reset]), textarea, select, "
    "[role=combobox], [role=textbox], [role=radio], [role=checkbox]"
)

# For each stamped field (data-vro-ref): the question a person would read next
# to it, whether it is required, and for radios the group question + option.
_JS_FIELD_INFO = """
(refs) => {
  const out = {};
  const text = (n) => (n && (n.innerText || n.textContent) || '').replace(/\\s+/g, ' ').trim();
  const byIds = (ids) => (ids || '').split(/\\s+/).map(i => text(document.getElementById(i))).filter(Boolean).join(' ');
  for (const ref of refs) {
    const el = document.querySelector(`[data-vro-ref="${ref}"]`);
    if (!el) continue;
    const type = (el.getAttribute('type') || '').toLowerCase();
    const isChoice = type === 'radio' || el.getAttribute('role') === 'radio';
    let own = el.getAttribute('aria-label') || byIds(el.getAttribute('aria-labelledby'))
      || (el.labels && el.labels[0] && text(el.labels[0])) || '';
    let question = isChoice ? '' : own;
    if (!question || isChoice) {
      // Walk up to the block that holds exactly this question.
      let node = el.parentElement;
      for (let d = 0; node && d < 5; d++, node = node.parentElement) {
        const legend = node.querySelector(':scope > legend, :scope > label, :scope > .label, :scope > [class*=label], :scope > [class*=question]');
        const group = node.getAttribute('role') === 'radiogroup' || node.getAttribute('role') === 'group'
          ? (node.getAttribute('aria-label') || byIds(node.getAttribute('aria-labelledby'))) : '';
        const t = group || text(legend);
        if (t && t !== own) { question = t; break; }
      }
    }
    const container = el.closest('fieldset, [class*=field], [class*=form-group], [class*=question]') || el.parentElement;
    const req = el.required || el.getAttribute('aria-required') === 'true'
      || /\\*/.test(question) || (text(container).length < 200 && /\\*/.test(text(container)));
    out[ref] = {
      question: question.slice(0, 200),
      option: isChoice ? (own || text(el.parentElement)).slice(0, 80) : '',
      group: isChoice ? (el.getAttribute('name') || question).slice(0, 80) : '',
      required: !!req,
    };
  }
  return out;
}
"""

_SYSTEM_PROMPT = """You fill in a job application form for the candidate described below.
For each FIELD return the value to enter, using ONLY facts in CANDIDATE PROFILE, ABOUT THE CANDIDATE and
ANSWERS THE USER GAVE. Never invent or estimate experience, skills, counts, degrees, salary, dates or
reasons. If the facts do not answer a field, return null for it.

Rules:
- Return ONLY JSON: {"answers": {"<id>": {"value": <value>, "evidence": "<short quote copied word for word from
  the facts that proves the value>"}, ...}} with every field id present. Use {"value": null} when unsure.
- Wrong information is worse than a blank field: if a fact is missing, ambiguous or would need a guess, value null.
- Text fields: the answer value only, never an explanation. Numbers as digits (e.g. "5").
- Dropdowns / radio groups: return exactly one of the listed options (copy its text).
- Checkboxes: true or false. Consent / terms-of-processing boxes: true. Newsletter or marketing boxes: false.
- Voluntary diversity / EEO / disability / veteran questions: the "prefer not to say" / "decline" option if
  listed, otherwise null.
- Phone: digits as in the profile. Salary: use the profile's CTC fields, in the unit the field asks for.
- Cover letter / "why do you want this job" free text: 2-3 honest sentences built only from the candidate facts.
- Fields that already hold a correct value need no change: return that same value."""


def _stamp_and_describe(sess) -> list[dict]:
    """Snapshot (stamps data-vro-ref, frames included) and keep real form fields. Owner thread."""
    sess._snap()
    ref_frames = getattr(sess, "_ref_frames", {})
    by_frame: dict[int, tuple[Any, list[int]]] = {}
    fields = []
    for e in sess._snapshot:
        tag, typ = e.get("tag"), (e.get("type") or "")
        is_field = (
            tag in ("textarea", "select")
            or (tag == "input" and typ not in ("password", "file", "submit", "button", "image", "reset"))
            or e.get("role") in ("combobox", "textbox", "radio", "checkbox")
        )
        if not is_field or e.get("disabled") or e.get("isPassword"):
            continue
        # Entries of an open custom dropdown are options, not fields.
        if e.get("role") in ("option", "menuitem", "menuitemradio"):
            continue
        fields.append(e)
        frame = ref_frames.get(e["ref"])
        by_frame.setdefault(id(frame), (frame, []))[1].append(e["ref"])

    info: dict[str, dict] = {}
    for frame, refs in by_frame.values():
        try:
            info.update((frame or sess._page).evaluate(_JS_FIELD_INFO, refs) or {})
        except Exception as exc:
            logger.debug("Field info failed: %s", exc)
    for e in fields:
        e.update(info.get(str(e["ref"]), {}))
    return fields


def _group_fields(fields: list[dict]) -> list[dict]:
    """Merge radio buttons of one question into a single field with options."""
    out, groups = [], {}
    for e in fields:
        if e.get("type") == "radio" or e.get("role") == "radio":
            key = e.get("group") or e.get("question") or str(e["ref"])
            if key in groups:
                groups[key]["choices"][e.get("option") or str(e["ref"])] = e["ref"]
                groups[key]["checked"] = groups[key]["checked"] or bool(e.get("checked"))
                continue
            g = {**e, "kind": "radio", "choices": {e.get("option") or str(e["ref"]): e["ref"]},
                 "checked": bool(e.get("checked"))}
            groups[key] = g
            out.append(g)
        else:
            out.append(e)
    return out


def _describe_for_model(f: dict) -> dict:
    d = {"id": f["ref"], "question": (f.get("question") or f.get("name") or "").replace("*", "").strip(), "required": bool(f.get("required"))}
    if f.get("kind") == "radio":
        d["type"] = "radio"
        d["options"] = list(f["choices"])
    elif f.get("tag") == "select":
        d["type"] = "dropdown"
        d["options"] = f.get("options") or []
    elif f.get("type") == "checkbox" or f.get("role") == "checkbox":
        d["type"] = "checkbox"
    elif f.get("popup") or f.get("role") == "combobox":
        d["type"] = "searchable dropdown"
    else:
        d["type"] = f.get("type") or ("textarea" if f.get("tag") == "textarea" else "text")
    if f.get("value") and d["type"] not in ("radio", "checkbox"):
        d["current_value"] = f["value"]
    return d


def _already_done(f: dict) -> bool:
    if f.get("kind") == "radio":
        return bool(f.get("checked"))
    if f.get("type") == "checkbox" or f.get("role") == "checkbox":
        return False
    if f.get("tag") == "select":
        return bool(f.get("value")) and (f.get("options") or [""])[0:1] != [f.get("value")]
    return bool(str(f.get("value") or "").strip())


def _ask_model(fields: list[dict], facts: str) -> dict:
    from autojob.core.llm_router import fast_chat

    prompt = f"{facts}\n\nFIELDS:\n" + json.dumps([_describe_for_model(f) for f in fields], ensure_ascii=False)
    raw = fast_chat(prompt=prompt, task_type="browser", system_prompt=_SYSTEM_PROMPT, temperature=0.0)
    parsed = bs._repair_json(raw) or {}
    answers = parsed.get("answers", parsed)
    return answers if isinstance(answers, dict) else {}


_DECLINE = re.compile(r"prefer not|decline|do not wish|don.t wish|not to (say|disclose|answer)", re.I)


def _norm(text: Any) -> str:
    return " " + re.sub(r"[^a-z0-9+#.@%]+", " ", str(text).lower()).strip() + " "


def _quote_in(quote: str, corpus: str) -> bool:
    q = _norm(quote).strip()
    if len(q) < 2:
        return False
    if f" {q} " in corpus:
        return True
    words = q.split()
    return len(words) >= 3 and sum(f" {w} " in corpus for w in words) / len(words) >= 0.85


def _checked_value(f: dict, raw: Any, corpus: str) -> Any:
    """
    The value to enter, or None. Unless the field is a checkbox, a free-text
    answer (textarea) or a "prefer not to say" choice, the model's evidence
    quote must really be in the candidate facts, and every number in the value
    must appear in that quote - otherwise the field is left for Boss.
    """
    if isinstance(raw, dict):
        value, evidence = raw.get("value"), str(raw.get("evidence") or "")
    else:  # plain value without evidence
        value, evidence = raw, ""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if f.get("type") == "checkbox" or f.get("role") == "checkbox" or isinstance(value, bool):
        return value
    if f.get("tag") == "textarea" or _DECLINE.search(str(value)):
        return value
    if not _quote_in(evidence, corpus):
        logger.warning("Dropped unproven answer for '%s': %r (evidence %r)",
                       (f.get("question") or "")[:60], value, evidence[:80])
        return None
    if not _numbers_ok(str(value), evidence):
        logger.warning("Dropped untraceable number in '%s': %r", (f.get("question") or "")[:60], value)
        return None
    return value


def _numbers_ok(value: str, evidence: str) -> bool:
    """Every number in the value is in the evidence; a range option ("5-7 years", "7+") must contain one."""
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", evidence)]
    rng = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to|–)\s*(\d+(?:\.\d+)?)", value)
    if rng:
        lo, hi = float(rng.group(1)), float(rng.group(2))
        return any(lo <= n <= hi for n in nums)
    plus = re.search(r"(\d+(?:\.\d+)?)\s*\+", value)
    if plus:
        return any(n >= float(plus.group(1)) for n in nums)
    ev_digits = re.sub(r"\D", "", evidence)
    return all(n in ev_digits for n in re.findall(r"\d+", value))


def _apply(sess, f: dict, value: Any) -> str:
    """Put one answer into the page (owner thread). Returns what was done."""
    if f.get("kind") == "radio":
        idx = bs._best_option(str(value), list(f["choices"]))
        if idx is None:
            raise ValueError(f"No option like '{value}'. Options: {list(f['choices'])}")
        label = list(f["choices"])[idx]
        sess._find(f["choices"][label]).check(timeout=bs.DEFAULT_TIMEOUT_MS)
        return label
    el = sess._find(f["ref"])
    el.scroll_into_view_if_needed(timeout=5000)
    if f.get("type") == "checkbox" or f.get("role") == "checkbox":
        want = value is True or str(value).strip().lower() in ("true", "yes")
        el.check() if want else el.uncheck()
        return str(want)
    if f.get("tag") == "select" or f.get("popup") or f.get("role") == "combobox":
        return bs._choose_option(sess, el, str(value))
    bs._fast_fill(sess, el, str(value))
    return str(value)[:40]


def fill_form_page(sess, facts: str) -> dict[str, Any]:
    """
    Fill every answerable field on the current page. `facts` is the trusted
    candidate text (profile, CV, saved answers). Returns
    {"filled": [...], "failed": [...], "unanswered_required": [question, ...], "fields_seen": n}.
    """
    fields = _group_fields(sess.submit(lambda: _stamp_and_describe(sess)))
    todo = [f for f in fields if not _already_done(f)]
    if not todo:
        return {"filled": [], "failed": [], "unanswered_required": [], "fields_seen": len(fields)}

    answers = _ask_model(todo, facts)
    corpus = _norm(facts)
    filled, failed, missing = [], [], []
    for f in todo:
        question = (f.get("question") or f.get("name") or str(f["ref"]))[:80]
        value = _checked_value(f, answers.get(str(f["ref"]), answers.get(f["ref"])), corpus)
        if value is None or (isinstance(value, str) and not value.strip()):
            if f.get("required"):
                missing.append(question)
            continue
        try:
            what = sess.submit(lambda f=f, v=value: _apply(sess, f, v))
            filled.append(f"{question} = {what}")
        except Exception as exc:
            failed.append(f"{question}: {bs._short_error({'error': str(exc)})}")
            if f.get("required"):
                missing.append(question)
    return {"filled": filled, "failed": failed, "unanswered_required": missing, "fields_seen": len(fields)}
