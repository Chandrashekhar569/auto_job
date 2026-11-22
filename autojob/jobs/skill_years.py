"""
"How many years of experience do you have in <X>?" - answered from Boss's own data.

Every company asks about different skills, and the CV lists skills without
years, so the evidence-checked answerer could not quote a number and stopped to
ask Boss for each one. This answers them deterministically, in this order:

    1. Boss's table   ABOUT_ME.md "## Years of experience by skill" ("- RAG: 1.5 years")
    2. absent         <X> appears nowhere in the profile/CV -> "0" (Boss's rule, 2026-11-05)
    3. AI family      an AI/GenAI tool that IS on the CV (LangChain, Vertex AI, Azure OpenAI...)
                      -> Boss's "Generative AI / LLM" figure from the table
    4. dated roles    the roles in "## Work history" whose text mentions <X>: their
                      months are added up (overlaps merged) and rounded DOWN to
                      completed years ("0.x" under one year)

Anything else returns None and goes to the evidence-checked answerer / Boss.
"""

import logging
import re
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)

SKILL_YEARS_Q = re.compile(
    r"\b(?:how many )?(?:years?|yrs)(?: of)?(?: (?:work|hands[- ]on|relevant|professional))? (?:experience|exp)"
    r"(?: do)?(?: you)?(?: have)?(?: working)? (?:in|with|on|into|using)\s+(.+?)\s*\??$"
    r"|\b(?:relevant )?(?:experience|exp) (?:in|with|on|into)\s+(.+?)\s*\??$", re.I)
_STOP = {"and", "or", "the", "with", "for", "in", "of", "on", "using", "development", "experience", "years",
         "skills", "skill", "tools", "tool", "technologies", "technology", "framework", "frameworks", "platform",
         "services", "service", "working", "hands", "do", "you", "have"}
AI_FAMILY = re.compile(
    r"\b(gen ?ai|generative|llms?|large language|rag|retrieval|langchain|langgraph|langsmith|llamaindex|agents?|"
    r"agentic|prompt|vertex|ai foundry|azure ai|openai|gpt|gemini|claude|anthropic|bedrock|vector|embeddings?|"
    r"mcp|hugging ?face|transformers|copilot|ragas|fine[- ]?tun|lora|qlora|peft|vllm|qdrant|pgvector|nlp)\b", re.I)
_GENAI_ROW = re.compile(r"generative ai|llm", re.I)
_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
                                       "nov", "dec"], 1)}


def _norm(text: str) -> str:
    return " " + " ".join(t.strip(".") for t in re.sub(r"[^a-z0-9+#.]+", " ", str(text).lower()).split()) + " "


def _skill_of(question: str) -> Optional[str]:
    m = SKILL_YEARS_Q.search(" ".join(str(question or "").split()))
    if not m:
        return None
    skill = (m.group(1) or m.group(2) or "").strip()
    if re.search(r"\b(domain|industry|team|managing|management|people|leading|client|onsite|shift)\b", skill, re.I):
        return None  # not a skill/tool: domain or role questions go to the answerer / Boss
    return skill or None


def _words(skill: str) -> list[str]:
    main = re.split(r"[(\[]", skill)[0]
    return [w for w in _norm(main).split() if len(w) >= 2 and w not in _STOP]


def _stem(w: str) -> str:
    """'troubleshooting' and 'troubleshot' share 'troubles'; short words must match whole."""
    return w[:8] if len(w) >= 9 else w


def _has_word(w: str, tokens: set) -> bool:
    if w in tokens:
        return True
    st = _stem(w)
    return st != w and any(t.startswith(st) for t in tokens)


def _section(text: str, name: str) -> str:
    m = re.search(rf"(?ms)^## {re.escape(name)}[^\n]*\n(.*?)(?=^## |\Z)", text)
    return m.group(1) if m else ""


def _fmt_years(years: float) -> str:
    if years >= 1:
        return str(int(years))  # completed years, never rounded up
    return str(int(years * 10) / 10)


def _table_value(about: str, words: list[str]) -> Optional[str]:
    for line in _section(about, "Years of experience by skill").splitlines():
        m = re.match(r"^-\s*(.+?):\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)?\s*$", line.strip(), re.I)
        if not m:
            continue
        names = {w for part in re.split(r"[/(),]", m.group(1)) for w in _words(part)}
        if words and all(w in names for w in words):
            return m.group(2)
    return None


def _genai_value(about: str) -> Optional[str]:
    for line in _section(about, "Years of experience by skill").splitlines():
        m = re.match(r"^-\s*(.+?):\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)", line.strip(), re.I)
        if m and _GENAI_ROW.search(m.group(1)):
            return m.group(2)
    return None


def _month_index(mon: str, year: str) -> int:
    return int(year) * 12 + _MONTHS[mon[:3].lower()] - 1


def _role_months(about: str, words: list[str], today: Optional[date] = None) -> int:
    today = today or date.today()
    spans = []
    for block in re.split(r"(?m)^### ", _section(about, "Work history (newest first)"))[1:]:
        m = re.search(r"(?i)\b([a-z]{3})[a-z]* (\d{4})\s*[–-]\s*(?:([a-z]{3})[a-z]* (\d{4})|present)", block)
        if not m:
            continue
        start = _month_index(m.group(1), m.group(2))
        end = _month_index(m.group(3), m.group(4)) if m.group(3) else today.year * 12 + today.month - 1
        body = block.partition("\n")[2]  # not the title line: "Engineer" is no skill
        nb = _norm(body)
        # One word: any form of it ("troubleshot"). Several words: the phrase itself,
        # so "data ... prompt-engineering" is not "Data Engineering".
        hit = _has_word(words[0], set(nb.split())) if len(words) == 1 else f" {' '.join(words)} " in nb
        if hit:
            spans.append((start, end + 1))
    months, last = 0, -1
    for s, e in sorted(spans):  # merge overlapping roles
        s = max(s, last)
        if e > s:
            months += e - s
            last = e
    return months


def answer_skill_years(question: str, about: str, cv_text: str = "", extra: str = "") -> Optional[str]:
    """The years to give for a "years of experience in <X>" question, or None."""
    skill = _skill_of(question)
    if not skill:
        return None
    words = _words(skill)
    if not words:
        return None

    value = _table_value(about, words)
    if value is not None:
        return value

    corpus = _norm(" ".join([about, cv_text, extra]))
    known = set(corpus.split())
    if not any(_has_word(w, known) for w in words):
        logger.info("Skill-years: '%s' not in CV/profile -> 0 (Boss's rule)", skill)
        return "0"

    # Only when the tool itself is on the CV ("Azure AI Search" is not "Azure AI Foundry").
    if AI_FAMILY.search(skill) and f" {' '.join(words)} " in corpus:
        genai = _genai_value(about)
        if genai is not None:
            logger.info("Skill-years: '%s' is an AI/GenAI tool on the CV -> GenAI figure %s", skill, genai)
            return genai

    months = _role_months(about, words)
    if months:
        logger.info("Skill-years: '%s' from dated roles = %d months", skill, months)
        return _fmt_years(months / 12)
    return None
