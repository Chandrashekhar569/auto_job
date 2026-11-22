"""
Ask Boss a screening question live, in the middle of an application.

When a form asks something the CV / profile / saved answers do not cover (each
company asks different things: "years of experience in RAG", "in Databricks"...),
the apply run no longer gives up on the job. It:

    1. tells Boss the question by voice (System Notification into the live session),
    2. waits up to ASK_TIMEOUT_S for Boss's answer (tool answer_job_question),
    3. types that answer, and saves it to the job agent's answers so the next
       company asking the same thing is answered without asking again.

Nothing is guessed: no answer in time -> None, and the caller hands the job
back exactly as before (questions_waiting.md). main.py registers the notifier;
without one (tests, scripts) ask_boss returns None at once.
"""

import logging
import re
import threading
from typing import Any, Callable, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

ASK_TIMEOUT_S = 300

_notifier: Optional[Callable[[str], None]] = None
_lock = threading.Lock()       # one question at a time
_state_lock = threading.Lock()
_pending: Optional[dict] = None  # {"question", "options", "event", "answer"}


def set_notifier(fn: Optional[Callable[[str], None]]) -> None:
    """main.py: fn(text) delivers a System Notification to the live voice session."""
    global _notifier
    _notifier = fn


def pending_question() -> Optional[dict]:
    with _state_lock:
        return {"question": _pending["question"], "options": _pending["options"]} if _pending else None


def _match_option(answer: str, options: list[str]) -> Optional[str]:
    a = answer.strip().lower()
    exact = next((o for o in options if o.strip().lower() == a), None)
    if exact:
        return exact
    hits = [o for o in options if a and (a in o.lower() or o.lower() in a)]
    if len(hits) == 1:
        return hits[0]
    # "2" for "2 years" / range options like "1-3 years"
    if re.fullmatch(r"\d+(\.\d+)?", a):
        n = float(a)
        for o in options:
            m = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to|–)\s*(\d+(?:\.\d+)?)", o)
            if m and float(m.group(1)) <= n <= float(m.group(2)):
                return o
            m = re.search(r"(\d+(?:\.\d+)?)\s*\+", o)
            if m and n >= float(m.group(1)):
                return o
    return None


def ask_boss(question: str, options: Optional[list[str]] = None, company: str = "",
             timeout: float = ASK_TIMEOUT_S) -> Optional[str]:
    """
    Ask Boss and wait. Returns the answer (for options: the exact option text)
    or None if nobody can be asked, Boss did not answer in time, or the answer
    does not fit the options.
    """
    global _pending
    question = " ".join(str(question or "").split())
    options = [o for o in (options or []) if str(o).strip()]
    if not question or _notifier is None:
        return None
    if not _lock.acquire(timeout=timeout):
        return None
    try:
        event = threading.Event()
        with _state_lock:
            _pending = {"question": question, "options": options, "event": event, "answer": None}
        opts = f"\nOptions: {' | '.join(options[:12])}" if options else ""
        where = f" for {company}" if company else ""
        try:
            _notifier(
                "System Notification: A job application" + where + " is waiting on one screening question that "
                f"Boss's CV and profile do not answer.\nQuestion: {question}{opts}\n\n"
                "Instructions: Ask Boss this question now, in one short line (read the options if any). When Boss "
                "answers, call answer_job_question with exactly what Boss said. If Boss says skip, call "
                f"answer_job_question with answer='skip'. Never answer it yourself. It waits {int(timeout // 60)} "
                "minutes, then the job is left for later.")
        except Exception as exc:
            logger.warning("Could not ask Boss: %s", exc)
            return None
        logger.info("Asked Boss live: %s", question[:120])
        if not event.wait(timeout):
            logger.info("No answer from Boss in %ds: %s", timeout, question[:80])
            return None
        with _state_lock:
            answer = (_pending or {}).get("answer")
        if not answer or answer.strip().lower() in ("skip", "skip it", "leave it", "don't know", "dont know"):
            return None
        if options:
            picked = _match_option(answer, options)
            if picked is None:
                logger.warning("Boss's answer %r is not one of the options %s", answer, options)
            return picked
        return answer.strip()
    finally:
        with _state_lock:
            _pending = None
        _lock.release()


@tool("answer_job_question")
def answer_job_question(answer: str) -> dict[str, Any]:
    """
    Boss's answer to the screening question a running job application just asked (it arrives as a System Notification). Pass exactly what Boss said, e.g. '2', '2 years', 'Yes', 'Bengaluru'; 'skip' leaves the question. The application continues with it, and the answer is saved so the same question is answered automatically next time.
    """
    with _state_lock:
        p = _pending
        if p is None:
            return {"error": "No job application is waiting for an answer right now."}
        p["answer"] = " ".join(str(answer or "").split())
        question, event = p["question"], p["event"]
    if p["answer"] and p["answer"].lower() not in ("skip", "skip it", "leave it"):
        try:
            from autojob.jobs.job_deep_agent import teach

            teach(f"{question} -> {p['answer']}", kind="answer")
        except Exception as exc:
            logger.warning("Could not save the answer: %s", exc)
    event.set()
    return {"status": "ok", "summary": "Got it - continuing the application with that answer."}


LIVE_QUESTION_TOOLS = [answer_job_question]
