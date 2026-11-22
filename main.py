"""
Auto Job - standalone job-application backend (no UI, no voice).

The same job agents as Veronica (Naukri / Indeed / LinkedIn quick-apply, and
company career portals), driven from the terminal. Every account creation and
every company-site Submit still waits for your yes here, and a continuous board
run asks one yes for the whole plan.

    python main.py                         interactive shell (type 'help')
    python main.py board naukri            continuous quick-apply on one board
    python main.py url <job-url>           apply to one job link
    python main.py auto "AI Engineer"      Naukri search + company-site jobs
    python main.py saved                   apply to saved company-site jobs
    python main.py report                  what has been applied so far
    python main.py teach "Notice period: 30 days" --answer
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Load .env before any job module reads os.environ at import time.
from autojob.core import llm_router  # noqa: E402,F401

from autojob.jobs import continuous_apply as ca  # noqa: E402
from autojob.jobs import job_apply_service as jas  # noqa: E402
from autojob.jobs import live_questions  # noqa: E402

log = logging.getLogger("autojob")


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

# Tool results carry instructions meant for the voice LLM ("call confirm_job_step...").
_LLM_SENTENCE = re.compile(
    r"[^.]*\b(call \w+\(|call (confirm|resume|start|answer|teach|apply)_\w+|ask the user|read this plan|"
    r"tell boss|end your turn|ask boss)[^.]*\.?", re.I)


def _clean(text: str) -> str:
    return re.sub(r"\s{2,}", " ", _LLM_SENTENCE.sub("", str(text or ""))).strip()


def _yes(prompt: str) -> bool:
    while True:
        try:
            ans = input(f"{prompt} [y/n]: ").strip().lower()
        except EOFError:
            return False
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False


def _show(result: dict) -> None:
    if not isinstance(result, dict):
        print(result)
        return
    if result.get("error"):
        print(f"\n[x] {result['error']}")
        return
    print(f"\n== {result.get('status', 'done')} ==")
    if result.get("summary"):
        print(_clean(result["summary"]))
    for key, label in (("applied_this_run", "Applied"), ("skipped_for_questions", "Skipped (questions)"),
                       ("failed", "Failed"), ("questions_for_boss", "Questions for you"),
                       ("dry_run_filled", "Filled in dry run")):
        items = result.get(key) or []
        if items:
            print(f"\n{label}:")
            for it in items:
                print(f"  - {it if isinstance(it, str) else json.dumps(it, ensure_ascii=False)}")
    extra = {k: v for k, v in result.items() if k not in (
        "status", "summary", "instruction", "applied_this_run", "skipped_for_questions", "failed",
        "questions_for_boss", "dry_run_filled", "error")}
    if extra:
        print("\n" + json.dumps(extra, indent=2, ensure_ascii=False, default=str)[:3000])


# Live screening questions: a running application asks you here instead of guessing.
def _ask_in_console(_text: str) -> None:
    q = live_questions.pending_question()
    if not q:
        return
    print("\n" + "=" * 70)
    print("A job application needs one answer your CV/profile does not cover:")
    print(f"  {q['question']}")
    if q.get("options"):
        print("  Options: " + " | ".join(q["options"][:12]))
    try:
        ans = input("Your answer (or 'skip'): ").strip()
    except EOFError:
        ans = "skip"
    print("=" * 70)
    live_questions.answer_job_question.invoke({"answer": ans or "skip"})


live_questions.set_notifier(_ask_in_console)


# ---------------------------------------------------------------------------
# Flows (each keeps you in the loop)
# ---------------------------------------------------------------------------

def _confirm_loop(result: dict) -> dict:
    """Company-site jobs: every account creation / final Submit waits for a yes."""
    while isinstance(result, dict) and result.get("status") == "needs_confirmation":
        _show(result)
        ok = _yes("\nGo ahead with exactly this step?")
        result = jas.confirm_job_step.invoke({"approve": ok})
    _show(result)
    return result


def _ask_questions(questions: list[str]) -> None:
    """Questions a board run skipped: answer now, they are filled in next run."""
    if not questions or not _yes("\nAnswer the skipped questions now (saved for next runs)?"):
        return
    for q in questions:
        try:
            ans = input(f"  {q}\n  > ").strip()
        except EOFError:
            return
        if ans and ans.lower() != "skip":
            jas.teach_job_agent.invoke({"note": f"{q} -> {ans}", "kind": "answer"})


def cmd_board(portal: str, keywords: str = "", location: str = "", max_apps: int = ca.DEFAULT_MAX,
              dry_run: bool = False) -> None:
    res = ca.start_continuous_apply.invoke({"portal": portal, "keywords": keywords, "location": location,
                                            "max_applications": max_apps, "dry_run": dry_run})
    if res.get("status") == "needs_portal":
        print("Pick a board: naukri, indeed or linkedin.")
        return
    if res.get("status") != "needs_confirmation":
        _show(res)
        return
    print("\nPlan:\n  " + _clean(res.get("plan") or res.get("summary")))
    if not _yes("\nStart this run? (one yes covers the whole run)"):
        ca.confirm_continuous_apply.invoke({"approve": False})
        print("Cancelled.")
        return
    res = ca.confirm_continuous_apply.invoke({"approve": True})
    while res.get("status") == "paused":
        _show(res)
        try:
            ans = input("\nFix it in the Edge window, then press Enter to continue (or 'q' to stop): ").strip()
        except EOFError:
            ans = "q"
        if ans.lower() in ("q", "quit", "stop"):
            jas.stop_job_applications.invoke({})
            print("Stopped.")
            return
        res = ca.resume_continuous_apply.invoke({})
    _show(res)
    _ask_questions(res.get("questions_for_boss") or [])
    if jas._run.queue:
        print(f"\n{len(jas._run.queue)} company-site job(s) are saved. Run 'saved' to apply to them.")


def cmd_url(url: str, company: str = "", title: str = "") -> None:
    _confirm_loop(jas.apply_to_job_url.invoke({"url": url, "company": company, "title": title}))


def cmd_auto(keywords: str = "", location: str = "", max_apps: int = 15, company_sites: bool = True) -> None:
    _confirm_loop(jas.auto_apply_jobs.invoke({"keywords": keywords, "location": location,
                                              "max_applications": max_apps,
                                              "include_company_sites": company_sites}))


def cmd_saved() -> None:
    _confirm_loop(jas.apply_saved_company_jobs.invoke({}))


def _first_line(text) -> str:
    return str(text or "?").splitlines()[0].strip() or "?"


def cmd_report() -> None:
    rep = jas.job_application_report.invoke({})
    print(f"\n{rep['summary']}")
    print("By status: " + ", ".join(f"{k} {v}" for k, v in rep["by_status"].items()))
    print(f"Company-site jobs saved: {rep['queued_jobs']} | portal accounts created: "
          f"{rep['portal_accounts_created']}")
    companies: dict[str, int] = {}
    for name, n in rep["applied_per_company"].items():
        companies[_first_line(name)] = companies.get(_first_line(name), 0) + n
    if companies:
        print("Applied at: " + ", ".join(f"{c} ({n})" if n > 1 else c
                                         for c, n in sorted(companies.items(), key=lambda kv: -kv[1])))
    if rep["needs_your_attention"]:
        print("\nNeeds your attention (latest):")
        for e in rep["needs_your_attention"]:
            print(f"  - {e['time']}  {e['title']} at {_first_line(e['company'])}\n      {e.get('notes', '')}"
                  f"\n      {e['url']}")
    print(f"\nFull log: {rep['log_file']}")
    st = ca.continuous_apply_status.invoke({})
    if st.get("status") != "idle":
        _show(st)


def cmd_teach(note: str, answer: bool) -> None:
    _show(jas.teach_job_agent.invoke({"note": note, "kind": "answer" if answer else "instruction"}))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autojob", description="Standalone job-apply backend")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="cmd")

    b = sub.add_parser("board", help="continuous quick-apply on naukri | indeed | linkedin")
    b.add_argument("portal")
    b.add_argument("-k", "--keywords", default="", help="comma separated; default: from your CV")
    b.add_argument("-l", "--location", default="", help="default: main cities + all India")
    b.add_argument("-n", "--max", type=int, default=ca.DEFAULT_MAX)
    b.add_argument("--dry-run", action="store_true", help="fill everything, never submit")

    u = sub.add_parser("url", help="apply to one job link")
    u.add_argument("url")
    u.add_argument("--company", default="")
    u.add_argument("--title", default="")

    a = sub.add_parser("auto", help="Naukri search + apply, then company-site jobs")
    a.add_argument("keywords", nargs="?", default="")
    a.add_argument("-l", "--location", default="")
    a.add_argument("-n", "--max", type=int, default=15)
    a.add_argument("--naukri-only", action="store_true", help="save company-site jobs instead of applying")

    sub.add_parser("saved", help="apply to saved company-site jobs")
    sub.add_parser("report", help="applications so far")

    t = sub.add_parser("teach", help="save an answer / preference for future applications")
    t.add_argument("note")
    t.add_argument("--answer", action="store_true", help="it is an answer to an application question")

    sub.add_parser("shell", help="interactive shell (default)")
    return p


def _dispatch(args: argparse.Namespace) -> None:
    if args.cmd == "board":
        cmd_board(args.portal, args.keywords, args.location, args.max, args.dry_run)
    elif args.cmd == "url":
        cmd_url(args.url, args.company, args.title)
    elif args.cmd == "auto":
        cmd_auto(args.keywords, args.location, args.max, not args.naukri_only)
    elif args.cmd == "saved":
        cmd_saved()
    elif args.cmd == "report":
        cmd_report()
    elif args.cmd == "teach":
        cmd_teach(args.note, args.answer)


def _shell(parser: argparse.ArgumentParser) -> None:
    print("Auto Job shell. Commands: board <naukri|indeed|linkedin> [-k ..] [-l ..] [-n N] [--dry-run], "
          "url <link>, auto [keywords], saved, report, teach \"note\" [--answer], help, exit")
    while True:
        try:
            line = input("\nautojob> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in ("exit", "quit", "q"):
            return
        if line in ("help", "?"):
            parser.print_help()
            continue
        try:
            args = parser.parse_args(shlex.split(line))
        except SystemExit:
            continue
        if args.cmd in (None, "shell"):
            continue
        _run_guarded(args)


def _run_guarded(args: argparse.Namespace) -> None:
    try:
        _dispatch(args)
    except KeyboardInterrupt:
        print("\nStopping after the current step...")
        jas.stop_job_applications.invoke({})


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(ROOT / "data" / "autojob.log", encoding="utf-8"), logging.StreamHandler()],
    )
    # Keep the console readable; full detail goes to data/autojob.log.
    logging.getLogger().handlers[1].setLevel(logging.DEBUG if args.verbose else logging.WARNING)
    for noisy in ("httpx", "httpcore", "urllib3", "google", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.cmd in (None, "shell"):
        _shell(parser)
    else:
        _run_guarded(args)


if __name__ == "__main__":
    main()
