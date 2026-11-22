"""
Run manager + daily scheduler.

One run at a time per user, at most SAAS_MAX_PARALLEL_RUNS browsers across all users.
The scheduler wakes every few minutes and, for each user with auto-apply on:
  1. once a day: makes the small Naukri profile edit (if not done today),
  2. then starts the day's apply run (from the user's chosen hour), until the daily limit.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime
from typing import Callable, Optional

from sqlalchemy import select

from saas.config import MAX_PARALLEL_RUNS, SCHEDULER_TICK_S
from saas.db import NaukriAccount, Resume, Run, Settings, applied_today, session_scope

logger = logging.getLogger("saas.runs")

_slots = threading.BoundedSemaphore(MAX_PARALLEL_RUNS)
_lock = threading.Lock()
_active: dict[int, "RunContext"] = {}


class RunContext:
    def __init__(self, user_id: int, run_id: int, kind: str):
        self.user_id, self.run_id, self.kind = user_id, run_id, kind
        self._stop = threading.Event()
        self.bot = None
        self.applied = 0
        self.lines: list[str] = []

    def should_stop(self) -> bool:
        return self._stop.is_set()

    def stop(self) -> None:
        self._stop.set()
        if self.bot is not None:
            try:
                self.bot.request_stop()
            except Exception:
                pass

    def log(self, line: str) -> None:
        logger.info("[user %s run %s] %s", self.user_id, self.run_id, line)
        self.lines.append(f"{datetime.now():%H:%M} {line}")
        self.lines = self.lines[-40:]
        with session_scope() as s:
            run = s.get(Run, self.run_id)
            if run:
                run.summary = "\n".join(self.lines)
                run.applied = self.applied


def _jobs() -> dict[str, Callable]:
    from saas import naukri_worker as w
    return {
        "login": w.job_login,
        "manual_login": w.job_manual_login,
        "profile_refresh": w.job_profile_refresh,
        "apply": w.job_apply,
        "dry_run": lambda uid, ctx: w.job_apply(uid, ctx, dry_run=True),
    }


def active_run(user_id: int) -> Optional[RunContext]:
    with _lock:
        return _active.get(user_id)


def start(user_id: int, kind: str, trigger: str = "manual") -> tuple[bool, str]:
    if kind not in _jobs():
        return False, f"Unknown job {kind}"
    with _lock:
        if user_id in _active:
            return False, f"A {_active[user_id].kind.replace('_', ' ')} run is already in progress."
        with session_scope() as s:
            run = Run(user_id=user_id, kind=kind, trigger=trigger, status="queued")
            s.add(run)
            s.flush()
            run_id = run.id
        ctx = RunContext(user_id, run_id, kind)
        _active[user_id] = ctx
    threading.Thread(target=_execute, args=(ctx,), daemon=True, name=f"run-{user_id}-{kind}").start()
    return True, "Started"


def stop(user_id: int) -> bool:
    ctx = active_run(user_id)
    if not ctx:
        return False
    ctx.stop()
    return True


def _finish(ctx: RunContext, status: str, summary: str) -> None:
    with session_scope() as s:
        run = s.get(Run, ctx.run_id)
        if run:
            run.status = status
            run.applied = ctx.applied
            run.summary = "\n".join(ctx.lines + [summary]).strip()
            run.finished_at = datetime.utcnow()


def _execute(ctx: RunContext) -> None:
    from saas.naukri_worker import NeedsAttention

    try:
        # Wait for a free browser slot (other users' runs).
        while not _slots.acquire(timeout=5):
            if ctx.should_stop():
                _finish(ctx, "stopped", "Stopped before it started.")
                return
        try:
            with session_scope() as s:
                run = s.get(Run, ctx.run_id)
                run.status, run.started_at = "running", datetime.utcnow()
            summary = _jobs()[ctx.kind](ctx.user_id, ctx)
            _finish(ctx, "stopped" if ctx.should_stop() else "done", summary or "Done")
        finally:
            _slots.release()
    except NeedsAttention as exc:
        _finish(ctx, "error", f"Needs your attention: {exc}")
    except Exception as exc:
        logger.exception("Run %s failed", ctx.run_id)
        _finish(ctx, "error", f"Failed: {exc}")
    finally:
        with _lock:
            _active.pop(ctx.user_id, None)


def recover_stale_runs() -> None:
    """Runs left 'running' by a server restart can never finish."""
    with session_scope() as s:
        for run in s.scalars(select(Run).where(Run.status.in_(("queued", "running")))):
            run.status, run.finished_at = "error", datetime.utcnow()
            run.summary = (run.summary + "\nInterrupted by a server restart.").strip()


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def _ran_today(s, user_id: int, kinds: tuple[str, ...]) -> bool:
    start_of_day = datetime.combine(date.today(), datetime.min.time())
    return s.scalar(select(Run.id).where(Run.user_id == user_id, Run.kind.in_(kinds), Run.trigger == "scheduler",
                                         Run.started_at >= start_of_day, Run.status != "error").limit(1)) is not None


def _failures_today(s, user_id: int) -> int:
    start_of_day = datetime.combine(date.today(), datetime.min.time())
    return len(s.scalars(select(Run.id).where(Run.user_id == user_id, Run.trigger == "scheduler",
                                              Run.started_at >= start_of_day, Run.status == "error")).all())


def tick() -> None:
    now = datetime.now()
    due: list[tuple[int, str]] = []
    with session_scope() as s:
        for st in s.scalars(select(Settings).where(Settings.auto_apply.is_(True))):
            uid = st.user_id
            acc = s.scalar(select(NaukriAccount).where(NaukriAccount.user_id == uid))
            res = s.scalar(select(Resume).where(Resume.user_id == uid))
            if not acc or acc.status != "connected" or not res or res.status != "ready":
                continue
            if now.hour < (st.run_hour or 0) or _failures_today(s, uid) >= 3:
                continue
            if applied_today(s, uid) < min(st.daily_limit, 50) and not _ran_today(s, uid, ("apply",)):
                due.append((uid, "apply"))   # the apply run does the profile refresh first
            elif st.daily_profile_refresh and acc.last_profile_refresh != date.today():
                due.append((uid, "profile_refresh"))
    for uid, kind in due:
        if active_run(uid) is None:
            ok, msg = start(uid, kind, trigger="scheduler")
            logger.info("Scheduler: user %s %s -> %s", uid, kind, msg)


def _loop() -> None:
    time.sleep(20)   # let the server come up
    while True:
        try:
            tick()
        except Exception:
            logger.exception("Scheduler tick failed")
        time.sleep(SCHEDULER_TICK_S)


def start_scheduler() -> None:
    threading.Thread(target=_loop, daemon=True, name="scheduler").start()
    logger.info("Scheduler started (every %ss)", SCHEDULER_TICK_S)
