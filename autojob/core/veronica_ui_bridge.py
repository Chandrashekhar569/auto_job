"""
Console stand-in for Veronica's dashboard bridge.

The job modules call emit({...}) with dashboard events. The standalone app has
no UI, so this prints the useful ones ("log" lines, agent status changes) to
the terminal instead. The API (emit / add_snapshot / start_ui_bridge) stays the
same, so the job modules need no changes.
"""

import logging
import threading

logger = logging.getLogger("autojob")

_print_lock = threading.Lock()
_last_agent_line = ""

_TONE_ICON = {"success": "[ok]", "warning": "[!]", "error": "[x]", "agent": "[..]"}


def emit(event: dict) -> None:
    """Print a dashboard event as one console line. Safe from any thread."""
    global _last_agent_line
    if not isinstance(event, dict):
        return
    kind = event.get("type")
    line = ""
    if kind == "log" and event.get("label"):
        line = f"{_TONE_ICON.get(event.get('tone'), '[-]')} {event['label']}"
    elif kind == "agent":
        bits = [str(event.get(k)) for k in ("status", "task", "action") if event.get(k)]
        line = "     agent: " + " | ".join(bits) if bits else ""
        if line == _last_agent_line:  # status pings repeat a lot
            return
        _last_agent_line = line
    if line:
        with _print_lock:
            print(line, flush=True)


def add_snapshot(provider) -> None:  # dashboard-only; nothing to snapshot here
    return None


def start_ui_bridge(port: int = 8766) -> None:  # no dashboard in the standalone app
    return None
