"""
downloads.py — one known place for everything the browser downloads.

Playwright keeps a download in a hidden temp folder under a random name and
deletes it when Edge closes, so after "download the video" neither the agent
nor the posting tools could find the file. BrowserSession now hands every
download to save_download(): the file is kept in data/downloads/ under its real
name and recorded in data/downloads/index.json (newest last), so

  * run_browser_task can tell Veronica the exact path it saved,
  * prepare_video_post(video="latest_download") finds the newest video, and the
    newest image downloaded around it becomes the thumbnail.

Files Boss downloads by hand into Windows' Downloads folder are found too
(latest_download falls back to it).
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from autojob.core.paths import DOWNLOADS_DIR

logger = logging.getLogger(__name__)

INDEX = DOWNLOADS_DIR / "index.json"
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_MAX_ENTRIES = 200
_lock = threading.Lock()


def kind_of(name: str) -> str:
    ext = Path(name).suffix.lower()
    return "video" if ext in VIDEO_EXTS else "image" if ext in IMAGE_EXTS else "other"


def _load() -> list[dict]:
    try:
        data = json.loads(INDEX.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _unique(name: str) -> Path:
    clean = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name).strip(" .") or "download"
    target = DOWNLOADS_DIR / clean
    stem, suffix, n = target.stem, target.suffix, 2
    while target.exists():
        target = DOWNLOADS_DIR / f"{stem} ({n}){suffix}"
        n += 1
    return target


def save_download(download, page_url: str = "") -> Optional[dict]:
    """Save a Playwright Download under its real name; returns its index entry.
    Must run on the browser's owner thread (it waits for the file to finish)."""
    try:
        DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
        name = download.suggested_filename or "download"
        target = _unique(name)
        download.save_as(str(target))
        if not target.exists() or target.stat().st_size == 0:
            failure = download.failure()
            raise RuntimeError(failure or "empty file")
        entry = {"file": str(target), "name": target.name, "kind": kind_of(target.name),
                 "size_kb": target.stat().st_size // 1024, "url": (download.url or "")[:300],
                 "page_url": page_url[:300], "at": datetime.now().isoformat(timespec="seconds")}
        with _lock:
            items = _load()
            items.append(entry)
            INDEX.write_text(json.dumps(items[-_MAX_ENTRIES:], indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Download saved: %s (%d KB)", target, entry["size_kb"])
        try:
            from autojob.core.veronica_ui_bridge import emit

            emit({"type": "log", "label": f"Downloaded {target.name} → {target}", "tone": "success"})
        except Exception:
            pass
        return entry
    except Exception as exc:
        logger.warning("Could not save a download (%s)", exc)
        return None


def unique_path(name: str) -> Path:
    """A free path in data/downloads for a file Veronica is about to write herself."""
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    return _unique(name)


def register_file(path: Path, url: str = "") -> dict:
    """Record a file already saved in data/downloads (e.g. by the Shorts browser,
    which saves its own downloads), so latest_download() finds it."""
    entry = {"file": str(path), "name": path.name, "kind": kind_of(path.name),
             "size_kb": path.stat().st_size // 1024, "url": url[:300], "page_url": url[:300],
             "at": datetime.now().isoformat(timespec="seconds")}
    with _lock:
        items = _load()
        items.append(entry)
        INDEX.write_text(json.dumps(items[-_MAX_ENTRIES:], indent=2, ensure_ascii=False), encoding="utf-8")
    return entry


def recent(kind: str = "", limit: int = 10) -> list[dict]:
    """Newest first; only entries whose file still exists."""
    out = []
    for e in reversed(_load()):
        if kind and e.get("kind") != kind:
            continue
        if Path(e.get("file", "")).is_file():
            out.append(e)
        if len(out) >= limit:
            break
    return out


def _windows_downloads(kind: str, since: datetime) -> Optional[Path]:
    """Newest file of that kind Boss downloaded by hand (Windows Downloads folder)."""
    folder = Path.home() / "Downloads"
    exts = VIDEO_EXTS if kind == "video" else IMAGE_EXTS
    try:
        files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts
                 and datetime.fromtimestamp(p.stat().st_mtime) >= since]
    except OSError:
        return None
    return max(files, key=lambda p: p.stat().st_mtime, default=None)


def latest_download(kind: str = "video", max_age_h: float = 48) -> Optional[Path]:
    """The newest downloaded file of a kind (Veronica's folder first, then Windows Downloads)."""
    since = datetime.now() - timedelta(hours=max_age_h)
    for e in recent(kind, limit=1):
        try:
            if datetime.fromisoformat(e["at"]) >= since:
                return Path(e["file"])
        except (KeyError, ValueError):
            continue
    return _windows_downloads(kind, since)


def thumbnail_for(video: Path, window_h: float = 3) -> Optional[Path]:
    """The image downloaded closest in time to the video (within window_h), if any."""
    try:
        when = datetime.fromtimestamp(video.stat().st_mtime)
    except OSError:
        return None
    best, gap = None, timedelta(hours=window_h)
    candidates = [Path(e["file"]) for e in recent("image", limit=30)]
    hand = _windows_downloads("image", when - timedelta(hours=window_h))
    if hand:
        candidates.append(hand)
    for img in candidates:
        try:
            d = abs(datetime.fromtimestamp(img.stat().st_mtime) - when)
        except OSError:
            continue
        if d <= gap:
            best, gap = img, d
    return best


def summary(entries: list[dict]) -> list[dict[str, Any]]:
    return [{"file": e["file"], "kind": e["kind"], "size_kb": e["size_kb"]} for e in entries]
