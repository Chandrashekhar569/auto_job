"""
Settings for the Auto Job SaaS web app. Everything can be overridden in .env.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

# Loads .env (LLM keys etc.) before anything reads os.environ.
from autojob.core import llm_router  # noqa: F401
from autojob.core.paths import DATA_DIR

SAAS_DIR = DATA_DIR / "saas"
USERS_DIR = SAAS_DIR / "users"          # per-user resumes + Naukri browser profiles
SAAS_DIR.mkdir(parents=True, exist_ok=True)
USERS_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("SAAS_DATABASE_URL") or f"sqlite:///{(SAAS_DIR / 'autojob.db').as_posix()}"


def _persistent_secret(env_name: str, filename: str, factory) -> str:
    """Use the env var if set, else a key generated once and kept in data/saas/."""
    val = os.getenv(env_name)
    if val:
        return val
    path = SAAS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    val = factory()
    path.write_text(val, encoding="utf-8")
    return val


def _fernet_key() -> str:
    from cryptography.fernet import Fernet
    return Fernet.generate_key().decode()


# Signs login cookies.
SECRET_KEY = _persistent_secret("SAAS_SECRET_KEY", ".secret_key", lambda: secrets.token_urlsafe(48))
# Encrypts stored Naukri passwords at rest.
FERNET_KEY = _persistent_secret("SAAS_FERNET_KEY", ".fernet_key", _fernet_key)

SESSION_DAYS = int(os.getenv("SAAS_SESSION_DAYS", "14"))
COOKIE_SECURE = os.getenv("SAAS_COOKIE_SECURE", "0") == "1"   # set 1 behind HTTPS

# Browser automation
HEADLESS = os.getenv("SAAS_HEADLESS", "1") == "1"
BROWSER_CHANNEL = os.getenv("SAAS_BROWSER_CHANNEL", "msedge")  # "" = Playwright's bundled Chromium
MAX_PARALLEL_RUNS = int(os.getenv("SAAS_MAX_PARALLEL_RUNS", "2"))

# Scheduler
SCHEDULER_ENABLED = os.getenv("SAAS_SCHEDULER", "1") == "1"
SCHEDULER_TICK_S = int(os.getenv("SAAS_SCHEDULER_TICK_S", "300"))
DEFAULT_DAILY_LIMIT = int(os.getenv("SAAS_DEFAULT_DAILY_LIMIT", "25"))
HARD_DAILY_LIMIT = 50   # Naukri itself caps applications per day


def user_dir(user_id: int) -> Path:
    d = USERS_DIR / str(user_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def resume_dir(user_id: int) -> Path:
    d = user_dir(user_id) / "resume"
    d.mkdir(parents=True, exist_ok=True)
    return d


def browser_profile_dir(user_id: int) -> Path:
    # Unique name: leftover-browser cleanup matches processes by this folder name.
    d = user_dir(user_id) / f"autojob_browser_u{user_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d
