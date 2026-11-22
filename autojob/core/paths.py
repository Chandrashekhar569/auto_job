"""
paths.py — every filesystem location Auto Job uses, in one place.

Modules import from here instead of building paths off their own __file__,
so moving a module never changes where its data lives.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = PROJECT_ROOT / "autojob"

ENV_FILE = PROJECT_ROOT / ".env"

# OAuth client + Gmail token (gitignored)
SECRETS_DIR = PROJECT_ROOT / "secrets"
GOOGLE_CREDENTIALS_FILE = SECRETS_DIR / "credentials.json"
GMAIL_TOKEN_FILE = SECRETS_DIR / "token.json"

# Runtime data (mostly gitignored)
DATA_DIR = PROJECT_ROOT / "data"
PROFILE_DIR = DATA_DIR / "profile"
ABOUT_ME_FILE = PROFILE_DIR / "ABOUT_ME.md"
CV_DIR = PROFILE_DIR / "cv"
CONTACTS_FILE = DATA_DIR / "contacts.json"
# Every file the browser agent downloads, under its real name (see downloads.py)
DOWNLOADS_DIR = DATA_DIR / "downloads"

BROWSER_EXTENSIONS_DIR = PACKAGE_DIR / "integrations" / "browser_extensions"
