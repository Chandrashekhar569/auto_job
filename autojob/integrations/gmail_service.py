import os
import json
import logging
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from autojob.core.paths import ENV_FILE, GMAIL_TOKEN_FILE, GOOGLE_CREDENTIALS_FILE, CONTACTS_FILE
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

_TOKEN_PATH = GMAIL_TOKEN_FILE
_CREDENTIALS_PATH = GOOGLE_CREDENTIALS_FILE
_ENV_LOADED = False
_creds_lock = threading.Lock()  # Lock for credentials refresh


def _ensure_env():
    """Make sure .env is loaded."""
    global _ENV_LOADED
    if not _ENV_LOADED:
        env_path = ENV_FILE
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key, val = key.strip(), val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val
        _ENV_LOADED = True

# Gmail API scopes - defined once
GMAIL_SCOPES = [
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/gmail.send'
]


# ---------------------------------------------------------------------------
# LangChain ChatOpenRouter — Email Agent (with Gemini fallback)
# ---------------------------------------------------------------------------

_email_llm = None


def _get_email_llm():
    """Return a cached ChatOpenRouter instance for email tasks.

    Returns None when NVIDIA_API_KEY is not set so callers can
    short-circuit to the Gemini fallback without raising pydantic errors.
    """
    global _email_llm
    if _email_llm is not None:
        return _email_llm

    _ensure_env()
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        logger.warning("NVIDIA_API_KEY not set — Nvidia email agent disabled")
        return None

    # pyrefly: ignore [missing-import]
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    _email_llm = ChatNVIDIA(
        api_key=api_key,
        model=os.getenv("EMAIL_AGENT_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"),
        temperature=0.2,
    )
    return _email_llm


def _is_auth_failure(exc: Exception) -> bool:
    """
    True when a token refresh failed because the grant itself is invalid
    (revoked, expired, or wrong client) rather than because of a transient
    network/server problem. Only the former justifies deleting token.json.
    """
    text = (type(exc).__name__ + " " + str(exc)).lower()
    return any(kw in text for kw in [
        "invalid_grant", "invalid_client", "unauthorized_client",
        "invalid_token", "token has been expired or revoked",
    ])


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception is a rate-limit / too-many-requests error."""
    exc_str = str(type(exc).__name__).lower() + str(exc).lower()
    return any(kw in exc_str for kw in [
        "toomanyrequests", "rate_limit", "ratelimit", "429", "quota",
    ])


_gemini_client = None


def _gemini_fallback(prompt: str) -> str:
    """Use Gemini API as fallback when OpenRouter is rate-limited."""
    global _gemini_client
    _ensure_env()

    api_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        return "[Gemini fallback error: GEMINI_API_KEY not set]"

    try:
        if _gemini_client is None:
            from google import genai
            _gemini_client = genai.Client(api_key=api_key)

        from autojob.core.llm_router import _no_afc_config

        response = _gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=_no_afc_config(temperature=0.7),
        )
        return response.text.strip()
    except Exception as exc:
        logger.exception("Gemini fallback also failed")
        return f"[Gemini fallback error: {exc}]"


def _fetch_todays_emails() -> list[dict] | None:
    """Fetch today's emails from Gmail API using ThreadPoolExecutor for speed.

    Handles pagination and timezone correctly by using UTC dates.
    """
    from concurrent.futures import ThreadPoolExecutor
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials

    service = get_gmail_service()
    if not service:
        # get_gmail_service already logged the actionable re-auth instruction.
        return None

    _ensure_env()
    token_path = _TOKEN_PATH
    if not token_path.exists():
        logger.error("token.json disappeared mid-run — re-run authenticate_gmail.py.")
        return None
    creds = Credentials.from_authorized_user_file(str(token_path), GMAIL_SCOPES)

    def fetch_single_msg(msg_id):
        try:
            # Build service per thread since the discovery service is not thread-safe
            thread_service = build('gmail', 'v1', credentials=creds)
            return thread_service.users().messages().get(
                userId='me',
                id=msg_id,
                format='metadata',
                metadataHeaders=['From', 'Subject', 'Date']
            ).execute()
        except Exception as e:
            logger.warning("Error fetching email %s: %s", msg_id, e)
            return None

    try:
        # Use UTC date to match Gmail's internal timestamps
        today_utc = datetime.now(timezone.utc).date()
        yesterday_utc = today_utc - timedelta(days=1)
        # Query messages since yesterday UTC to handle timezone differences
        q = f"after:{yesterday_utc.strftime('%Y/%m/%d')}"

        # Fetch ALL messages with pagination
        all_messages = []
        page_token = None
        max_total = 500  # Safety limit

        while len(all_messages) < max_total:
            results = service.users().messages().list(
                userId='me', q=q, maxResults=min(100, max_total - len(all_messages)), pageToken=page_token
            ).execute()
            messages = results.get('messages', [])
            if not messages:
                break
            all_messages.extend(messages)
            page_token = results.get('nextPageToken')
            if not page_token:
                break

        if not all_messages:
            return []

        # Fetch message details in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(fetch_single_msg, m['id']) for m in all_messages]
            fetched_msgs = [f.result() for f in futures]

        today_mails = []
        for msg in fetched_msgs:
            if not msg:
                continue
            internal_date_ms = int(msg.get('internalDate', 0))
            # Gmail internal dates are in UTC milliseconds since epoch
            msg_dt = datetime.fromtimestamp(internal_date_ms / 1000.0, tz=timezone.utc)
            msg_date = msg_dt.date()

            if msg_date == today_utc:
                label_ids = msg.get('labelIds', [])
                is_unread = 'UNREAD' in label_ids
                headers = {h['name']: h['value'] for h in msg.get('payload', {}).get('headers', [])}
                today_mails.append({
                    'id': msg['id'],
                    'from': headers.get('From', 'Unknown'),
                    'subject': headers.get('Subject', 'No Subject'),
                    'date': msg_dt.strftime('%Y-%m-%d %H:%M:%S'),
                    'unread': is_unread,
                    'labels': label_ids,
                    'snippet': msg.get('snippet', '')
                })
        return today_mails
    except Exception as exc:
        logger.exception("Failed to fetch today's emails: %s", exc)
        return []


def get_email_response(query: str, emails: list[dict] | None) -> str:
    """
    Analyze emails and answer user queries based on fetched data.

    Flow: OpenRouter → retry → Gemini fallback.
    """
    if emails is None:
        return '{"summary": "Gmail authentication has expired. Please re-authenticate Gmail.", "error": "auth_failed"}'

    if not emails:
        emails_text = "No emails received today."
    else:
        emails_list_str = []
        for idx, mail in enumerate(emails, 1):
            status = "Unread" if mail['unread'] else "Read"
            labels_str = ", ".join(mail.get('labels', []))
            emails_list_str.append(
                f"{idx}. ID: {mail['id']}\n"
                f"   From: {mail['from']}\n"
                f"   Subject: {mail['subject']}\n"
                f"   Date: {mail['date']}\n"
                f"   Status: {status}\n"
                f"   Labels: {labels_str}\n"
                f"   Snippet: {mail['snippet']}\n"
            )
        emails_text = "\n".join(emails_list_str)

    prompt = (
        f"You are a helpful email assistant. The user is asking a question about their emails received today.\n\n"
        f"Today's date is: {date.today().isoformat()}\n\n"
        f"Here are the emails received today:\n"
        f"{emails_text}\n\n"
        f"User question: {query}\n\n"
        f"Please answer the user's question accurately based ONLY on the emails listed above. Be concise, clear, and direct."
    )

    from autojob.core.llm_router import fast_chat
    try:
        response = fast_chat(prompt=prompt, task_type="email", system_prompt="You are an email assistant.")
        if response and not response.startswith("[Error"):
            return response
    except Exception:
        pass

    # ── Attempt 1: OpenRouter ──
    try:
        llm = _get_email_llm()
        if llm is None:
            logger.info("OpenRouter unavailable (no API key), using Gemini directly")
            return _gemini_fallback(prompt)
        response = llm.invoke(prompt)
        return response.content
    except Exception as exc:
        if not _is_rate_limit_error(exc):
            logger.warning("OpenRouter email agent failed (non-rate-limit), using Gemini: %s", exc)
            return _gemini_fallback(prompt)
        logger.warning("OpenRouter rate-limited for email, retrying in 3s...")

    # ── Attempt 2: Retry ──
    time.sleep(3)
    try:
        llm = _get_email_llm()
        if llm is None:
            logger.info("OpenRouter unavailable (no API key), using Gemini fallback")
            return _gemini_fallback(prompt)
        response = llm.invoke(prompt)
        return response.content
    except Exception as exc:
        logger.warning("OpenRouter email retry failed, using Gemini fallback: %s", exc)

    # ── Attempt 3: Gemini fallback ──
    logger.info("Using Gemini fallback for email agent")
    return _gemini_fallback(prompt)


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

_gmail_lock = threading.Lock()

@tool
def ask_email_agent(query: str) -> dict[str, Any]:
    """Talk to the email agent. Only handles current-date emails."""
    with _gmail_lock:
        today_note = f" (Note: Only processing today's emails — {date.today().isoformat()})"
        emails = _fetch_todays_emails()
        if emails is None:
            return {"summary": "Gmail authentication has expired. Please re-authenticate Gmail.", "error": "auth_failed"}
        response = get_email_response(query, emails)
        return {"summary": response + today_note}


@tool
def gmail_send(to: str, subject: str, body: str) -> dict[str, Any]:
    """Send an email using Gmail API."""
    import re
    # Basic email validation
    if not re.match(r'^[^@]+@[^@]+\.[^@]+$', to):
        return {"status": "error", "error": f"Invalid email format: {to}", "summary": f"Invalid email format: {to}"}

    with _gmail_lock:
        service = get_gmail_service()
        if not service:
            return {"status": "error", "error": "Gmail not authenticated", "summary": "Failed to send email. Gmail is not authenticated."}

        import base64
        from email.mime.text import MIMEText

        try:
            message = MIMEText(body)
            message['to'] = to
            message['subject'] = subject
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')

            send_result = service.users().messages().send(
                userId='me',
                body={'raw': raw_message}
            ).execute()

            logger.info("Email sent successfully: %s", send_result)
            return {"status": "success", "summary": f"Email successfully sent to {to} with subject '{subject}'."}
        except Exception as exc:
            logger.exception("Failed to send email")
            return {"status": "error", "error": str(exc), "summary": f"Error sending email: {exc}"}


# ---------------------------------------------------------------------------
# Contacts
# ---------------------------------------------------------------------------

_CONTACTS_PATH = CONTACTS_FILE
_contacts_lock = threading.Lock()


def _load_contacts() -> list[dict]:
    """Load contacts from contacts.json (thread-safe)."""
    with _contacts_lock:
        if not _CONTACTS_PATH.exists():
            return []
        try:
            data = json.loads(_CONTACTS_PATH.read_text(encoding="utf-8"))
            return data.get("contacts", [])
        except Exception:
            return []


def _save_contacts(contacts: list[dict]) -> None:
    """Save contacts to contacts.json (thread-safe)."""
    with _contacts_lock:
        _CONTACTS_PATH.write_text(
            json.dumps({"contacts": contacts}, indent=4, ensure_ascii=False),
            encoding="utf-8",
        )


@tool
def lookup_contact(name: str) -> dict[str, Any]:
    """Look up a contact by name or nickname to find their email address. ALWAYS use this FIRST when user says 'send email to [name]'."""
    contacts = _load_contacts()
    name_lower = name.strip().lower()
    for c in contacts:
        nicknames = c.get("nickname", [])
        if isinstance(nicknames, str):
            nicknames = [nicknames]
        nicknames_lower = [n.lower() for n in nicknames]

        if (c.get("name", "").lower() == name_lower or
                name_lower in nicknames_lower):
            return {"found": True, "contact": c, "summary": f"Found: {c['name']} — {c.get('email', 'no email')}"}

    # Search today's emails as fallback
    try:
        emails = _fetch_todays_emails()
        for mail in emails:
            sender = mail.get('from', '')
            if name_lower in sender.lower():
                # Extract name and email
                display_name = sender
                email_addr = ""
                if "<" in sender and ">" in sender:
                    display_name = sender.split("<")[0].strip().strip('"')
                    email_addr = sender.split("<")[1].split(">")[0].strip()
                else:
                    email_addr = sender.strip()
                
                return {
                    "found": True,
                    "contact": {"name": display_name, "email": email_addr},
                    "summary": f"Found in today's emails: {display_name} — {email_addr}"
                }
    except Exception as exc:
        logger.warning("Failed to search today's emails for contact lookup: %s", exc)

    return {"found": False, "summary": f"No contact found with name '{name}'."}


@tool
def gmail_mark_as_read(message_id: str) -> dict[str, Any]:
    """Mark a specific email as read in Gmail using its message ID."""
    with _gmail_lock:
        service = get_gmail_service()
        if not service:
            return {"status": "error", "error": "Gmail not authenticated", "summary": "Gmail service is not authenticated."}
        try:
            service.users().messages().batchModify(
                userId='me',
                body={
                    'ids': [message_id],
                    'removeLabelIds': ['UNREAD']
                }
            ).execute()
            return {"status": "success", "summary": f"Email {message_id} marked as read."}
        except Exception as exc:
            logger.exception("Failed to mark email as read")
            return {"status": "error", "error": str(exc), "summary": f"Failed to mark email as read: {exc}"}


@tool
def gmail_mark_all_as_read() -> dict[str, Any]:
    """Mark all today's unread emails as read in Gmail."""
    with _gmail_lock:
        emails = _fetch_todays_emails()
        if emails is None:
            return {"status": "error", "error": "auth_failed", "summary": "Gmail authentication has expired. Please re-authenticate Gmail."}
        unread_ids = [m['id'] for m in emails if m['unread']]
        if not unread_ids:
            return {"status": "success", "count": 0, "summary": "No unread emails today to mark as read."}

        service = get_gmail_service()
        if not service:
            return {"status": "error", "error": "Gmail not authenticated", "summary": "Gmail service is not authenticated."}
        try:
            service.users().messages().batchModify(
                userId='me',
                body={
                    'ids': unread_ids,
                    'removeLabelIds': ['UNREAD']
                }
            ).execute()
            return {
                "status": "success",
                "count": len(unread_ids),
                "summary": f"Successfully marked all {len(unread_ids)} today's unread emails as read."
            }
        except Exception as exc:
            logger.exception("Failed to mark all emails as read")
            return {"status": "error", "error": str(exc), "summary": f"Failed to mark today's emails as read: {exc}"}


@tool
def add_contact(name: str, email: str, nickname: str = "") -> dict[str, Any]:
    """Save a new contact with name, email, and optional nickname."""
    import re
    # Basic email validation
    if not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
        return {"error": f"Invalid email format: {email}", "summary": f"Invalid email format: {email}"}

    with _gmail_lock:
        contacts = _load_contacts()
        contacts.append({"name": name, "email": email, "nickname": nickname})
        _save_contacts(contacts)
    return {"status": "success", "summary": f"Contact '{name}' ({email}) saved."}


@tool
def list_contacts() -> dict[str, Any]:
    """List all saved contacts."""
    with _gmail_lock:
        contacts = _load_contacts()
    if not contacts:
        return {"status": "success", "count": 0, "contacts": [], "summary": "No contacts saved."}
    return {
        "status": "success",
        "count": len(contacts),
        "contacts": contacts,
        "summary": f"{len(contacts)} contact(s) saved.",
    }


# ---------------------------------------------------------------------------
# Gmail LangChain Tools
# ---------------------------------------------------------------------------

GMAIL_LANGCHAIN_TOOLS = [
    ask_email_agent,
    gmail_send,
    lookup_contact,
    add_contact,
    list_contacts,
    gmail_mark_as_read,
    gmail_mark_all_as_read
]


def get_gmail_tools() -> list[dict]:
    """Return the tools list ready to merge into LiveConnectConfig."""
    from autojob.core.langchain_bridge import get_gemini_tools_from_langchain
    return get_gemini_tools_from_langchain(GMAIL_LANGCHAIN_TOOLS)


def execute_gmail_function(name: str, args: dict) -> dict[str, Any]:
    """Dispatch a function call to the right Gmail tool handler."""
    for t in GMAIL_LANGCHAIN_TOOLS:
        if t.name == name:
            try:
                # Invoke the LangChain tool
                return t.invoke(args)
            except Exception as exc:
                return {"error": f"Invalid arguments or execution failed for {name}: {exc}"}
    return {"error": f"Unknown Gmail function: {name}"}


def is_gmail_configured() -> bool:
    """Check if Gmail credentials file exists."""
    return _CREDENTIALS_PATH.exists()


def is_gmail_authenticated() -> bool:
    """Check if the user has a valid Gmail token."""
    token_path = _TOKEN_PATH
    return token_path.exists()


def get_gmail_service() -> Any:
    """Get the Gmail service using stored credentials and token.

    Thread-safe credential refresh with locking to prevent race conditions.
    """
    _ensure_env()
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    token_path = _TOKEN_PATH

    # The token must be loaded *inside* the lock. Loading it first and then
    # locking means two threads both read the same stale token, both refresh,
    # and the second refresh fails against an already-rotated refresh_token —
    # which used to delete the good token the first thread had just written.
    with _creds_lock:
        creds = None
        if token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(token_path), GMAIL_SCOPES)
            except Exception as exc:
                logger.warning("Failed to load credentials from token.json: %s", exc)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                    token_path.write_text(creds.to_json(), encoding="utf-8")
                    logger.info("Gmail token refreshed successfully")
                except Exception as exc:
                    # Only discard the token when the grant itself is dead.
                    # A transient network failure must not force the user
                    # through the whole OAuth flow again.
                    if _is_auth_failure(exc):
                        logger.warning(
                            "Gmail refresh token rejected: %s. Removing invalid token.json.", exc
                        )
                        token_path.unlink(missing_ok=True)
                    else:
                        logger.warning(
                            "Gmail token refresh failed transiently (%s); keeping token.json "
                            "so the next attempt can retry.", exc
                        )
                    creds = None
            else:
                creds = None

    if not creds:
        logger.error(
            "Gmail is not authenticated. Run 'python scripts/authenticate_gmail.py' "
            "(or auth_gmail.bat) to sign in again — email tools stay disabled "
            "until you do."
        )
        return None

    return build('gmail', 'v1', credentials=creds)
