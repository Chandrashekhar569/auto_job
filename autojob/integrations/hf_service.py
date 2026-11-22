"""
HuggingFace + OpenRouter service for Chanda AI.

Provides free AI capabilities using LangChain's ChatOpenRouter:
- Text generation via OpenRouter free models (role-based model selection)
- Grammar checking for English lessons
- Question generation for interview prep
- Answer evaluation for interview & English coaching
- Vocabulary word generation
- Embeddings via NVIDIA nemotron-3-embed-1b

Best free model is selected per task role:
  - Code tasks    → poolside/laguna-m.1:free       (optimised for SW engineering)
  - Reasoning     → google/gemma-4-31b-it:free     (strong reasoning & multilingual)
  - Language/Eval → google/gemma-4-31b-it:free     (great at language understanding)
  - General       → google/gemma-4-31b-it:free     (all-rounder)

Fallback: If OpenRouter rate-limits (TooManyRequests), automatically retries
once after a short delay, then falls back to Gemini text API.
"""

import json
import logging
import os
import time
from pathlib import Path
from autojob.core.paths import PROJECT_ROOT, ENV_FILE
from typing import Any

import sys

logger = logging.getLogger(__name__)

# Embeddings are hosted on NVIDIA, so sentence_transformers is never used. Mark it
# missing so libraries that probe for it with `except ImportError` (e.g.
# langchain_text_splitters) skip it instead of loading torch, whose DLLs can be
# blocked on Windows (Smart App Control / WDAC -> "OSError: [WinError 4551]").
sys.modules.setdefault("sentence_transformers", None)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PROJECT_DIR = PROJECT_ROOT
_ENV_LOADED = False


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


def _get_openrouter_key() -> str:
    _ensure_env()
    return os.environ.get("OPENROUTER_API_KEY", "")


def _get_gemini_key() -> str:
    _ensure_env()
    return os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")


# ---------------------------------------------------------------------------
# Model Registry — best models per role (free tier with paid fallback)
# ---------------------------------------------------------------------------

# Role → OpenRouter model ID (free tier)
# Note: Model IDs should match what's available on OpenRouter
MODEL_REGISTRY: dict[str, str] = {
    "code":       "poolside/laguna-m.1:free",        # flagship coding agent
    "reasoning":  "google/gemma-2-27b-it:free",      # strong reasoning + multilingual
    "language":   "google/gemma-2-27b-it:free",      # English teaching, grammar
    "evaluation": "google/gemma-2-27b-it:free",      # answer evaluation, scoring
    "general":    "google/gemma-2-27b-it:free",      # default fallback
    # TTS is not available via OpenRouter text models - use NVIDIA NIM directly
}

# Paid fallback models (used when free tier is rate-limited)
MODEL_REGISTRY_PAID: dict[str, str] = {
    "code":       "poolside/laguna-m.1",
    "reasoning":  "google/gemma-2-27b-it",
    "language":   "google/gemma-2-27b-it",
    "evaluation": "google/gemma-2-27b-it",
    "general":    "google/gemma-2-27b-it",
}

# Gemini model for fallback (non-live, text generation)
GEMINI_FALLBACK_MODEL = "gemini-3.6-flash"

# Cached LLM instances per role (keyed by role + temperature)
_llm_cache: dict[str, Any] = {}


def _get_llm(role: str = "general", temperature: float = 0.7, use_paid: bool = False):
    """
    Return a cached ChatOpenRouter instance for the given role.

    Uses MODEL_REGISTRY to pick the best model for each task type.
    Falls back to paid model if use_paid is True (for rate-limit retry).
    """
    cache_key = f"{role}:{temperature}:{'paid' if use_paid else 'free'}"

    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    _ensure_env()
    from langchain_openrouter import ChatOpenRouter

    model_id = MODEL_REGISTRY.get(role, MODEL_REGISTRY["general"])
    if use_paid:
        model_id = MODEL_REGISTRY_PAID.get(role, MODEL_REGISTRY_PAID["general"])

    api_key = _get_openrouter_key()

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in .env")

    llm = ChatOpenRouter(
        api_key=api_key,
        model=model_id,
        temperature=temperature,
        timeout=60,  # 60 second timeout
    )
    _llm_cache[cache_key] = llm
    logger.info("Created ChatOpenRouter for role=%s model=%s%s temperature=%s",
                role, model_id, " (paid)" if use_paid else "", temperature)
    return llm


# ---------------------------------------------------------------------------
# Gemini Fallback — used when OpenRouter is rate-limited
# ---------------------------------------------------------------------------

_gemini_client = None


def _get_gemini_client():
    """Return a cached google.genai Client for fallback text generation."""
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    _ensure_env()
    api_key = _get_gemini_key()
    if not api_key:
        return None

    from google import genai
    _gemini_client = genai.Client(api_key=api_key)
    logger.info("Created Gemini fallback client (model=%s)", GEMINI_FALLBACK_MODEL)
    return _gemini_client


def _gemini_generate(prompt: str, system_prompt: str = "", temperature: float = 0.7) -> str:
    """
    Generate text using Gemini API as fallback.
    Uses gemini-3.6-flash (fast, free tier friendly).
    """
    client = _get_gemini_client()
    if client is None:
        return "[Fallback error: GEMINI_API_KEY not set]"

    try:
        from autojob.core.llm_router import _no_afc_config

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        response = client.models.generate_content(
            model=GEMINI_FALLBACK_MODEL,
            contents=full_prompt,
            config=_no_afc_config(temperature=temperature),
        )
        return response.text.strip()
    except Exception as exc:
        logger.exception("Gemini fallback also failed")
        return f"[Gemini fallback error: {exc}]"


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception is a rate-limit / too-many-requests error.

    Checks both exception type and HTTP status code if available.
    """
    exc_str = str(type(exc).__name__).lower() + str(exc).lower()

    # Check common rate limit keywords
    rate_limit_keywords = [
        "toomanyrequests", "rate_limit", "ratelimit", "429", "quota",
        "rate limit exceeded", "too many requests", "throttle",
    ]

    for kw in rate_limit_keywords:
        if kw in exc_str:
            return True

    # Check if exception has HTTP status code attribute (OpenAI/httpx exceptions)
    if hasattr(exc, 'status_code'):
        return exc.status_code == 429

    if hasattr(exc, 'response') and hasattr(exc.response, 'status_code'):
        return exc.response.status_code == 429

    return False


# ---------------------------------------------------------------------------
# Embeddings — NVIDIA nemotron-3-embed-1b (hosted)
# ---------------------------------------------------------------------------

_embeddings_instance = None


def get_embeddings():
    """Return a cached NVIDIA hosted embeddings instance."""
    global _embeddings_instance
    if _embeddings_instance is not None:
        return _embeddings_instance

    _ensure_env()
    nvidia_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not nvidia_key:
        raise RuntimeError("Embeddings unavailable: NVIDIA_API_KEY not set in .env")

    try:
        from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

        nvidia_model = os.getenv("NVIDIA_EMBEDDINGS_MODEL", "nvidia/nemotron-3-embed-1b")
        _embeddings_instance = NVIDIAEmbeddings(
            model=nvidia_model, api_key=nvidia_key, truncate="END",
        )
        logger.info("NVIDIA embeddings ready (model=%s)", nvidia_model)
        return _embeddings_instance
    except Exception as exc:
        logger.exception("Failed to load NVIDIA embeddings")
        raise RuntimeError(f"Embeddings model failed to load: {exc}") from exc


# ---------------------------------------------------------------------------
# Text Generation — OpenRouter (primary) → Gemini (fallback)
# ---------------------------------------------------------------------------

def generate_text(
    prompt: str,
    system_prompt: str = "",
    max_tokens: int = 1024,
    temperature: float = 0.7,
    role: str = "general",
    fast_task: str = "general",
) -> str:
    """
    Generate text with automatic retry + Gemini fallback.

    1. Try OpenRouter (LangChain ChatOpenRouter) with role-based model (free tier)
    2. If rate-limited → wait 3s → retry with paid model
    3. If still failing → fall back to Gemini text API (gemini-3.6-flash)
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))

    # Validate role
    valid_roles = set(MODEL_REGISTRY.keys())
    if role not in valid_roles:
        logger.warning("Invalid role '%s', defaulting to 'general'", role)
        role = "general"

    # ── Attempt 1: OpenRouter (free tier) ──
    try:
        llm = _get_llm(role, temperature=temperature, use_paid=False)
        response = llm.invoke(messages)
        return response.content.strip()

    except Exception as exc:
        if not _is_rate_limit_error(exc):
            # Not a rate limit — try Gemini directly
            logger.warning("OpenRouter failed (non-rate-limit), falling back to the LLM router: %s", exc)
            return _router_generate(prompt, system_prompt, temperature, max_tokens, fast_task)

        logger.warning("OpenRouter rate-limited, retrying with paid model in 3s...")

    # ── Attempt 2: Retry with paid model after delay ──
    time.sleep(3)
    try:
        llm = _get_llm(role, temperature=temperature, use_paid=True)
        response = llm.invoke(messages)
        return response.content.strip()

    except Exception as exc:
        logger.warning("OpenRouter paid model retry failed, falling back to Gemini: %s", exc)

    # ── Attempt 3: LLM router (NVIDIA -> Groq -> Gemini) ──
    return _router_generate(prompt, system_prompt, temperature, max_tokens, fast_task)


def _router_generate(prompt: str, system_prompt: str, temperature: float, max_tokens: int,
                     fast_task: str = "general") -> str:
    """
    Fallback through llm_router.fast_chat, which tries NVIDIA and Groq before
    Gemini. Going straight to Gemini burned its free tier (20 requests/day)
    whenever OpenRouter wasn't configured, and then every caller failed.
    """
    try:
        from autojob.core.llm_router import fast_chat

        text = fast_chat(prompt, task_type=fast_task, system_prompt=system_prompt,
                         temperature=temperature, max_tokens=max_tokens)
        if text and not text.startswith("[Error"):
            return text
        logger.warning("LLM router fallback returned no text: %s", (text or "")[:200])
    except Exception as exc:
        logger.warning("LLM router fallback failed: %s", exc)
    logger.info("Using Gemini fallback (model=%s)", GEMINI_FALLBACK_MODEL)
    return _gemini_generate(prompt, system_prompt, temperature)


# ---------------------------------------------------------------------------
# Grammar Checking — uses 'language' role model
# ---------------------------------------------------------------------------

def check_grammar(text: str) -> dict[str, Any]:
    """
    Check grammar of English text using LangChain ChatOpenRouter (language model).
    Returns corrections, explanation, and corrected text.
    """
    prompt = f"""Analyze this English sentence for grammar errors. Respond in JSON format ONLY.

Sentence: "{text}"

Return this exact JSON format:
{{
    "original": "{text}",
    "corrected": "<corrected sentence>",
    "is_correct": true/false,
    "errors": [
        {{
            "error": "<what's wrong>",
            "correction": "<how to fix it>",
            "explanation": "<why in simple Hindi-English mix>"
        }}
    ],
    "score": <0-100 grammar score>
}}"""

    result = generate_text(
        prompt=prompt,
        system_prompt="You are an English grammar expert. Always respond with valid JSON only, no markdown.",
        temperature=0.2,
        role="language",
    )

    try:
        # Try to parse JSON from the response
        # Handle potential markdown wrapping
        cleaned = result.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(cleaned)
    except (json.JSONDecodeError, IndexError):
        return {
            "original": text,
            "corrected": text,
            "is_correct": True,
            "errors": [],
            "score": 70,
            "raw_response": result,
        }


# ---------------------------------------------------------------------------
# Question Generation — uses 'reasoning' role model
# ---------------------------------------------------------------------------

def generate_question(
    context: str,
    topic: str = "",
    difficulty: str = "medium",
    question_type: str = "conceptual",
) -> dict[str, Any]:
    """
    Generate a question from given context using LangChain ChatOpenRouter (reasoning model).
    """
    prompt = f"""Generate ONE interview question from this context.

Context: {context[:2000]}
Topic: {topic or 'general'}
Difficulty: {difficulty}
Type: {question_type}

Respond in JSON format ONLY:
{{
    "question": "<the question>",
    "expected_answer_points": ["<key point 1>", "<key point 2>", ...],
    "difficulty": "{difficulty}",
    "topic": "<specific topic>",
    "hint": "<subtle hint>"
}}"""

    result = generate_text(
        prompt=prompt,
        system_prompt="You are an expert AI/ML interviewer. Generate challenging but fair questions. Always respond with valid JSON only.",
        temperature=0.8,
        role="reasoning",
    )

    try:
        cleaned = result.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(cleaned)
    except (json.JSONDecodeError, IndexError):
        return {
            "question": result,
            "expected_answer_points": [],
            "difficulty": difficulty,
            "topic": topic,
            "hint": "",
        }


# ---------------------------------------------------------------------------
# Answer Evaluation — uses 'evaluation' role model
# ---------------------------------------------------------------------------

def evaluate_answer(
    question: str,
    user_answer: str,
    reference_text: str = "",
) -> dict[str, Any]:
    """
    Evaluate a user's answer using LangChain ChatOpenRouter (evaluation model).
    """
    prompt = f"""Evaluate this interview answer. Be encouraging but honest.

Question: {question}
User's Answer: {user_answer}
Reference Material: {reference_text[:1500] if reference_text else 'None provided'}

Respond in JSON format ONLY:
{{
    "score": <0-100>,
    "feedback": "<detailed feedback in Hinglish - mix of Hindi and English>",
    "correct_points": ["<what they got right>"],
    "missed_points": ["<what they missed>"],
    "improvement_tip": "<one specific tip to improve>",
    "rating": "<excellent/good/average/needs_improvement>"
}}"""

    result = generate_text(
        prompt=prompt,
        system_prompt="You are a friendly AI/ML interview evaluator. Give constructive feedback in Hinglish. Always respond with valid JSON only.",
        temperature=0.3,
        role="evaluation",
    )

    try:
        cleaned = result.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(cleaned)
    except (json.JSONDecodeError, IndexError):
        return {
            "score": 50,
            "feedback": result,
            "correct_points": [],
            "missed_points": [],
            "improvement_tip": "Try to be more specific in your answers.",
            "rating": "average",
        }


# ---------------------------------------------------------------------------
# Vocabulary Word Generation — uses 'language' role model
# ---------------------------------------------------------------------------

def generate_vocabulary_word(level: str = "intermediate", known_words: list[str] = None) -> dict[str, Any]:
    """Generate a new English vocabulary word with examples using language model."""
    known_str = ", ".join(known_words[-20:]) if known_words else "none yet"

    prompt = f"""Generate ONE new English vocabulary word for a {level} learner.
Already known words: {known_str}

Pick a DIFFERENT word. Respond in JSON format ONLY:
{{
    "word": "<the word>",
    "pronunciation": "<phonetic pronunciation>",
    "meaning": "<meaning in simple English + Hindi translation>",
    "part_of_speech": "<noun/verb/adjective/etc>",
    "example_sentences": ["<example 1>", "<example 2>"],
    "usage_tip": "<when and how to use this word, in Hinglish>"
}}"""

    result = generate_text(
        prompt=prompt,
        system_prompt="You are an English vocabulary teacher for Indian students. Respond with valid JSON only.",
        temperature=0.9,
        role="language",
        # A short, low-stakes JSON reply: the local model answers in seconds with no quota.
        fast_task="quick",
    )

    try:
        cleaned = result.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(cleaned)
    except (json.JSONDecodeError, IndexError):
        return {
            "word": "articulate",
            "pronunciation": "ahr-TIK-yuh-layt",
            "meaning": "To express clearly (स्पष्ट रूप से व्यक्त करना)",
            "part_of_speech": "verb",
            "example_sentences": [
                "She articulated her ideas clearly in the meeting.",
                "He couldn't articulate his feelings."
            ],
            "usage_tip": "Jab tum kisi cheez ko clearly explain karte ho, tab 'articulate' use karo.",
        }
