"""
llm_router.py — Multi-Provider LLM Router for Veronica AI

Routes AI tasks to the fastest available provider:
  - Groq      → Wake word transcription (Whisper), fast chat (LLaMA)
  - NVIDIA NIM → Code generation (Nemotron)
  - Gemini    → Last resort (free tier is only ~20 requests/day)

All API keys are read from environment variables (.env loaded by main.py).
"""

import io
import logging
import os
import re
import time
from pathlib import Path
from autojob.core.paths import ENV_FILE
from typing import Optional

logger = logging.getLogger(__name__)

def _load_env_file():
    """Ensure .env is loaded into os.environ."""
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

_load_env_file()

# ---------------------------------------------------------------------------
# Provider clients (lazy-initialised singletons)
# ---------------------------------------------------------------------------

_groq_client = None
_nvidia_client = None

# Providers whose credentials the API has explicitly rejected. A bad key does
# not fix itself, so once we see a 401 we stop calling that provider for the
# rest of the process instead of paying a round-trip on every request.
_disabled_providers: set[str] = set()


def _is_auth_error(exc: Exception) -> bool:
    """True for 401/403-style credential rejections."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(
        kw in text
        for kw in ("401", "403", "invalid api key", "invalid_api_key",
                   "unauthorized", "authentication", "permission denied")
    )


def _is_rate_limit_error(exc: Exception) -> bool:
    """True for 429 / quota-exhausted responses."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(
        kw in text
        for kw in ("429", "resource_exhausted", "rate limit", "rate_limit", "quota")
    )


# Rate-limited providers → monotonic time at which they may be tried again.
# Unlike a bad key, a quota does recover, so we pause instead of disabling.
_cooldown_until: dict[str, float] = {}

# Per-day quotas report a short retryDelay that is meaningless — retrying in
# ~40s just burns another request and gets another 429. Back off much longer.
_DAILY_QUOTA_COOLDOWN_S = 30 * 60
_DEFAULT_RATE_LIMIT_COOLDOWN_S = 60


def _rate_limit_cooldown_seconds(exc: Exception) -> float:
    text = str(exc)
    if "PerDay" in text or "per day" in text.lower():
        return _DAILY_QUOTA_COOLDOWN_S
    minutes = re.search(r"try again in (\d+)m([\d.]+)s", text, re.IGNORECASE)  # Groq: "6m33.98s"
    if minutes:
        return int(minutes.group(1)) * 60 + float(minutes.group(2))
    match = re.search(r"(?:retry|try again) in ([\d.]+)s", text, re.IGNORECASE) or re.search(
        r"retryDelay'?\"?:\s*'?\"?(\d+)s", text
    )
    if match:
        return max(float(match.group(1)), 5.0)
    return _DEFAULT_RATE_LIMIT_COOLDOWN_S


def _in_cooldown(name: str) -> bool:
    until = _cooldown_until.get(name)
    if until is None:
        return False
    if time.monotonic() >= until:
        del _cooldown_until[name]
        logger.info("%s rate-limit cooldown over — retrying", name.upper())
        return False
    return True


def _start_cooldown(name: str, exc: Exception) -> None:
    seconds = _rate_limit_cooldown_seconds(exc)
    _cooldown_until[name] = time.monotonic() + seconds
    logger.warning(
        "%s rate limit / quota exhausted — pausing it for %.0fs. (%s)",
        name.upper(), seconds, str(exc).split("{", 1)[0].strip(),
    )


def _disable_provider(name: str, exc: Exception) -> None:
    if name in _disabled_providers:
        return
    _disabled_providers.add(name)
    logger.error(
        "%s credentials rejected (%s). Disabling %s for this session — "
        "fix the API key in .env and restart to re-enable.",
        name.upper(), exc, name.upper(),
    )


def get_groq_client():
    """Return a Groq client (lazy init). Returns None if key missing/rejected."""
    global _groq_client
    if "groq" in _disabled_providers:
        return None
    if _groq_client is not None:
        return _groq_client
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        logger.warning("GROQ_API_KEY not set — Groq provider unavailable")
        _disabled_providers.add("groq")
        return None
    try:
        from groq import Groq
        _groq_client = Groq(
            api_key=api_key,
            timeout=30.0,  # 30 second timeout
            # A 429 made the SDK sleep ~31 s and retry; the router's own fallback
            # (NVIDIA, Ollama, Gemini) answers right away instead.
            max_retries=0,
        )
        logger.info("Groq client initialised with 30s timeout")
        return _groq_client
    except Exception as exc:
        logger.error("Failed to create Groq client: %s", exc)
        return None


def get_nvidia_client():
    """Return an OpenAI-compatible client pointing at NVIDIA NIM."""
    global _nvidia_client
    if "nvidia" in _disabled_providers:
        return None
    if _nvidia_client is not None:
        return _nvidia_client
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        logger.warning("NVIDIA_API_KEY not set — NVIDIA provider unavailable")
        _disabled_providers.add("nvidia")
        return None
    try:
        from openai import OpenAI
        _nvidia_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
            timeout=60.0,  # 60 second timeout for NVIDIA (longer for reasoning models)
            # The SDK's own 2 retries turned one slow model into a 3-minute wait;
            # the router's fallback chain (previous model, Groq, Ollama, Gemini) is faster.
            max_retries=0,
        )
        logger.info("NVIDIA NIM client initialised with 60s timeout")
        return _nvidia_client
    except Exception as exc:
        logger.error("Failed to create NVIDIA client: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Fast transcription (Groq Whisper → Gemini Flash fallback)
# ---------------------------------------------------------------------------

def _no_afc_config(**kwargs):
    """
    Build a GenerateContentConfig with automatic function calling switched off.

    google-genai turns AFC on for *any* config, even one with no tools, and
    then warns that direct AFC use in generate_content is discouraged. None of
    these calls use function calling, so we opt out explicitly: no warning and
    no AFC bookkeeping on every request.
    """
    from google.genai import types

    return types.GenerateContentConfig(
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        **kwargs,
    )


def fast_transcribe(wav_bytes: bytes) -> str:
    """
    Transcribe audio bytes using the fastest available provider.
    Priority: Groq Whisper-large-v3 (~200 ms) → Gemini Flash (~1 s).
    """
    # --- Try Groq Whisper first ---
    client = None if _in_cooldown("groq") else get_groq_client()
    if client is not None:
        try:
            audio_file = io.BytesIO(wav_bytes)
            audio_file.name = "wake.wav"
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                language="en",
                response_format="text",
            )
            text = (transcription or "").strip()
            if text:
                logger.info("Groq Whisper transcription: %s", text)
                return text
        except Exception as exc:
            if _is_auth_error(exc):
                # Logged once, then Groq is skipped entirely. Previously this
                # warned on every wake-word check and added a wasted network
                # round-trip to each one.
                _disable_provider("groq", exc)
            elif _is_rate_limit_error(exc):
                _start_cooldown("groq", exc)
            else:
                logger.warning("Groq Whisper failed, falling back to Gemini: %s", exc)

    # --- Fallback to Gemini Flash ---
    if _in_cooldown("gemini"):
        # Quota exhausted — skip silently rather than 429 + error log per utterance.
        return ""
    try:
        from google import genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("No Gemini API key available for transcription fallback")
            return ""
        gemini_client = genai.Client(api_key=api_key)
        # Use model from env or default
        transcription_model = os.getenv("GEMINI_TRANSCRIPTION_MODEL", "gemini-3.6-flash")
        response = gemini_client.models.generate_content(
            model=transcription_model,
            config=_no_afc_config(),
            contents=[
                types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                "Transcribe this audio precisely. If you hear the name 'Veronica' "
                "(or sounds like 'Weronica', 'Ronica', 'Veronica'), transcribe the "
                "text. If you don't hear it, just reply 'No'.",
            ],
        )
        text = (response.text or "").strip()
        logger.info("Gemini transcription (%s): %s", transcription_model, text)
        return text
    except Exception as exc:
        if _is_rate_limit_error(exc):
            _start_cooldown("gemini", exc)
        else:
            logger.error("Gemini transcription fallback failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Fast chat — routes by task type
# ---------------------------------------------------------------------------

# Model routing table - each task goes to the fastest model that does it well.
# Measured on NIM 2026-10-17: Super VL answered in 1-8 s with thinking off while
# Lightning sat in the queue for 55-100 s; Step 5 solved a reasoning check in
# 3.6 s while Ultra was overloaded.
_FAST_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"  # 12B active, sees images/video: everyday chat + vision
_LIGHT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"  # previous fast model, the fallback
_REASONING_MODEL = "stepfun-ai/step-5-preview"          # 27B active, 1M context: multi-step reasoning
_DEEP_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"       # previous reasoning model, the fallback
_CODE_MODEL = "z-ai/glm-5.3-flash"                      # fast coding and UI/design generation
_CREATIVE_MODEL = "meta/muse-glimmer-30b"               # conversational / creative voice
_VISION_MODEL = _FAST_MODEL                             # NIM model for screenshots / images

# NIM chat-template switch that turns off the thinking phase.
_NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}

# Models that keep their thinking phase. Step 5 ignores enable_thinking=False and
# writes its reasoning into the answer instead, so it must think in its own channel.
_THINKING_MODELS = {_DEEP_MODEL, _REASONING_MODEL}

_TASK_MODELS = {
    "email":     ("nvidia", _FAST_MODEL),
    "grammar":   ("nvidia", _FAST_MODEL),
    "general":   ("nvidia", _FAST_MODEL),
    "browser":   ("nvidia", _FAST_MODEL),   # each agent step must answer in seconds
    "code":      ("nvidia", _CODE_MODEL),
    "design":    ("nvidia", _CODE_MODEL),
    "interview": ("nvidia", _FAST_MODEL),   # live Q&A: latency matters most
    "creative":  ("nvidia", _CREATIVE_MODEL),
    "reasoning": ("nvidia", _REASONING_MODEL),
    "tts":       ("nvidia", "nvidia/magpie-tts-zeroshot"),
    "quick":     ("nvidia", _FAST_MODEL),   # one-liners, yes/no, quick definitions
}

# Models each task used before the current routing; known to work, so they are
# the first fallback when the new model fails or returns an empty reply.
_PREVIOUS_TASK_MODELS = {
    "email":     _LIGHT_MODEL,
    "grammar":   _LIGHT_MODEL,
    "general":   _LIGHT_MODEL,
    "browser":   _LIGHT_MODEL,
    "quick":     _LIGHT_MODEL,
    "reasoning": _DEEP_MODEL,
    "code":      "nvidia/minimaxai/minimax-m3",
    "design":    "nvidia/minimaxai/minimax-m3",
    "interview": "nvidia/minimaxai/minimax-m3",
    "creative":  _DEEP_MODEL,
}

# Groq model used as the default route and as the alternate when NVIDIA fails.
_GROQ_FALLBACK_MODEL = "qwen/qwen3.8-27b"

# Wall-clock cap for a whole NVIDIA streaming response. The client timeout only
# bounds each read, so a slow reasoning model can otherwise stream for many
# minutes; past this deadline we give up and let fast_chat fall through.
_NVIDIA_STREAM_DEADLINE_S = 105.0   # the reasoning (thinking) models
_NVIDIA_FAST_DEADLINE_S = 45.0      # everything else answers in seconds or is stuck

# A model that timed out or that NIM reports DEGRADED is skipped for a while
# instead of costing another minute on every call (fast_chat falls through).
_MODEL_COOLDOWN_S = 600.0
_model_cooldown_until: dict[str, float] = {}


def _model_resting(model: str) -> bool:
    return time.monotonic() < _model_cooldown_until.get(model, 0.0)


def _rest_model(model: str, why: str) -> None:
    _model_cooldown_until[model] = time.monotonic() + _MODEL_COOLDOWN_S
    logger.warning("NVIDIA [%s] %s; skipping it for %.0f min", model, why, _MODEL_COOLDOWN_S / 60)

# NVIDIA model aliases for easy reference
_NVIDIA_MODELS = {
    "nemotron-70b": "nvidia/llama-3.1-nemotron-70b-instruct",
    "glm-5.2": "z-ai/glm-5.2",
    "glm-5.3-flash": _CODE_MODEL,
    "super-vl": _FAST_MODEL,
    "lightning": _LIGHT_MODEL,
    "step-5": _REASONING_MODEL,
    "ultra": _DEEP_MODEL,
    "muse-glimmer": _CREATIVE_MODEL,
    "magpie-tts": "nvidia/magpie-tts-zeroshot",
    "llama-3.1-nemotron-70b": "nvidia/llama-3.1-nemotron-70b-instruct",
}


def get_nvidia_model(alias: str) -> str:
    """Resolve NVIDIA model alias to full model ID."""
    return _NVIDIA_MODELS.get(alias, alias)

# Task-specific temperature settings (lower = more deterministic)
_TASK_TEMPERATURES = {
    "email":     0.3,
    "grammar":   0.2,
    "general":   0.7,
    "code":      0.2,
    "design":    0.4,
    "interview": 0.4,
    "creative":  0.8,
    "reasoning": 0.3,
    "browser":   0.1,
    "quick":     0.3,
}

# Task-specific max tokens
_TASK_MAX_TOKENS = {
    "email":     2048,
    "grammar":   1024,
    "general":   2048,
    "code":      4096,
    "design":    4096,
    "interview": 2048,
    "creative":  2048,
    # Reasoning models emit long chains of thought — give them headroom.
    "reasoning": 4096,
    "browser":   4096,   # long JSON plans on busy pages
    "quick":     400,
}


def fast_chat(
    prompt: str,
    task_type: str = "general",
    system_prompt: str = "",
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Send a chat completion to the fastest available provider.

    task_type controls routing:
      "email" / "grammar" / "general"             → Nemotron 3.5 Super VL (fast)
      "browser"                                   → Nemotron 3.5 Super VL (fast)
      "code" / "design"                           → GLM-5.3 Flash
      "interview"                                 → Nemotron 3.5 Super VL (fast)
      "creative"                                  → Muse Glimmer 30B
      "reasoning"                                 → Step 5 Preview (thinking on)
      "quick"                                     → Nemotron 3.5 Super VL (short answers)

    Failure chain: primary → previous NVIDIA model → alternate provider → Gemini.
    """
    # Use task-specific defaults if not explicitly provided
    if temperature is None:
        temperature = _TASK_TEMPERATURES.get(task_type, 0.7)
    if max_tokens is None:
        max_tokens = _TASK_MAX_TOKENS.get(task_type, 2048)

    provider, model = _TASK_MODELS.get(task_type, ("groq", _GROQ_FALLBACK_MODEL))

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # --- Try primary provider ---
    result = _try_provider(provider, model, messages, temperature, max_tokens)
    if result:
        return result

    # --- New model failed: retry on the task's previous NVIDIA model ---
    previous = _PREVIOUS_TASK_MODELS.get(task_type)
    # A primary that just stalled (rested) means NIM itself is slow: another NIM
    # model would only add another timeout, so go straight to Groq below.
    if provider == "nvidia" and previous and previous != model and not _model_resting(model):
        logger.info("NVIDIA [%s] gave no answer; retrying on previous model %s", model, previous)
        result = _try_provider("nvidia", previous, messages, temperature, max_tokens)
        if result:
            return result

    # --- Try alternate fast provider ---
    if provider == "nvidia":
        alt_result = _try_provider("groq", _GROQ_FALLBACK_MODEL, messages, temperature, max_tokens)
        if alt_result:
            return alt_result
    elif provider == "groq":
        alt_result = _try_provider("nvidia", _FAST_MODEL, messages, temperature, max_tokens)
        if alt_result:
            return alt_result

    # --- Final fallback to Gemini ---
    return _try_gemini_chat(prompt, system_prompt, temperature)


def _try_provider(
    provider: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> Optional[str]:
    """Attempt a chat completion on the given provider. Returns None on failure."""
    if provider == "groq" and _in_cooldown("groq"):
        return None  # rate-limited a moment ago: go straight to the next provider
    if provider == "nvidia" and _model_resting(model):
        return None  # timed out / degraded a moment ago
    try:
        if provider == "groq":
            client = get_groq_client()
            if client is None:
                return None
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content or ""
            logger.info("Groq [%s] response: %s chars", model, len(text))
            return text.strip()

        elif provider == "nvidia":
            client = get_nvidia_client()
            if client is None:
                return None
            
            
            create_kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": 0.95,
                "max_tokens": max_tokens,
                "stream": True,
            }
            
            # Hybrid reasoning models think before answering by default: ~10x the
            # tokens and latency, and a short max_tokens can be used up entirely
            # by thinking (empty reply). Only the reasoning models keep thinking on.
            thinking = model in _THINKING_MODELS
            if not thinking:
                create_kwargs["extra_body"] = _NO_THINKING
            # Per-read timeout: a healthy fast model starts streaming within seconds; a
            # stalled one used to hold every caller for the full 60 s before Groq got a try.
            create_kwargs["timeout"] = 45.0 if thinking else 15.0

            completion = client.chat.completions.create(**create_kwargs)
            
            deadline = time.monotonic() + (_NVIDIA_STREAM_DEADLINE_S if thinking
                                           else _NVIDIA_FAST_DEADLINE_S)
            content_parts = []
            for chunk in completion:
                if time.monotonic() > deadline:
                    _rest_model(model, "is too slow (stream deadline passed)")
                    close = getattr(completion, "close", None)
                    if close:
                        close()
                    return None
                if not chunk.choices:
                    continue
                # You can log reasoning_content if needed
                delta = chunk.choices[0].delta
                text_chunk = getattr(delta, "content", None)
                if text_chunk:
                    content_parts.append(text_chunk)
            
            text = "".join(content_parts).strip()
            logger.info("NVIDIA NIM [%s] response: %s chars", model, len(text))
            return text

    except Exception as exc:
        if _is_auth_error(exc):
            _disable_provider(provider, exc)
        elif provider == "groq" and _is_rate_limit_error(exc):
            _start_cooldown("groq", exc)
        elif provider == "nvidia" and re.search(r"timed out|timeout|degraded|cannot be invoked|50[234]",
                                                str(exc), re.I):
            _rest_model(model, f"failed ({str(exc)[:80]})")
        else:
            logger.warning("%s [%s] failed: %s", provider.upper(), model, exc)
        return None


def _try_gemini_chat(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.7,
) -> str:
    """Final fallback — Gemini 3.6 Flash via google.genai."""
    try:
        from google import genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "[Error: No API key available for any provider]"
        client = genai.Client(api_key=api_key)
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=full_prompt,
            config=_no_afc_config(temperature=temperature),
        )
        text = (response.text or "").strip()
        logger.info("Gemini Flash fallback response: %s chars", len(text))
        return text
    except Exception as exc:
        logger.error("All providers failed. Gemini fallback error: %s", exc)
        return f"[Error: All AI providers failed — {exc}]"
