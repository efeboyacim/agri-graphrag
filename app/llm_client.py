"""The only way AgriGraphRAG talks to an LLM: every call goes through the Portkey gateway.

- Refuses to work without PORTKEY_API_KEY (and GROQ_API_KEY); there is deliberately no
  fallback to calling Groq directly or to console-only logging.
- Each call is tagged with a trace_id, a span (named after the agent step) and metadata, so
  Portkey groups all LLM calls of one user query into one trace. The gateway itself records
  model, latency, token usage and cost for the request it proxies.
- The same numbers are mirrored locally (CALL_LOG) so /query, test_e2e.py and the eval
  scripts can print them without depending on Portkey's paid-tier read APIs.
"""
from __future__ import annotations

import logging
import os
import re
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app import config  # noqa: F401  (loads .env)

log = logging.getLogger("agrigraphrag.llm")

_PLACEHOLDERS = {"", "your-portkey-api-key", "your-groq-api-key"}

# Groq list prices, USD per 1M tokens (input, output). Portkey computes its own cost server-side;
# this table only feeds the local mirror shown in API responses and eval reports.
_PRICES_PER_M = {
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
}

# Optional saved Portkey config (e.g. gateway retries), referenced by its "pc-..." slug. Inline configs
# are not sent because many Portkey workspaces enforce block_inline_config.
_PORTKEY_CONFIG_ID = os.getenv("PORTKEY_CONFIG", "").strip() or None
# Client-side retries for Groq rate limits / transient gateway errors, honouring retry-after.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_LOCAL_WAIT_S = 90
_LOCAL_RETRIES = 4

T = TypeVar("T", bound=BaseModel)


class PortkeyNotConfiguredError(RuntimeError):
    """Raised when the Portkey gateway (or the Groq key behind it) is not configured."""


class LLMRateLimitError(RuntimeError):
    """Raised when Groq keeps rate-limiting us beyond what is sensible to wait for."""


@dataclass
class LLMCall:
    trace_id: str
    span_id: str
    step: str
    model: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    portkey_trace_id: str | None  # echoed back by the gateway -> proof the call went through Portkey

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


CALL_LOG: deque[LLMCall] = deque(maxlen=10_000)


def new_trace_id(prefix: str = "agri") -> str:
    return f"{prefix}-{uuid.uuid4()}"


def new_span_id() -> str:
    return uuid.uuid4().hex[:16]


def _provider() -> str:
    """'@slug' = a saved Portkey Model Catalog integration (Groq key stored in Portkey);
    'groq' = inline provider, with GROQ_API_KEY forwarded through the gateway."""
    return os.getenv("PORTKEY_PROVIDER", "").strip() or "groq"


def _uses_saved_integration() -> bool:
    return _provider().startswith("@")


def require_portkey() -> None:
    """Fail fast unless the Portkey gateway is configured. Called at app startup."""
    required = ["PORTKEY_API_KEY"] + ([] if _uses_saved_integration() else ["GROQ_API_KEY"])
    missing = [name for name in required if os.getenv(name, "").strip() in _PLACEHOLDERS]
    if missing:
        raise PortkeyNotConfiguredError(
            f"{' and '.join(missing)} not set. AgriGraphRAG routes every LLM call through the "
            "Portkey gateway (for trace/latency/token/cost visibility) and will not start without "
            "it; there is no fallback to direct Groq calls or console-only logging. "
            "Copy .env.example to .env and fill in the REQUIRED keys."
        )


@lru_cache(maxsize=1)
def get_client():
    require_portkey()
    from portkey_ai import Portkey

    auth = {} if _uses_saved_integration() else {"Authorization": f"Bearer {os.environ['GROQ_API_KEY'].strip()}"}
    return Portkey(
        api_key=os.environ["PORTKEY_API_KEY"].strip(),
        provider=_provider(),
        config=_PORTKEY_CONFIG_ID,
        **auth,
    )


def calls_for_trace(trace_id: str) -> list[LLMCall]:
    return [c for c in CALL_LOG if c.trace_id == trace_id]


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price_in, price_out = _PRICES_PER_M.get(model, (0.0, 0.0))
    return round((prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000, 8)


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    value = headers.get("retry-after") if hasattr(headers, "get") else None
    if value:
        try:
            return float(value)
        except ValueError:
            pass
    # Groq puts the wait in the message too: "... Please try again in 6m11.52s."
    match = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", str(exc))
    if match and any(match.groups()):
        hours, minutes, seconds = (float(g) if g else 0.0 for g in match.groups())
        return hours * 3600 + minutes * 60 + seconds
    return None


def _create_with_retries(client, request: dict[str, Any]):
    for attempt in range(_LOCAL_RETRIES + 1):
        try:
            return client.chat.completions.create(**request)
        except Exception as exc:  # the SDK raises vendored openai exception classes
            status = getattr(exc, "status_code", None)
            if status not in _RETRYABLE_STATUS:
                raise
            wait = _retry_after_seconds(exc)
            if attempt == _LOCAL_RETRIES or (wait is not None and wait > _MAX_LOCAL_WAIT_S):
                if status != 429:
                    raise
                hint = f" (Groq asks to wait ~{wait:.0f}s)" if wait else ""
                raise LLMRateLimitError(
                    f"Groq rate limit for {request['model']} still exceeded after {attempt + 1} attempt(s){hint}. "
                    "On the free tier this usually means the daily token cap was reached; retry later "
                    "or evaluate fewer cases with --limit."
                ) from exc
            wait = wait if wait is not None else min(5.0 * 2**attempt, 60.0)
            log.warning("HTTP %s from Groq/Portkey for %s; retrying in %.1fs", status, request["model"], wait)
            time.sleep(wait + 0.5)
    raise AssertionError("unreachable")


def chat(
    messages: list[dict[str, str]],
    *,
    model: str,
    step: str,
    trace_id: str,
    parent_span_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    json_mode: bool = False,
    temperature: float = 0.2,
    max_tokens: int = 1500,
) -> str:
    """Send one chat completion through Portkey and return the assistant text."""
    span_id = new_span_id()
    meta = {"app": "agrigraphrag", "step": step, "model": model, **(metadata or {})}
    options: dict[str, Any] = {
        "trace_id": trace_id,
        "span_id": span_id,
        "span_name": step,
        # Portkey metadata values must be strings of at most 128 characters.
        "metadata": {k: str(v)[:128] for k, v in meta.items()},
    }
    if parent_span_id:
        options["parent_span_id"] = parent_span_id
    client = get_client().with_options(**options)

    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if model.startswith("openai/gpt-oss"):
        # Reasoning tokens count against the token budget; keep them small.
        request["reasoning_effort"] = "low"
    if json_mode:
        request["response_format"] = {"type": "json_object"}

    started = time.perf_counter()
    response = _create_with_retries(client, request)
    choice = response.choices[0]
    if not (choice.message.content or "").strip() and choice.finish_reason == "length":
        # Reasoning models can burn the whole budget before answering; retry once with more room.
        request["max_tokens"] = max_tokens * 2
        response = _create_with_retries(client, request)
        choice = response.choices[0]
    latency_ms = int((time.perf_counter() - started) * 1000)

    usage = response.usage
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    try:
        gateway_headers = response.get_headers() or {}
    except Exception:  # pragma: no cover - header access is best effort
        gateway_headers = {}
    CALL_LOG.append(
        LLMCall(
            trace_id=trace_id,
            span_id=span_id,
            step=step,
            model=model,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=estimate_cost(model, prompt_tokens, completion_tokens),
            portkey_trace_id=gateway_headers.get("trace-id") or gateway_headers.get("x-portkey-trace-id"),
        )
    )
    return _clean(choice.message.content)


# gpt-oss likes typographic characters (e.g. "FungiStop‑X" with a non-breaking hyphen, or citations
# in CJK brackets like "【G1】"), which break plain-text matching; map them to ASCII equivalents.
_TYPOGRAPHY = str.maketrans({
    "‐": "-", "‑": "-", " ": " ", " ": " ", " ": " ", "【": "[", "】": "]",
})


def _clean(text: str | None) -> str:
    return (text or "").translate(_TYPOGRAPHY).strip()


def _extract_json(text: str) -> str:
    """Tolerate code fences or chatter around a JSON object."""
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


def chat_json(messages: list[dict[str, str]], schema: type[T], **kwargs: Any) -> T:
    """chat() in JSON mode, validated against a pydantic schema (one corrective retry)."""
    text = ""
    error: Exception | None = None
    for attempt in range(2):
        request_messages = messages
        if attempt and text:
            request_messages = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"That JSON was invalid ({error}). Reply with corrected JSON only."},
            ]
        try:
            text = chat(request_messages, json_mode=True, **kwargs)
            return schema.model_validate_json(_extract_json(text))
        except ValidationError as exc:
            error = exc
        except Exception as exc:
            # Groq answers 400 json_validate_failed when the model produced malformed JSON.
            if getattr(exc, "status_code", None) != 400:
                raise
            error = exc
    raise ValueError(f"{kwargs.get('step')}: model did not return valid {schema.__name__} JSON: {error}")
