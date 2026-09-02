# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""MCP server backed by Serper for web search and webpage scraping with
forced LLM summary.

A single Serper backend powers the web workflow, with a hardened scrape
pipeline ported from ``searching_zhipu_mcp_server.py``:

* ``web_search``     - ``POST https://google.serper.dev/search``
                       (Google SERP via Serper, organic + rich blocks)
* ``scrape_website`` - Serper Scrape (primary) + Python httpx (fallback)
                       → **forced** LLM extraction (no raw-content
                       passthrough).


  1. **Fully async** — ``httpx.AsyncClient`` end-to-end (search / scrape /
     summary), so high concurrency no longer blocks on sync ``requests``.
  2. **Python httpx fallback scrape** — if Serper scrape fails, fall back
     to a plain ``httpx`` GET with a browser User-Agent.
  3. **FORCED summary** — ``scrape_website`` ALWAYS runs LLM extraction;
     if ``info_to_extract`` is empty, a generic focus is used.
     ``ENABLE_SCRAPE_SUMMARY`` is removed.
  4. **Repeat-output detection** — if the summary LLM degenerates
     (last 50 chars repeat > 5×), retry.
  5. **Scrape content cap** — keep at most ``MAX_SCRAPE_CHARS`` chars
     from a scraped page before summary.

What is intentionally KEPT:
  * The 4-provider summary LLM support (gemini / azure / maas / openai)
    with auto-detection.
  * Serper as the primary search + scrape backend.
  * ``decode_http_urls_in_dict`` on search output for URL readability.
  * MaaS reasoning-model special handling (enable_thinking=false +
    reasoning_content fallback).

Environment variables
---------------------
SERPER_API_KEY        (required) Serper API key.
SERPER_BASE_URL       (optional) Search endpoint base, default
                       ``https://google.serper.dev``.
SERPER_SCRAPE_URL     (optional) Scrape endpoint, default
                       ``https://scrape.serper.dev``.
SERPER_SEARCH_LIMIT   (optional, int, default 10) ``num`` parameter.
SERPER_DEFAULT_GL     (optional, default ``us``) Google country code.
SERPER_DEFAULT_HL     (optional, default ``en``) Google interface language.
MAX_SCRAPE_CHARS      (optional, int, default 409600) max chars before summary.

Summary LLM (shared semantics with searching_zhipu_mcp_server_dev):
SUMMARY_LLM_BASE_URL, SUMMARY_LLM_MODEL_NAME, SUMMARY_LLM_API_KEY,
SUMMARY_LLM_PROVIDER ("openai" | "azure" | "gemini" | "maas";
                      auto-inferred if empty),
SUMMARY_LLM_API_VERSION (Azure only).
"""

import asyncio
import json
import logging
import os
import re
from typing import Annotated, Any, Dict, List, Optional

from pydantic import Field

import httpx
from mcp.server.fastmcp import FastMCP

from .utils import decode_http_urls_in_dict, strip_markdown_links


# ── Shared async HTTP client (connection-pool reuse) ─────────────────
_HTTP_CLIENT: Optional[httpx.AsyncClient] = None


def _get_http_client() -> httpx.AsyncClient:
    """Return the module-level shared ``httpx.AsyncClient``, creating it
    lazily on first call."""
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None or _HTTP_CLIENT.is_closed:
        _HTTP_CLIENT = httpx.AsyncClient(
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=20,
            ),
        )
    return _HTTP_CLIENT


# ── HTML → plain-text helper (no external deps) ─────────────────────
_RE_SCRIPT_STYLE = re.compile(
    r"<\s*(script|style)[^>]*>.*?</\s*\1\s*>", re.DOTALL | re.IGNORECASE
)
_RE_TAG = re.compile(r"<[^>]+>")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
_RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def _strip_html_tags(html: str) -> str:
    """Lightweight HTML → text conversion using regex only (no extra deps).

    Strips <script>/<style> blocks, all tags, collapses whitespace.
    Good-enough for turning a fallback httpx response into readable text.
    """
    text = _RE_SCRIPT_STYLE.sub("", html)
    text = _RE_TAG.sub("", text)
    text = _RE_MULTI_SPACE.sub(" ", text)
    text = _RE_MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


# ── Configuration ─────────────────────────────────────────────────────
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SERPER_BASE_URL = os.getenv("SERPER_BASE_URL", "https://google.serper.dev").rstrip("/")
SERPER_SCRAPE_URL = os.getenv("SERPER_SCRAPE_URL", "https://scrape.serper.dev").rstrip("/")
SERPER_SEARCH_LIMIT = int(os.getenv("SERPER_SEARCH_LIMIT", "10"))
SERPER_DEFAULT_GL = os.getenv("SERPER_DEFAULT_GL", "us")
SERPER_DEFAULT_HL = os.getenv("SERPER_DEFAULT_HL", "en")

# Max chars to keep from a scraped page before summary.
MAX_SCRAPE_CHARS = int(os.getenv("MAX_SCRAPE_CHARS", str(102400 * 4)))

# ── Summary LLM Configuration ────────────────────────────────────────
SUMMARY_LLM_BASE_URL = os.getenv("SUMMARY_LLM_BASE_URL", "")
SUMMARY_LLM_MODEL_NAME = os.getenv("SUMMARY_LLM_MODEL_NAME", "")
SUMMARY_LLM_API_KEY = os.getenv("SUMMARY_LLM_API_KEY", "")
SUMMARY_LLM_API_VERSION = os.getenv("SUMMARY_LLM_API_VERSION", "")

# Auto-detect provider: "gemini" | "azure" | "maas" | "openai"
_provider_env = os.getenv("SUMMARY_LLM_PROVIDER", "").strip().lower()
if _provider_env:
    SUMMARY_LLM_PROVIDER = _provider_env
elif "gemini" in SUMMARY_LLM_MODEL_NAME.lower():
    SUMMARY_LLM_PROVIDER = "gemini"
else:
    SUMMARY_LLM_PROVIDER = "openai"


def _derive_summary_model_label() -> str:
    """Human-friendly model label for trace logs (``model_used`` field).

    Falls back to "unknown" only when nothing else is available. For MaaS,
    pull the routing slug from the URL path when ``SUMMARY_LLM_MODEL_NAME``
    is empty.
    """
    if SUMMARY_LLM_MODEL_NAME:
        return SUMMARY_LLM_MODEL_NAME
    if SUMMARY_LLM_PROVIDER == "maas" and SUMMARY_LLM_BASE_URL:
        try:
            from urllib.parse import urlparse
            parts = [p for p in urlparse(SUMMARY_LLM_BASE_URL).path.split("/") if p]
            if parts:
                return f"maas:{parts[0]}"
        except Exception:
            pass
        return "maas"
    return "unknown"


SUMMARY_MODEL_LABEL = _derive_summary_model_label()

logger = logging.getLogger("searching-serper-mcp-server")
mcp = FastMCP("searching-serper-mcp-server")


# ── Async retry helpers ──────────────────────────────────────────────
_RETRYABLE_STATUS = {408, 429}


def _is_retryable_status(status_code: int) -> bool:
    return status_code >= 500 or status_code in _RETRYABLE_STATUS


# ── Common helpers ───────────────────────────────────────────────────
def _is_huggingface_dataset_or_space_url(url: str) -> bool:
    """Filter out HF dataset / space URLs (prevents benchmark answer-leakage)."""
    if not url:
        return False
    return "huggingface.co/datasets" in url or "huggingface.co/spaces" in url


def _serper_headers() -> Dict[str, str]:
    return {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }


def _missing_key_json() -> str:
    return json.dumps(
        {
            "success": False,
            "error": "SERPER_API_KEY environment variable not set",
            "results": [],
        },
        ensure_ascii=False,
    )


# ── Async HTTP POST to Serper with retry ─────────────────────────────
async def _serper_post(
    url: str, payload: Dict[str, Any]
) -> httpx.Response:
    """POST to a Serper endpoint with async retry (manual loop)."""
    retry_delays = [1, 2, 4, 8]
    last_exc: Optional[Exception] = None
    client = _get_http_client()
    for attempt, delay in enumerate(retry_delays, 1):
        try:
            resp = await client.post(
                url,
                json=payload,
                headers=_serper_headers(),
                timeout=httpx.Timeout(None, connect=20, read=30),
            )
            if _is_retryable_status(resp.status_code):
                if attempt < len(retry_delays):
                    logger.info(
                        "Serper: HTTP %s (retryable), retry in %ss",
                        resp.status_code,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                resp.raise_for_status()
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError as e:
            last_exc = e
            status = e.response.status_code if e.response is not None else 0
            if _is_retryable_status(status) and attempt < len(retry_delays):
                logger.info(
                    "Serper: HTTP %s (retryable), retry in %ss", status, delay
                )
                await asyncio.sleep(delay)
                continue
            raise
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
            last_exc = e
            if attempt < len(retry_delays):
                logger.info(
                    "Serper: %s, retry in %ss", type(e).__name__, delay
                )
                await asyncio.sleep(delay)
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("Serper: retry loop exhausted without response")


# ── web_search tool (async) ──────────────────────────────────────────
@mcp.tool()
async def web_search(
    q: Annotated[str, Field(description="Search query string.")],
) -> str:
    """Perform a web search via Serper (Google SERP).

    Args:
        q: Search query string.

    Returns:
        JSON string with the same shape as the other ``*_search`` backends::

            {
              "searchParameters": {"q": "...", "engine": "serper",
                                   "gl": "...", "hl": "...", "num": N},
              "organic": [{"title", "link", "snippet", "media", "date"}, ...],
              "answerBox": {...} (optional),
              "knowledgeGraph": {...} (optional),
              "peopleAlsoAsk": [...] (optional),
              "relatedSearches": [...] (optional)
            }

        Or on failure::

            {"success": false, "error": "...", "results": []}
    """
    if not SERPER_API_KEY:
        return _missing_key_json()
    if not q or not q.strip():
        return json.dumps(
            {
                "success": False,
                "error": "Search query 'q' is required and cannot be empty",
                "results": [],
            },
            ensure_ascii=False,
        )

    payload: Dict[str, Any] = {
        "q": q.strip(),
        "gl": SERPER_DEFAULT_GL,
        "hl": SERPER_DEFAULT_HL,
        "num": SERPER_SEARCH_LIMIT,
        "autocorrect": False,
    }

    try:
        response = await _serper_post(f"{SERPER_BASE_URL}/search", payload)
        data = response.json() or {}
    except httpx.HTTPStatusError as e:
        status = e.response.status_code if e.response is not None else "unknown"
        body = ""
        try:
            body = e.response.text[:200] if e.response is not None else ""
        except Exception:
            pass
        msg = (
            f"Serper rejected the search (HTTP {status}): {body}. "
            "Reformulate the query or check SERPER_API_KEY."
        )
        return json.dumps(
            {"success": False, "error": msg, "results": []}, ensure_ascii=False
        )
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
        logger.warning(
            "serper search exhausted retries for q=%r (%s)",
            (q or "")[:200],
            type(e).__name__,
        )
        msg = (
            "Serper search is temporarily unreachable after retries. "
            "Please try a different search tool or scrape a known-good URL."
        )
        return json.dumps(
            {"success": False, "error": msg, "results": []}, ensure_ascii=False
        )
    except Exception as e:
        logger.exception("serper search unexpected failure")
        msg = (
            f"Internal error in Serper search backend "
            f"({type(e).__name__}): {str(e)[:200]}"
        )
        return json.dumps(
            {"success": False, "error": msg, "results": []}, ensure_ascii=False
        )

    # Normalise organic results into the shared schema, dropping HF traps.
    organic_out: List[Dict[str, str]] = []
    for item in data.get("organic") or []:
        link = (item.get("link") or "").strip()
        if not link or _is_huggingface_dataset_or_space_url(link):
            continue
        organic_out.append(
            {
                "title": item.get("title") or "",
                "link": link,
                "snippet": item.get("snippet") or "",
                "media": item.get("source") or item.get("displayLink") or "",
                "date": item.get("date") or "",
            }
        )

    output: Dict[str, Any] = {
        "searchParameters": {
            "q": q.strip(),
            "engine": "serper",
            "gl": SERPER_DEFAULT_GL,
            "hl": SERPER_DEFAULT_HL,
            "num": SERPER_SEARCH_LIMIT,
        },
        "organic": organic_out,
    }
    # Pass through rich blocks when present.
    for k in ("answerBox", "knowledgeGraph", "peopleAlsoAsk", "relatedSearches"):
        if k in data and data[k]:
            output[k] = data[k]

    return json.dumps(decode_http_urls_in_dict(output), ensure_ascii=False)


# ── Scrape: Serper (primary, async) ──────────────────────────────────
async def _scrape_with_serper(url: str) -> Dict[str, Any]:
    """Scrape a page with the Serper scrape API. Returns a result dict with
    keys: success, content (capped markdown), error."""
    retry_delays = [1, 2, 4, 8]
    last_err = ""
    client = _get_http_client()
    for attempt, delay in enumerate(retry_delays, 1):
        try:
            resp = await client.post(
                SERPER_SCRAPE_URL,
                json={"url": url},
                headers=_serper_headers(),
                timeout=httpx.Timeout(None, connect=20, read=60),
            )
            if _is_retryable_status(resp.status_code) and attempt < len(retry_delays):
                last_err = f"HTTP {resp.status_code}"
                await asyncio.sleep(delay)
                continue
            resp.raise_for_status()
            data = resp.json() or {}
            # Prefer markdown (LLM-friendly); fall back to plain text.
            markdown = data.get("markdown") or data.get("text") or ""
            meta = data.get("metadata") or {}
            title = meta.get("title", "") or ""
            description = meta.get("description", "") or ""

            if not markdown:
                return {"success": False, "content": "", "error": "No content from Serper scrape"}

            parts: List[str] = []
            if title:
                parts.append(f"Title: {title}")
            if description:
                parts.append(f"Description: {description}")
            parts.append(f"URL: {url}")
            parts.append("")
            parts.append(strip_markdown_links(markdown))
            raw = "\n".join(parts)
            # Content cap (mirror dev max_chars).
            raw = raw[:MAX_SCRAPE_CHARS]
            return {"success": True, "content": raw, "error": ""}

        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < len(retry_delays):
                await asyncio.sleep(delay)
                continue
            return {"success": False, "content": "", "error": last_err}
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else 0
            last_err = f"HTTP {status}"
            if _is_retryable_status(status) and attempt < len(retry_delays):
                await asyncio.sleep(delay)
                continue
            return {"success": False, "content": "", "error": last_err}
        except Exception as e:
            return {"success": False, "content": "", "error": f"Unexpected: {e}"}

    return {"success": False, "content": "", "error": last_err or "serper retry exhausted"}


# ── Scrape: Python httpx fallback (async, no sandbox) ────────────────
async def _scrape_with_python(url: str) -> Dict[str, Any]:
    """Fallback scrape using a plain async ``httpx`` GET with a browser
    User-Agent. No sandbox / no extra API keys."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
    }
    retry_delays = [1, 2, 4]
    last_err = ""
    client = _get_http_client()
    for attempt, delay in enumerate(retry_delays, 1):
        try:
            resp = await client.get(
                url,
                headers=headers,
                timeout=httpx.Timeout(None, connect=20, read=60),
            )
            if _is_retryable_status(resp.status_code) and attempt < len(retry_delays):
                last_err = f"HTTP {resp.status_code}"
                await asyncio.sleep(delay)
                continue
            resp.raise_for_status()
            content = resp.text
            if not content:
                return {"success": False, "content": "", "error": "No content from URL"}
            # Raw HTML → plain text (the fallback returns HTML, not markdown).
            content = _strip_html_tags(content)
            body = strip_markdown_links(content)
            raw = f"URL: {url}\n\n{body}"[:MAX_SCRAPE_CHARS]
            return {"success": True, "content": raw, "error": ""}
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < len(retry_delays):
                await asyncio.sleep(delay)
                continue
            return {"success": False, "content": "", "error": last_err}
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else 0
            last_err = f"HTTP {status}"
            if _is_retryable_status(status) and attempt < len(retry_delays):
                await asyncio.sleep(delay)
                continue
            return {"success": False, "content": "", "error": last_err}
        except Exception as e:
            return {"success": False, "content": "", "error": f"Unexpected: {e}"}

    return {"success": False, "content": "", "error": last_err or "python retry exhausted"}


# ── LLM Information Extraction (async, multi-provider, repeat-guarded) ─
EXTRACT_INFO_PROMPT = """You are given a piece of content and the requirement of information to extract. Your task is to extract the information specifically requested. Be precise and focus exclusively on the requested information.

INFORMATION TO EXTRACT:
{}

INSTRUCTIONS:
1. Extract the information relevant to the focus above.
2. If the exact information is not found, extract the most closely related details.
3. Be specific and include exact details when available.
4. Clearly organize the extracted information for easy understanding.
5. Do not include general summaries or unrelated content.

CONTENT TO ANALYZE:
{}

EXTRACTED INFORMATION:"""


def _build_summary_request(prompt: str, max_tokens: int):
    """Build (url, headers, payload) for the configured summary provider."""
    model = SUMMARY_MODEL_LABEL
    is_gemini = SUMMARY_LLM_PROVIDER == "gemini"
    is_azure = SUMMARY_LLM_PROVIDER == "azure"
    is_maas = SUMMARY_LLM_PROVIDER == "maas"

    if is_gemini:
        base = SUMMARY_LLM_BASE_URL.rstrip("/")
        url = f"{base}/google/v1:generateContent"
        headers = {"Content-Type": "application/json", "api-key": SUMMARY_LLM_API_KEY or ""}
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 1.0, "maxOutputTokens": max_tokens},
        }
    elif is_azure:
        base = SUMMARY_LLM_BASE_URL.rstrip("/")
        api_version = SUMMARY_LLM_API_VERSION or "2024-12-01-preview"
        url = (
            f"{base}/deployments/{model}/chat/completions"
            f"?api-version={api_version}"
        )
        headers = {"Content-Type": "application/json", "api-key": SUMMARY_LLM_API_KEY or ""}
        payload = {
            "max_completion_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
    elif is_maas:
        url = SUMMARY_LLM_BASE_URL
        headers = {"Content-Type": "application/json", "api-key": SUMMARY_LLM_API_KEY or ""}
        payload = {
            "model": SUMMARY_LLM_MODEL_NAME or "",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    else:
        url = SUMMARY_LLM_BASE_URL
        headers = {"Content-Type": "application/json"}
        if SUMMARY_LLM_API_KEY:
            headers["Authorization"] = f"Bearer {SUMMARY_LLM_API_KEY}"
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.0,
        }
    return url, headers, payload


def _set_prompt_in_payload(payload: Dict[str, Any], prompt: str) -> None:
    if SUMMARY_LLM_PROVIDER == "gemini":
        payload["contents"][0]["parts"][0]["text"] = prompt
    else:
        payload["messages"][0]["content"] = prompt


def _extract_text_from_response(data: Dict[str, Any], is_gemini: bool) -> tuple:
    """Parse the LLM response JSON and extract the generated text.

    Returns ``(extracted, tokens, error)``.  On success ``error`` is empty;
    on failure ``extracted`` is empty.
    """
    if is_gemini:
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text_parts = [
                p["text"] for p in parts if "text" in p and not p.get("thought")
            ]
            extracted = "\n".join(text_parts)
            usage = data.get("usageMetadata", {})
            tokens = usage.get("promptTokenCount", 0) + usage.get(
                "candidatesTokenCount", 0
            )
            return extracted, tokens, ""
        if "error" in data:
            return "", 0, f"Gemini API error: {data['error']}"
        return "", 0, f"Unexpected Gemini response: {data}"

    # OpenAI / Azure / MaaS compatible
    if "choices" in data and len(data["choices"]) > 0:
        msg = data["choices"][0].get("message", {}) or {}
        extracted = msg.get("content")
        if not extracted:
            extracted = msg.get("reasoning_content") or ""
        tokens = data.get("usage", {}).get("total_tokens", 0)
        return extracted, tokens, ""
    if "error" in data:
        return "", 0, f"LLM API error: {data['error']}"
    return "", 0, f"Unexpected LLM response: {data}"


def _is_context_overflow(resp_text: str) -> bool:
    """Detect common LLM 'context too long' error patterns."""
    low = resp_text.lower()
    return (
        "maximum context length" in low
        or "context length" in low
        or "token count exceeds" in low
        or "exceeds the maximum" in low
        or ("too long" in low and "input" in low)
    )


async def _extract_info_with_llm(
    content: str, info_to_extract: str, max_tokens: int = 8192
) -> Dict[str, Any]:
    """Extract targeted info from scraped content using the summary LLM.

    Async, multi-provider (gemini/azure/maas/openai), with:
      * context-overflow auto-truncation + retry,
      * repeat-output detection on EXTRACTED TEXT (not raw HTTP body).
    """
    model = SUMMARY_MODEL_LABEL
    is_gemini = SUMMARY_LLM_PROVIDER == "gemini"
    _fail = lambda err: {  # noqa: E731
        "success": False,
        "extracted_info": "",
        "error": err,
        "model_used": model,
        "tokens_used": 0,
    }

    if not SUMMARY_LLM_BASE_URL:
        return _fail("SUMMARY_LLM_BASE_URL not configured")
    if not content or not content.strip():
        return _fail("Content is empty")

    prompt = EXTRACT_INFO_PROMPT.format(info_to_extract, content)
    url, headers, payload = _build_summary_request(prompt, max_tokens)

    retry_delays = [1, 2, 4, 8]
    client = _get_http_client()
    for attempt, delay in enumerate(retry_delays, 1):
        try:
            resp = await client.post(
                url,
                json=payload,
                headers=headers,
                timeout=httpx.Timeout(None, connect=30, read=300),
            )

            # Context-overflow auto-truncation (check BEFORE raise_for_status
            # because some providers return 200 with an error body).
            if _is_context_overflow(resp.text or ""):
                truncate_chars = 40960 * attempt
                if len(content) > truncate_chars:
                    truncated = content[:-truncate_chars] + "\n[...truncated]"
                    prompt = EXTRACT_INFO_PROMPT.format(info_to_extract, truncated)
                    _set_prompt_in_payload(payload, prompt)
                    logger.warning(
                        "Scrape summary: context overflow, truncating %d chars "
                        "(attempt %d)",
                        truncate_chars,
                        attempt,
                    )
                    continue
                # Content too short to truncate further — give up.
                return _fail(
                    f"Context overflow but content too short to truncate further "
                    f"(len={len(content)}, would need to cut {truncate_chars})"
                )

            resp.raise_for_status()
            data = resp.json()

            # Parse response & extract text.
            extracted, tokens, parse_err = _extract_text_from_response(data, is_gemini)
            if parse_err:
                return _fail(parse_err)

            # Repeat-output detection on EXTRACTED TEXT (not raw JSON).
            if extracted and len(extracted) >= 50:
                tail_50 = extracted[-50:]
                if extracted.count(tail_50) > 5 and attempt < len(retry_delays):
                    logger.warning(
                        "Scrape summary: repeat detected in extracted text "
                        "(attempt %d), retrying",
                        attempt,
                    )
                    await asyncio.sleep(delay)
                    continue

            return {
                "success": True,
                "extracted_info": extracted,
                "error": "",
                "model_used": model,
                "tokens_used": tokens,
            }

        except (httpx.ConnectTimeout, httpx.ConnectError) as e:
            if attempt < len(retry_delays):
                logger.warning(
                    "Scrape summary: %s (attempt %d), retrying in %ds",
                    type(e).__name__,
                    attempt,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            return _fail(f"LLM API connection error after all retries: {e}")
        except httpx.ReadTimeout:
            if attempt < len(retry_delays):
                logger.warning(
                    "Scrape summary: read timeout (attempt %d)", attempt
                )
                continue
            return _fail("LLM API timeout after all retries")
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else 0
            if _is_retryable_status(status) and attempt < len(retry_delays):
                logger.warning(
                    "Scrape summary: HTTP %s (retryable), retry in %ds", status, delay
                )
                await asyncio.sleep(delay)
                continue
            return _fail(f"LLM API HTTP {status}")
        except Exception as e:
            logger.error("Scrape summary: unexpected error: %s", e)
            return _fail(f"Unexpected error: {str(e)}")

    return _fail("Max retry attempts reached")


# ── Scrape Tool (FORCED summary) ─────────────────────────────────────
_DEFAULT_INFO_TO_EXTRACT = (
    "the main content, key facts, figures, dates, names and conclusions of this page"
)


@mcp.tool()
async def scrape_website(
    url: Annotated[str, Field(description="The URL to scrape. Must start with http:// or https://.")],
    info_to_extract: Annotated[str, Field(default="", description="The specific information to extract (a question or topic focus). Strongly recommended; if omitted, a generic focus is used so a summary is still produced.")],
) -> str:
    """Scrape a website and return LLM-extracted information (summary FORCED).

    The page is fetched via the Serper scrape API; if that fails it falls back
    to a plain Python httpx GET. The scraped content is then ALWAYS passed to a
    summary LLM that extracts the information requested in ``info_to_extract``.
    There is no raw-content passthrough — this tool always returns
    LLM-extracted information.

    Args:
        url: The URL to scrape. Must start with http:// or https://.
        info_to_extract: The specific information to extract (a question or
            topic focus). Strongly recommended; if omitted, a generic focus is
            used so a summary is still produced.

    Returns:
        JSON string: {success, url, extracted_info, model_used, tokens_used}.
    """
    if not url or not url.startswith(("http://", "https://")):
        return json.dumps(
            {
                "success": False,
                "url": url,
                "extracted_info": "",
                "error": f"Invalid URL: '{url}'. Must start with http:// or https://",
            },
            ensure_ascii=False,
        )

    if "huggingface.co/datasets" in url or "huggingface.co/spaces" in url:
        return json.dumps(
            {
                "success": False,
                "url": url,
                "extracted_info": "",
                "error": (
                    "You are trying to scrape a Hugging Face dataset for answers, "
                    "please do not use the scrape tool for this purpose."
                ),
            },
            ensure_ascii=False,
        )

    if not SERPER_API_KEY:
        return json.dumps(
            {
                "success": False,
                "url": url,
                "extracted_info": "",
                "error": "SERPER_API_KEY is not set, scrape_website tool is not available.",
            },
            ensure_ascii=False,
        )

    # 1) Primary scrape via Serper; 2) Python httpx fallback.
    scrape_result = await _scrape_with_serper(url)
    if not scrape_result["success"]:
        logger.warning(
            "scrape_website: Serper scrape failed (%s), trying Python httpx fallback",
            scrape_result["error"],
        )
        scrape_result = await _scrape_with_python(url)
        if not scrape_result["success"]:
            return json.dumps(
                {
                    "success": False,
                    "url": url,
                    "extracted_info": "",
                    "error": f"Scraping failed (both Serper and Python): {scrape_result['error']}",
                },
                ensure_ascii=False,
            )
        logger.info("scrape_website: Python fallback succeeded for %s", url)

    raw_content = scrape_result["content"]

    # FORCED summary — always run LLM extraction.
    focus = info_to_extract.strip() if info_to_extract and info_to_extract.strip() else _DEFAULT_INFO_TO_EXTRACT
    llm_result = await _extract_info_with_llm(raw_content, focus)

    if llm_result["success"]:
        return json.dumps(
            {
                "success": True,
                "url": url,
                "extracted_info": llm_result["extracted_info"],
                "model_used": llm_result["model_used"],
                "tokens_used": llm_result["tokens_used"],
            },
            ensure_ascii=False,
        )

    # Resilience: if the summary LLM ultimately fails, do not leave the agent
    # empty-handed — return the (capped) raw content with the error noted.
    logger.warning(
        "scrape_website: summary failed (%s), returning raw content as fallback",
        llm_result["error"],
    )
    return json.dumps(
        {
            "success": False,
            "url": url,
            "extracted_info": raw_content,
            "model_used": llm_result["model_used"],
            "tokens_used": llm_result["tokens_used"],
            "error": f"Summary failed, returning raw content: {llm_result['error']}",
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    mcp.run()
