# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""MCP server backed by **Jina Reader** for webpage scraping with forced LLM summary.

Scrape-only server, designed to be paired with another backend's ``web_search``
(typically ``tool-serper-search``) while that backend's ``scrape_website`` is
blacklisted in the agent yaml. The tool name, signature and JSON output contract
are **identical** to ``searching_serper_mcp_server.scrape_website``, so this is a
drop-in replacement:

* ``scrape_website`` - Jina Reader (primary) + Python httpx (fallback)
                       → **forced** LLM extraction (no raw-content passthrough).

Why a dedicated server rather than a scrape option on the Serper one: keeping
each backend in its own process means an agent config can mix them freely (Serper
SERP + Jina scrape) by blacklisting the tool it does not want, with no shared
state and no flag threading. This server is fully async (``httpx.AsyncClient``)
and reuses the retry / summary pipeline from ``searching_serper_mcp_server``.

Network isolation (important in restricted networks)
----------------------------------------------------
``*.jina.ai`` is DNS-blocked in some internal networks, so an HTTP proxy may be
required. To avoid perturbing anything else, the proxy is applied **only** to the
outbound web fetches (Jina Reader + the python fallback) via ``JINA_PROXY``:

  * web client  -> ``proxy=JINA_PROXY``, ``trust_env=False``
  * LLM  client -> **no proxy**,        ``trust_env=False``

``trust_env=False`` on both means ambient ``HTTP(S)_PROXY`` variables are ignored,
so the summary-LLM call (usually an *internal* endpoint) is never routed through
the proxy, and no other tool/server is affected.

Environment variables
---------------------
JINA_API_KEY          (required) Jina API key (``Bearer`` token).
JINA_BASE_URL         (optional) Reader base, default ``https://r.jina.ai``.
JINA_PROXY            (optional) HTTP(S) proxy for Jina + python fallback only,
                       e.g. ``http://proxy.example.com:3128``. Empty = direct.
JINA_ENGINE           (optional, default ``direct``) primary ``X-Engine``.
                       NOTE: ``browser`` (the value in Jina's doc example) is a
                       BAD default for our workload — on arxiv/ar5iv it returns
                       the caption of a lazily-rendered SVG instead of the paper
                       (328 B vs 80 KB). ``direct`` is also ~4x faster and ties
                       on Wikipedia. Empty string omits the header.
JINA_FALLBACK_ENGINE  (optional, default ``browser``) engine used only to
                       escalate when the primary returns a thin body (real JS
                       SPAs). Set equal to JINA_ENGINE to disable escalation.
JINA_THIN_CONTENT_CHARS (optional, int, default 1200) below this many chars of
                       markdown body, escalate to the next ladder rung.
JINA_MIN_USABLE_CHARS (optional, int, default 200) below this — after every
                       escalation — the page is declared unusable and the summary
                       LLM is skipped entirely.
JINA_RETAIN_IMAGES    (optional, default ``none``) ``X-Retain-Images``.
JINA_WITH_IMAGES_SUMMARY (optional, default ``false``) send
                       ``X-With-Images-Summary: true`` when truthy. Off by
                       default: this server always feeds the page to a summary
                       LLM, and an image inventory only adds input tokens.
JINA_NO_CACHE         (optional, default ``false``) force ``X-No-Cache: true`` on
                       EVERY request. Off by default because the ladder already
                       retries with no-cache when a body looks thin, which keeps
                       the fast cached path for the ~90% of pages that are fine.
JINA_CONNECT_TIMEOUT  (optional, int, default 20) connect timeout (s).
JINA_READ_TIMEOUT     (optional, int, default 90) read timeout (s); the browser
                       engine can be slow on heavy pages.
JINA_PYTHON_FALLBACK  (optional, default ``true``) enable plain-httpx fallback.
MAX_SCRAPE_CHARS      (optional, int, default 409600) max chars before summary.

Summary LLM (identical semantics to searching_serper_mcp_server):
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
from typing import Annotated, Any, Dict, Optional

from pydantic import Field

import httpx
from mcp.server.fastmcp import FastMCP

from .utils import strip_markdown_links


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


# ── Configuration ─────────────────────────────────────────────────────
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_BASE_URL = os.getenv("JINA_BASE_URL", "https://r.jina.ai").rstrip("/")
JINA_PROXY = os.getenv("JINA_PROXY", "").strip()

# Engine defaults are MEASURED, not copied from the Jina docs example.
# On academic pages the doc's suggested ``X-Engine: browser`` returns the caption
# of a lazily-rendered SVG figure instead of the document:
#   arxiv.org/html/2509.13313v3   browser -> 328 B ("websailor30b_tool_call.svg")
#                                 direct  -> 80,101 B (the actual paper)
# On Wikipedia both engines are byte-identical, and ``direct`` is ~4x faster.
# So ``direct`` is primary and ``browser`` is the escalation for pages that
# genuinely need JS rendering. See _scrape_with_jina for the ladder.
JINA_ENGINE = os.getenv("JINA_ENGINE", "direct").strip()
JINA_FALLBACK_ENGINE = os.getenv("JINA_FALLBACK_ENGINE", "browser").strip()
JINA_RETAIN_IMAGES = os.getenv("JINA_RETAIN_IMAGES", "none").strip()
JINA_WITH_IMAGES_SUMMARY = _env_flag("JINA_WITH_IMAGES_SUMMARY", "false")
# Jina's cache can serve a POISONED entry: the arxiv URL above returns 423 B with
# "Warning: This is a cached snapshot..." even on the direct engine, while the
# same request with X-No-Cache returns the full 80 KB page. Rather than paying
# ~+1s on every scrape by disabling the cache globally, the cached read is tried
# first and a no-cache read is used only as escalation when the body looks thin.
JINA_NO_CACHE = _env_flag("JINA_NO_CACHE", "false")
# Markdown body (Jina's header stripped) shorter than this means "try harder":
# escalate to no-cache, then to the fallback engine.
JINA_THIN_CONTENT_CHARS = int(os.getenv("JINA_THIN_CONTENT_CHARS", "1200"))
# Only below THIS length (after every escalation) is a page declared unusable and
# the summary LLM skipped. Deliberately far smaller than the escalation
# threshold: legitimately short pages must still be summarized.
JINA_MIN_USABLE_CHARS = int(os.getenv("JINA_MIN_USABLE_CHARS", "200"))

JINA_CONNECT_TIMEOUT = int(os.getenv("JINA_CONNECT_TIMEOUT", "20"))
JINA_READ_TIMEOUT = int(os.getenv("JINA_READ_TIMEOUT", "90"))
JINA_PYTHON_FALLBACK = _env_flag("JINA_PYTHON_FALLBACK", "true")

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
    """Human-friendly model label for trace logs (``model_used`` field)."""
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

logger = logging.getLogger("searching-jina-mcp-server")
mcp = FastMCP("searching-jina-mcp-server")


# ── HTTP clients ─────────────────────────────────────────────────────
# Two separate pools so the proxy applies ONLY to outbound web traffic:
#   _WEB_CLIENT -> Jina Reader + python fallback (proxied when JINA_PROXY set)
#   _LLM_CLIENT -> summary LLM (never proxied; usually an internal endpoint)
# trust_env=False on both => ambient HTTP(S)_PROXY is ignored, so behaviour is
# fully explicit and nothing else in the process/environment is affected.
_WEB_CLIENT: Optional[httpx.AsyncClient] = None
_LLM_CLIENT: Optional[httpx.AsyncClient] = None

_LIMITS = httpx.Limits(max_connections=50, max_keepalive_connections=20)


def _get_web_client() -> httpx.AsyncClient:
    global _WEB_CLIENT
    if _WEB_CLIENT is None or _WEB_CLIENT.is_closed:
        kwargs: Dict[str, Any] = {
            "follow_redirects": True,
            "limits": _LIMITS,
            "trust_env": False,
        }
        if JINA_PROXY:
            kwargs["proxy"] = JINA_PROXY
        _WEB_CLIENT = httpx.AsyncClient(**kwargs)
    return _WEB_CLIENT


def _get_llm_client() -> httpx.AsyncClient:
    global _LLM_CLIENT
    if _LLM_CLIENT is None or _LLM_CLIENT.is_closed:
        _LLM_CLIENT = httpx.AsyncClient(
            follow_redirects=True, limits=_LIMITS, trust_env=False
        )
    return _LLM_CLIENT


# ── HTML → plain-text helper (no external deps) ─────────────────────
_RE_SCRIPT_STYLE = re.compile(
    r"<\s*(script|style)[^>]*>.*?</\s*\1\s*>", re.DOTALL | re.IGNORECASE
)
_RE_TAG = re.compile(r"<[^>]+>")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
_RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def _strip_html_tags(html: str) -> str:
    """Lightweight HTML → text conversion using regex only (no extra deps)."""
    text = _RE_SCRIPT_STYLE.sub("", html)
    text = _RE_TAG.sub("", text)
    text = _RE_MULTI_SPACE.sub(" ", text)
    text = _RE_MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


# ── Async retry helpers ──────────────────────────────────────────────
_RETRYABLE_STATUS = {408, 409, 425, 429}


def _is_retryable_status(status_code: int) -> bool:
    return status_code >= 500 or status_code in _RETRYABLE_STATUS


def _is_huggingface_dataset_or_space_url(url: str) -> bool:
    """Filter out HF dataset / space URLs (prevents benchmark answer-leakage)."""
    if not url:
        return False
    return "huggingface.co/datasets" in url or "huggingface.co/spaces" in url


def _jina_headers(engine: str = "", no_cache: bool = False) -> Dict[str, str]:
    """Build Jina Reader request headers for one attempt of the engine ladder."""
    headers = {"Authorization": f"Bearer {JINA_API_KEY}"}
    if engine:
        headers["X-Engine"] = engine
    if JINA_RETAIN_IMAGES:
        headers["X-Retain-Images"] = JINA_RETAIN_IMAGES
    if JINA_WITH_IMAGES_SUMMARY:
        headers["X-With-Images-Summary"] = "true"
    if no_cache or JINA_NO_CACHE:
        headers["X-No-Cache"] = "true"
    return headers


# Jina prefixes its markdown with a small header block; those lines are not page
# content, so they must not count toward the "is this page thin?" decision.
_RE_JINA_HEADER = re.compile(
    r"^(?:Title|URL Source|Published Time|Warning|Image \d+):.*$", re.MULTILINE
)
_CACHED_SNAPSHOT_MARKER = "cached snapshot of the original page"

# Interstitials that return HTTP 200 with no real content. Matching these lets us
# skip a pointless summary-LLM call and tell the model to try another source.
_BOT_WALL_PATTERNS = (
    "verifying your browser",
    "robot challenge screen",
    "checking the site connection security",
    "just a moment...",
    "enable javascript and cookies to continue",
    "attention required! | cloudflare",
    "captcha",
    "access denied",
    "are you a robot",
    "unusual traffic",
)


def _markdown_body(raw: str) -> str:
    """Return only the page content from a Jina Reader response.

    Strips the ``Title:``/``URL Source:``/``Warning:`` header block and the
    ``Markdown Content:`` separator so length checks measure real content.
    """
    if not raw:
        return ""
    body = raw.split("Markdown Content:", 1)[-1]
    body = _RE_JINA_HEADER.sub("", body)
    return body.strip()


def _content_len(raw: str) -> int:
    """Length of what the model will ACTUALLY receive from this response.

    Must match the final transform (strip links, then drop Jina's header),
    otherwise link-heavy pages lie: a Zhihu nav shell measures 516 chars raw but
    collapses to 246 once markdown links are stripped. Every decision — rung
    comparison, thinness, usability — uses this one metric so they cannot drift.
    """
    if not raw:
        return 0
    return len(_markdown_body(strip_markdown_links(raw)))


def _is_thin(raw: str) -> bool:
    """True if this response warrants escalating to a harder fetch mode."""
    return _content_len(raw) < JINA_THIN_CONTENT_CHARS


def _bot_wall_reason(raw: str) -> str:
    """Return the matched interstitial pattern, or '' if the page looks real."""
    head = raw[:2000].lower()
    for pat in _BOT_WALL_PATTERNS:
        if pat in head:
            return pat
    return ""


def _strip_reader_prefix(url: str) -> str:
    """Undo an accidental double reader prefix (model sometimes pastes it).

    Handles both the configured ``JINA_BASE_URL`` and the canonical
    ``https://r.jina.ai`` (the historical hardcoded value).
    """
    for base in {JINA_BASE_URL, "https://r.jina.ai"}:
        prefix = base.rstrip("/") + "/"
        if url.startswith(prefix) and url.count("http") >= 2:
            return url[len(prefix) :]
    return url


def _detect_jina_body_error(body: str) -> str:
    """Return a human-readable error if the body is a Jina error envelope.

    Jina returns HTTP 200 with a JSON error body in some cases (notably
    ``InsufficientBalanceError``). Empty string => not an error body.
    """
    text = (body or "").lstrip()
    if not text.startswith("{"):
        return ""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return ""
    if not isinstance(data, dict):
        return ""
    name = str(data.get("name") or "")
    if name == "InsufficientBalanceError":
        return "Jina: insufficient balance (top up the JINA_API_KEY account)"
    if name or data.get("error"):
        detail = str(data.get("message") or data.get("error") or name)[:200]
        return f"Jina error: {detail}"
    return ""


# ── Scrape: Jina Reader (primary, async) ─────────────────────────────
async def _jina_fetch(url: str, engine: str, no_cache: bool) -> Dict[str, Any]:
    """One rung of the engine ladder: fetch ``url`` with the given mode.

    Retries only TRANSPORT failures (timeouts, 5xx, 429). Content quality is the
    caller's concern. Returns {success, raw (uncapped), error}.
    """
    reader_url = f"{JINA_BASE_URL}/{url}"
    retry_delays = [1, 2, 4, 8]
    last_err = ""
    client = _get_web_client()
    for attempt, delay in enumerate(retry_delays, 1):
        try:
            resp = await client.get(
                reader_url,
                headers=_jina_headers(engine=engine, no_cache=no_cache),
                timeout=httpx.Timeout(
                    None, connect=JINA_CONNECT_TIMEOUT, read=JINA_READ_TIMEOUT
                ),
            )
            if _is_retryable_status(resp.status_code) and attempt < len(retry_delays):
                last_err = f"HTTP {resp.status_code}"
                logger.info("Jina: %s (retryable), retry in %ss", last_err, delay)
                await asyncio.sleep(delay)
                continue
            resp.raise_for_status()

            content = resp.text or ""
            body_err = _detect_jina_body_error(content)
            if body_err:
                # Balance / auth style errors are not worth retrying.
                return {"success": False, "raw": "", "error": body_err}
            if not content.strip():
                last_err = "Jina returned empty content"
                if attempt < len(retry_delays):
                    await asyncio.sleep(delay)
                    continue
                return {"success": False, "raw": "", "error": last_err}

            # Uncapped and unstripped: the caller compares rungs by body length
            # and only then cleans + caps the winner.
            return {"success": True, "raw": content, "error": ""}

        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < len(retry_delays):
                logger.info("Jina: %s, retry in %ss", type(e).__name__, delay)
                await asyncio.sleep(delay)
                continue
            return {"success": False, "raw": "", "error": last_err}
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else 0
            detail = ""
            try:
                detail = (e.response.text or "")[:200] if e.response is not None else ""
            except Exception:
                pass
            last_err = f"HTTP {status}{(': ' + detail) if detail else ''}"
            if _is_retryable_status(status) and attempt < len(retry_delays):
                await asyncio.sleep(delay)
                continue
            return {"success": False, "raw": "", "error": last_err}
        except Exception as e:
            return {"success": False, "raw": "", "error": f"Unexpected: {e}"}

    return {"success": False, "raw": "", "error": last_err or "jina retry exhausted"}


async def _scrape_with_jina(url: str) -> Dict[str, Any]:
    """Scrape a page with Jina Reader, escalating until the content looks real.

    Ladder (each rung only runs if the previous one came back thin):
      1. primary engine  + cache      — fast path, serves most pages
      2. primary engine  + X-No-Cache — defeats poisoned cache entries
      3. fallback engine + X-No-Cache — for pages that truly need JS rendering

    The longest body wins, so escalation can never make the result worse.
    Returns {success, content (capped), error, engine_used, unusable, reason}.
    """
    rungs = [(JINA_ENGINE, False), (JINA_ENGINE, True)]
    if JINA_FALLBACK_ENGINE and JINA_FALLBACK_ENGINE != JINA_ENGINE:
        rungs.append((JINA_FALLBACK_ENGINE, True))

    best_raw = ""
    best_label = ""
    best_len = -1
    last_err = ""
    for engine, no_cache in rungs:
        label = f"{engine or 'auto'}{'+nocache' if no_cache else ''}"
        res = await _jina_fetch(url, engine=engine, no_cache=no_cache)
        if not res["success"]:
            last_err = res["error"]
            # A hard transport/auth/balance error will not be fixed by another
            # rung of the same ladder.
            if "insufficient balance" in last_err.lower():
                break
            continue

        raw = res["raw"]
        this_len = _content_len(raw)
        if this_len > best_len:
            best_raw, best_label, best_len = raw, label, this_len

        if this_len >= JINA_THIN_CONTENT_CHARS:
            break
        logger.info(
            "Jina: thin body (%d chars) via %s for %s, escalating",
            this_len,
            label,
            url,
        )

    if not best_raw:
        return {
            "success": False,
            "content": "",
            "error": last_err or "jina returned no usable response",
            "engine_used": "",
            "bot_wall": "",
            "thin": True,
            "content_chars": 0,
        }

    # Content-quality verdict, judged on the post-strip length the model will see.
    #
    # Two DIFFERENT outcomes, deliberately not conflated:
    #   bot_wall — the body is an interstitial, i.e. definitively NOT the page.
    #              Nothing can be salvaged; the caller should say so and move on.
    #   thin     — a real but very short page (example.com is ~113 chars). It may
    #              still hold the answer, so it must never be discarded. The
    #              caller returns it verbatim instead of paying for a summary.
    wall = _bot_wall_reason(best_raw)
    cleaned = strip_markdown_links(best_raw)
    capped = f"URL: {url}\n\n{cleaned}"[:MAX_SCRAPE_CHARS]
    return {
        "success": True,
        "content": capped,
        "error": "",
        "engine_used": best_label,
        "bot_wall": f"bot-verification/interstitial page ({wall})" if wall else "",
        "thin": best_len < JINA_MIN_USABLE_CHARS,
        "content_chars": best_len,
    }


# ── Scrape: Python httpx fallback (async) ────────────────────────────
async def _scrape_with_python(url: str) -> Dict[str, Any]:
    """Fallback scrape using a plain async ``httpx`` GET with a browser UA.

    Uses the same (optionally proxied) web client as Jina, since a direct
    connection to the open internet may be blocked in restricted networks.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
    }
    retry_delays = [1, 2, 4]
    last_err = ""
    client = _get_web_client()
    for attempt, delay in enumerate(retry_delays, 1):
        try:
            resp = await client.get(
                url,
                headers=headers,
                timeout=httpx.Timeout(
                    None, connect=JINA_CONNECT_TIMEOUT, read=JINA_READ_TIMEOUT
                ),
            )
            if _is_retryable_status(resp.status_code) and attempt < len(retry_delays):
                last_err = f"HTTP {resp.status_code}"
                await asyncio.sleep(delay)
                continue
            resp.raise_for_status()
            content = resp.text
            if not content:
                return {"success": False, "content": "", "error": "No content from URL"}
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

    Returns ``(extracted, tokens, error)``.
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
    """Extract targeted info from scraped content using the summary LLM."""
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
    client = _get_llm_client()
    for attempt, delay in enumerate(retry_delays, 1):
        try:
            resp = await client.post(
                url,
                json=payload,
                headers=headers,
                timeout=httpx.Timeout(None, connect=30, read=300),
            )

            # Context-overflow auto-truncation (some providers return 200 with
            # an error body, so check before raise_for_status).
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
                return _fail(
                    f"Context overflow but content too short to truncate further "
                    f"(len={len(content)}, would need to cut {truncate_chars})"
                )

            resp.raise_for_status()
            data = resp.json()

            extracted, tokens, parse_err = _extract_text_from_response(data, is_gemini)
            if parse_err:
                return _fail(parse_err)

            # Repeat-output detection on EXTRACTED TEXT.
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
                logger.warning("Scrape summary: read timeout (attempt %d)", attempt)
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

    The page is fetched via the Jina Reader API; if that fails it falls back to a
    plain Python httpx GET. The scraped content is then ALWAYS passed to a summary
    LLM that extracts the information requested in ``info_to_extract``. There is
    no raw-content passthrough — this tool always returns LLM-extracted
    information.

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

    if _is_huggingface_dataset_or_space_url(url):
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

    if not JINA_API_KEY:
        return json.dumps(
            {
                "success": False,
                "url": url,
                "extracted_info": "",
                "error": "JINA_API_KEY is not set, scrape_website tool is not available.",
            },
            ensure_ascii=False,
        )

    target = _strip_reader_prefix(url)

    # 1) Primary scrape via Jina Reader; 2) optional Python httpx fallback.
    scrape_result = await _scrape_with_jina(target)
    if not scrape_result["success"]:
        if JINA_PYTHON_FALLBACK:
            logger.warning(
                "scrape_website: Jina scrape failed (%s), trying Python httpx fallback",
                scrape_result["error"],
            )
            jina_err = scrape_result["error"]
            scrape_result = await _scrape_with_python(target)
            if not scrape_result["success"]:
                return json.dumps(
                    {
                        "success": False,
                        "url": url,
                        "extracted_info": "",
                        "error": (
                            f"Scraping failed (both Jina and Python): "
                            f"jina={jina_err}; python={scrape_result['error']}"
                        ),
                    },
                    ensure_ascii=False,
                )
            logger.info("scrape_website: Python fallback succeeded for %s", target)
        else:
            return json.dumps(
                {
                    "success": False,
                    "url": url,
                    "extracted_info": "",
                    "error": f"Scraping failed (Jina): {scrape_result['error']}",
                },
                ensure_ascii=False,
            )

    # Jina returned an INTERSTITIAL, not the page. Nothing here can be salvaged,
    # so try the python path (a different UA/IP sometimes gets through) and
    # otherwise say so plainly. Summarizing a "Verifying your browser" page just
    # burns ~400 tokens to be told "cannot be extracted".
    if scrape_result.get("bot_wall"):
        reason = scrape_result["bot_wall"]
        recovered = False
        if JINA_PYTHON_FALLBACK:
            logger.warning(
                "scrape_website: interstitial for %s (%s), trying Python",
                target,
                reason,
            )
            py = await _scrape_with_python(target)
            if py["success"] and not _bot_wall_reason(py["content"]):
                scrape_result = py
                recovered = True
                logger.info("scrape_website: Python recovered %s", target)
        if not recovered:
            return json.dumps(
                {
                    "success": False,
                    "url": url,
                    "extracted_info": "",
                    "engine_used": scrape_result.get("engine_used", ""),
                    "error": (
                        f"The page could not be read: {reason}. No summary was "
                        f"attempted. Try a different source for this information."
                    ),
                },
                ensure_ascii=False,
            )

    raw_content = scrape_result["content"]

    # A real but very short page (e.g. example.com, ~113 chars). Never discard it
    # — it may hold the answer — but there is nothing to compress either, so
    # return it verbatim and skip the summary round-trip entirely.
    if scrape_result.get("thin") and not scrape_result.get("bot_wall"):
        logger.info(
            "scrape_website: %s is short (%s chars), returning verbatim",
            target,
            scrape_result.get("content_chars"),
        )
        return json.dumps(
            {
                "success": True,
                "url": url,
                "extracted_info": raw_content,
                "model_used": "none (page too short to summarize)",
                "tokens_used": 0,
                "engine_used": scrape_result.get("engine_used", ""),
            },
            ensure_ascii=False,
        )

    # FORCED summary — always run LLM extraction.
    focus = (
        info_to_extract.strip()
        if info_to_extract and info_to_extract.strip()
        else _DEFAULT_INFO_TO_EXTRACT
    )
    llm_result = await _extract_info_with_llm(raw_content, focus)

    if llm_result["success"]:
        return json.dumps(
            {
                "success": True,
                "url": url,
                "extracted_info": llm_result["extracted_info"],
                "model_used": llm_result["model_used"],
                "tokens_used": llm_result["tokens_used"],
                # Which ladder rung produced the content. Additive vs serper's
                # contract (readers use .get()); makes runs auditable offline.
                "engine_used": scrape_result.get("engine_used", ""),
            },
            ensure_ascii=False,
        )

    # Resilience: if the summary LLM ultimately fails, return the (capped) raw
    # content with the error noted rather than leaving the agent empty-handed.
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
