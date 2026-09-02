# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Unit tests for searching_jina_mcp_server content-quality handling.

The escalation ladder exists because of two measured Jina failure modes found
while validating the integration on browsecomp_zh:

  1. POISONED CACHE — `arxiv.org/html/2509.13313v3` returns 423 B with
     "Warning: This is a cached snapshot of the original page" even on the
     `direct` engine; the same request with `X-No-Cache: true` returns 80,101 B.
  2. WRONG ENGINE — `X-Engine: browser` (the value in Jina's own doc example)
     returns the caption of a lazily-rendered SVG figure instead of the document
     (328 B vs 80,101 B for `direct`). Wikipedia is byte-identical on both.

Plus a third, backend-independent issue: sites return HTTP 200 with a
bot-verification interstitial, which used to be fed to the summary LLM and cost
~400 tokens to be told "cannot be extracted".
"""

import sys
from pathlib import Path

import pytest

_APP = Path(__file__).resolve().parent.parent
_TOOLS_SRC = _APP / "tools" / "src"
for p in (str(_APP), str(_TOOLS_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from miroflow_tools.mcp_servers import searching_jina_mcp_server as J  # noqa: E402


# ---- real captured responses (trimmed) --------------------------------
POISONED_ARXIV = """Title: websailor30b_tool_call.svg

URL Source: https://arxiv.org/html/2509.13313v3

Published Time: Tue, 11 Aug 2026 20:14:40 GMT

Warning: This is a cached snapshot of the original page, consider retry with caching opt-out.

Markdown Content:
a graph for score after importing of different studied papers
"""

BOT_WALL_OPENREVIEW = """Title: Verifying your browser | OpenReview

URL Source: https://openreview.net/pdf/baa47089.pdf

Markdown Content:
Verifying your browser, please wait...
"""

BOT_WALL_CLOUDFLARE = """Title: Robot Challenge Screen

URL Source: https://aronhack.com/paper-resum/

Markdown Content:
Checking the site connection security
"""

GOOD_PAGE = (
    "Title: Alan Turing - Wikipedia\n\n"
    "URL Source: https://en.wikipedia.org/wiki/Alan_Turing\n\n"
    "Markdown Content:\n" + ("Alan Turing was a British mathematician. " * 200)
)


# ---- _markdown_body ---------------------------------------------------
def test_markdown_body_strips_jina_header():
    body = J._markdown_body(POISONED_ARXIV)
    assert "Title:" not in body
    assert "URL Source:" not in body
    assert "Warning:" not in body
    assert "cached snapshot" not in body
    assert body == "a graph for score after importing of different studied papers"


def test_markdown_body_keeps_real_content():
    body = J._markdown_body(GOOD_PAGE)
    assert body.startswith("Alan Turing was a British mathematician")
    assert len(body) > 5000


def test_markdown_body_handles_empty_and_headerless():
    assert J._markdown_body("") == ""
    assert J._markdown_body(None) == ""
    assert J._markdown_body("just some text") == "just some text"


# ---- _is_thin ---------------------------------------------------------
def test_poisoned_cache_response_is_thin():
    """The whole point: a 423 B poisoned snapshot must trigger escalation."""
    assert J._is_thin(POISONED_ARXIV) is True


def test_real_page_is_not_thin():
    assert J._is_thin(GOOD_PAGE) is False


def test_thinness_ignores_header_length():
    """A long header must not make an empty page look substantial."""
    padded = (
        "Title: " + "x" * 3000 + "\n\nURL Source: https://e.com\n\n"
        "Markdown Content:\ntiny\n"
    )
    assert J._is_thin(padded) is True


# ---- _content_len: the single metric all decisions share --------------
LINK_SHELL = (
    "Title: 知乎\n\nURL Source: https://www.zhihu.com/column/27242893\n\n"
    "Markdown Content:\n"
    + "".join(f"[关注{i}](https://www.zhihu.com/follow/{i})\n" for i in range(20))
)


def test_content_len_measures_post_strip_length():
    """A nav shell of pure links must not look substantial.

    Regression: thinness was measured on the RAW body (516 chars for a Zhihu
    shell) while the model received the link-stripped version (246 chars), so
    junk pages were reported usable.
    """
    raw_body = len(J._markdown_body(LINK_SHELL))
    measured = J._content_len(LINK_SHELL)
    assert measured < raw_body, "links must be stripped before measuring"


def test_content_len_matches_what_the_model_receives():
    """The metric must equal the body of the actual returned payload."""
    from miroflow_tools.mcp_servers.utils import strip_markdown_links

    for raw in (GOOD_PAGE, POISONED_ARXIV, LINK_SHELL):
        delivered = J._markdown_body(strip_markdown_links(raw))
        assert J._content_len(raw) == len(delivered)


def test_content_len_handles_empty():
    assert J._content_len("") == 0
    assert J._content_len(None) == 0


# ---- _bot_wall_reason -------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        (BOT_WALL_OPENREVIEW, "verifying your browser"),
        (BOT_WALL_CLOUDFLARE, "robot challenge screen"),
        (GOOD_PAGE, ""),
        ("", ""),
    ],
)
def test_bot_wall_detection(raw, expected):
    assert J._bot_wall_reason(raw) == expected


def test_bot_wall_detection_is_case_insensitive():
    assert J._bot_wall_reason("Title: JUST A MOMENT...\n") == "just a moment..."


def test_bot_wall_only_inspects_the_head():
    """A page merely *discussing* captchas deep in the body is not a bot wall."""
    raw = GOOD_PAGE + "\n" + ("filler " * 500) + "\ncaptcha research notes\n"
    assert J._bot_wall_reason(raw) == ""


# ---- header construction ---------------------------------------------
def test_headers_use_requested_engine_and_no_cache():
    h = J._jina_headers(engine="direct", no_cache=True)
    assert h["X-Engine"] == "direct"
    assert h["X-No-Cache"] == "true"
    assert h["Authorization"].startswith("Bearer ")


def test_headers_omit_engine_when_blank():
    h = J._jina_headers(engine="", no_cache=False)
    assert "X-Engine" not in h
    assert "X-No-Cache" not in h


def test_images_summary_off_by_default():
    """It only adds input tokens: this server always summarizes anyway."""
    assert J.JINA_WITH_IMAGES_SUMMARY is False
    assert "X-With-Images-Summary" not in J._jina_headers(engine="direct")


# ---- measured defaults ------------------------------------------------
def test_primary_engine_is_direct_not_browser():
    """Regression guard: 'browser' returns 328 B vs 80 KB on arxiv."""
    assert J.JINA_ENGINE == "direct"
    assert J.JINA_FALLBACK_ENGINE == "browser"


def test_escalation_threshold_far_above_unusable_threshold():
    """Escalate eagerly, but only discard a page when it is truly empty."""
    assert J.JINA_THIN_CONTENT_CHARS > J.JINA_MIN_USABLE_CHARS


# ---- bot_wall vs thin must stay distinct ------------------------------
def test_thin_and_bot_wall_are_different_outcomes():
    """A short REAL page must never be treated like an interstitial.

    example.com is ~113 chars of genuine content. Discarding it would lose
    information; a Cloudflare challenge page contains none, so it is discarded.
    Conflating the two either throws away real pages or wastes summary tokens.
    """
    short_real = (
        "Title: Example Domain\n\nURL Source: https://example.com\n\n"
        "Markdown Content:\nThis domain is for use in illustrative examples.\n"
    )
    assert J._bot_wall_reason(short_real) == ""          # not an interstitial
    assert J._content_len(short_real) < J.JINA_MIN_USABLE_CHARS  # but thin

    assert J._bot_wall_reason(BOT_WALL_CLOUDFLARE) != ""  # interstitial



# ---- _strip_reader_prefix --------------------------------------------
def test_double_reader_prefix_is_stripped():
    assert (
        J._strip_reader_prefix("https://r.jina.ai/https://example.com")
        == "https://example.com"
    )


def test_normal_url_is_untouched():
    assert J._strip_reader_prefix("https://example.com") == "https://example.com"


def test_url_merely_containing_the_host_is_untouched():
    u = "https://example.com/page?ref=https://r.jina.ai/x"
    assert J._strip_reader_prefix(u) == u


# ---- jina error envelopes -------------------------------------------
def test_insufficient_balance_is_detected():
    body = '{"name":"InsufficientBalanceError","message":"no tokens left"}'
    assert "insufficient balance" in J._detect_jina_body_error(body).lower()


def test_plain_markdown_is_not_an_error_envelope():
    assert J._detect_jina_body_error(GOOD_PAGE) == ""


def test_json_page_content_is_not_misread_as_error():
    assert J._detect_jina_body_error('{"data": {"title": "hello"}}') == ""


# ---- hf guard --------------------------------------------------------
@pytest.mark.parametrize(
    "url,blocked",
    [
        ("https://huggingface.co/datasets/foo/bar", True),
        ("https://huggingface.co/spaces/foo/bar", True),
        ("https://huggingface.co/meta-llama/Llama-3", False),
        ("https://example.com", False),
    ],
)
def test_hf_dataset_guard(url, blocked):
    assert J._is_huggingface_dataset_or_space_url(url) is blocked


# ---- client isolation ------------------------------------------------
def test_llm_client_is_never_proxied():
    """The summary LLM is an internal endpoint; it must not go via JINA_PROXY."""
    llm = J._get_llm_client()
    assert llm.trust_env is False


def test_web_client_ignores_ambient_proxy_env():
    web = J._get_web_client()
    assert web.trust_env is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-o", "addopts="]))
