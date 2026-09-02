# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Regression tests for execution-time tool_blacklist enforcement.

Background
----------
`tool_blacklist` used to be applied ONLY in `get_all_tool_definitions()`, i.e.
blacklisted tools were hidden from the prompt but remained callable. A model
with a strong prior (fine-tuned on traces where the blacklisted flat name
existed) keeps emitting e.g. `tool-serper-search__scrape_website`, and because
the serper server is still configured for its *other* tool (`web_search`), the
call succeeded — silently routing scrape traffic to the disabled backend.

Measured on a browsecomp_zh smoke run with `serper_jina_search_agent`: 4 of 9
scrape calls (44%) went to the blacklisted serper backend instead of Jina.

These tests pin the fix: a blacklisted (server, tool) pair must be rejected in
`execute_tool_call` before any transport happens, and the error must name the
surviving backend so the model can retry correctly.
"""

import asyncio
import sys
from pathlib import Path

import pytest

_APP = Path(__file__).resolve().parent.parent
_TOOLS_SRC = _APP / "tools" / "src"
for p in (str(_APP), str(_TOOLS_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from miroflow_tools.manager import ToolManager  # noqa: E402


def _mgr(blacklist=None):
    """ToolManager over two fake servers; no transport is ever established."""
    configs = [
        {"name": "tool-serper-search", "params": "http://fake-serper.invalid"},
        {"name": "tool-jina-scrape", "params": "http://fake-jina.invalid"},
    ]
    return ToolManager(configs, tool_blacklist=blacklist)


def _seed_index(mgr):
    """Simulate what get_all_tool_definitions() records after discovery."""
    mgr._visible_tool_servers = {
        "web_search": ["tool-serper-search"],
        "scrape_website": ["tool-jina-scrape"],
    }


BL = {("tool-serper-search", "scrape_website")}


def test_blacklisted_call_is_rejected():
    mgr = _mgr(BL)
    _seed_index(mgr)
    res = asyncio.run(
        mgr.execute_tool_call(
            server_name="tool-serper-search",
            tool_name="scrape_website",
            arguments={"url": "https://example.com"},
        )
    )
    assert "error" in res, "blacklisted call must not return a result"
    assert "result" not in res
    assert res["server_name"] == "tool-serper-search"
    assert res["tool_name"] == "scrape_website"
    assert "disabled" in res["error"]


def test_rejection_names_the_surviving_backend():
    mgr = _mgr(BL)
    _seed_index(mgr)
    res = asyncio.run(
        mgr.execute_tool_call(
            server_name="tool-serper-search",
            tool_name="scrape_website",
            arguments={"url": "https://example.com"},
        )
    )
    # The model must be told the exact flat name to retry with.
    assert "tool-jina-scrape__scrape_website" in res["error"]


def test_rejection_happens_before_transport():
    """No socket/subprocess may be touched: params are deliberately invalid."""
    mgr = _mgr(BL)
    _seed_index(mgr)

    def _boom(_name):
        raise AssertionError("get_server_params must not be reached")

    mgr.get_server_params = _boom
    res = asyncio.run(
        mgr.execute_tool_call(
            server_name="tool-serper-search",
            tool_name="scrape_website",
            arguments={"url": "https://example.com"},
        )
    )
    assert "error" in res


def test_non_blacklisted_tool_on_same_server_still_passes_the_gate():
    """serper's web_search must survive — only its scrape_website is disabled."""
    mgr = _mgr(BL)
    _seed_index(mgr)
    # Force a recognisable downstream failure to prove we got past the gate.
    mgr.get_server_params = lambda _name: None
    res = asyncio.run(
        mgr.execute_tool_call(
            server_name="tool-serper-search",
            tool_name="web_search",
            arguments={"q": "hello"},
        )
    )
    assert "not found" in res["error"], (
        "web_search should reach the server lookup, not the blacklist gate"
    )
    assert "disabled" not in res["error"]


def test_allowed_backend_passes_the_gate():
    mgr = _mgr(BL)
    _seed_index(mgr)
    mgr.get_server_params = lambda _name: None
    res = asyncio.run(
        mgr.execute_tool_call(
            server_name="tool-jina-scrape",
            tool_name="scrape_website",
            arguments={"url": "https://example.com"},
        )
    )
    assert "disabled" not in res["error"]


def test_empty_blacklist_blocks_nothing():
    mgr = _mgr(None)
    mgr.get_server_params = lambda _name: None
    res = asyncio.run(
        mgr.execute_tool_call(
            server_name="tool-serper-search",
            tool_name="scrape_website",
            arguments={"url": "https://example.com"},
        )
    )
    assert "disabled" not in res["error"]


def test_missing_index_degrades_gracefully():
    """Rejection must still work if discovery never ran (no suggestion known)."""
    mgr = _mgr(BL)  # _visible_tool_servers left at its {} default
    res = asyncio.run(
        mgr.execute_tool_call(
            server_name="tool-serper-search",
            tool_name="scrape_website",
            arguments={"url": "https://example.com"},
        )
    )
    assert "disabled" in res["error"]
    assert "No alternative backend" in res["error"]


def test_blacklisted_alternative_is_not_suggested():
    """If every backend for a tool is blacklisted, suggest none of them."""
    mgr = _mgr(
        {
            ("tool-serper-search", "scrape_website"),
            ("tool-jina-scrape", "scrape_website"),
        }
    )
    _seed_index(mgr)
    res = asyncio.run(
        mgr.execute_tool_call(
            server_name="tool-serper-search",
            tool_name="scrape_website",
            arguments={"url": "https://example.com"},
        )
    )
    assert "tool-jina-scrape" not in res["error"]
    assert "No alternative backend" in res["error"]


def test_index_is_built_from_discovery_output():
    """get_all_tool_definitions() must populate the suggestion index."""
    mgr = _mgr(BL)

    async def fake_discovery():
        # Mimic the real return shape, post-blacklist-filtering.
        servers = [
            {"name": "tool-serper-search", "tools": [{"name": "web_search"}]},
            {"name": "tool-jina-scrape", "tools": [{"name": "scrape_website"}]},
            # error entries must not poison the index
            {"name": "tool-broken", "tools": [{"error": "Unable to fetch tools"}]},
        ]
        mgr._visible_tool_servers = {}
        for s in servers:
            for t in s.get("tools", []):
                if t.get("name"):
                    mgr._visible_tool_servers.setdefault(t["name"], []).append(s["name"])
        return servers

    asyncio.run(fake_discovery())
    assert mgr._visible_tool_servers == {
        "web_search": ["tool-serper-search"],
        "scrape_website": ["tool-jina-scrape"],
    }


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-o", "addopts="]))
