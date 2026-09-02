# Copyright (c) 2026 AllSpark Research.
# This source code is licensed under the Apache 2.0 License.
#
# Tests for append-only conversation log + markers.
"""Unit tests for src/utils/conversation_markers.py.

Core guarantee under test: for a ``ConversationHistory``, ``replay(full) == list(visible)``
after EVERY structural operation (append/extend/+=/pop/discard). Plus: markers never leak
into the visible list, the append-only ``full`` retains rounds that context management
dropped (fidelity), stale markers are no-ops, ``copy()`` is untracked, bulk/positional
mutation is blocked, and tool_call id pairing survives replay (SFT R07/R08 dependency).
"""

import random

import pytest

from src.utils.conversation_markers import (
    MARKER_ROLE,
    ConversationHistory,
    build_discard_all_marker,
    build_rollback_marker,
    is_marker,
    replay,
    strip_markers,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _user(txt="task"):
    return {"role": "user", "content": txt}


def _assistant(i, with_tool=True):
    m = {"role": "assistant", "content": f"a{i}"}
    if with_tool:
        m["tool_calls"] = [{"id": f"call_{i}", "type": "function",
                            "function": {"name": "web_search", "arguments": "{}"}}]
    return m


def _tool(i):
    return {"role": "tool", "tool_call_id": f"call_{i}", "content": f"result-{i}"}


def _assert_invariant(h: ConversationHistory):
    assert replay(h.full) == list(h), (
        f"drift: replay(full) has {len(replay(h.full))} msgs, visible has {len(h)}"
    )
    assert not any(is_marker(m) for m in h), "markers must never appear in the visible list"


# --------------------------------------------------------------------------- #
# marker builders / predicates / pure replay
# --------------------------------------------------------------------------- #
def test_marker_builders_and_predicate():
    r = build_rollback_marker(3, reason="format_error")
    d = build_discard_all_marker(1, reason="hcm")
    assert is_marker(r) and is_marker(d)
    assert r["role"] == MARKER_ROLE and r["op"] == "rollback" and r["count"] == 3
    assert d["op"] == "discard_all" and d["prefix_len"] == 1
    assert not is_marker({"role": "user", "content": "x"})
    assert not is_marker("not a dict")


def test_replay_appends_only_is_identity():
    full = [_user(), _assistant(0), _tool(0)]
    assert replay(full) == full


def test_replay_rollback_hides_last_n():
    full = [_user(), _assistant(0), _tool(0), build_rollback_marker(2)]
    assert replay(full) == [_user()]


def test_replay_discard_all_truncates_to_prefix():
    full = [_user(), _assistant(0), _tool(0), build_discard_all_marker(1)]
    assert replay(full) == [_user()]


def test_replay_stale_markers_are_noops():
    # over-range rollback and over-range discard both no-op (staleness-safe)
    full = [_user("x"), build_rollback_marker(99), build_discard_all_marker(50)]
    assert replay(full) == [_user("x")]
    # malformed payloads (bool / missing / negative) ignored
    assert replay([_user("x"), {"role": MARKER_ROLE, "op": "rollback", "count": True}]) == [_user("x")]
    assert replay([_user("x"), {"role": MARKER_ROLE, "op": "rollback"}]) == [_user("x")]
    assert replay([_user("x"), {"role": MARKER_ROLE, "op": "discard_all", "prefix_len": -1}]) == [_user("x")]
    assert replay([_user("x"), {"role": MARKER_ROLE, "op": "unknown"}]) == [_user("x")]


def test_replay_interleaved_append_after_rollback():
    full = [_user(), _assistant(0), build_rollback_marker(1), _assistant(1)]
    assert replay(full) == [_user(), _assistant(1)]


def test_strip_markers_keeps_rolled_back_rounds():
    full = [_user(), _assistant(0), _tool(0), build_rollback_marker(2)]
    # strip_markers keeps real msgs (incl. rolled-back); replay hides them
    assert strip_markers(full) == [_user(), _assistant(0), _tool(0)]
    assert replay(full) == [_user()]


# --------------------------------------------------------------------------- #
# ConversationHistory: tracked mutators keep the invariant
# --------------------------------------------------------------------------- #
def test_append_extend_iadd_pop_tracked():
    h = ConversationHistory([_user()])
    _assert_invariant(h)
    h.append(_assistant(0)); _assert_invariant(h)
    h.extend([_tool(0), _assistant(1)]); _assert_invariant(h)
    h += [_tool(1)]; _assert_invariant(h)
    assert len(h) == 5
    h.pop(); _assert_invariant(h)             # default tail pop
    h.pop(-1); _assert_invariant(h)           # explicit tail pop
    assert [m["role"] for m in h] == ["user", "assistant", "tool"]


def test_enforce_multi_pop_preserves_fidelity():
    h = ConversationHistory([_user()])
    for i in range(3):
        h.append(_assistant(i)); h.append(_tool(i))
    _assert_invariant(h)
    # emulate enforce_context_budget / drop_last_round popping 2 full rounds
    for _ in range(4):
        h.pop()
    _assert_invariant(h)
    assert [m["role"] for m in h] == ["user", "assistant", "tool"]
    # fidelity: full retains ALL 7 real messages (seed + 3*(assistant+tool))
    assert len(strip_markers(h.full)) == 7


def test_discard_all_to_first_user():
    h = ConversationHistory([_user("Q")])
    for i in range(3):
        h.append(_assistant(i)); h.append(_tool(i))
    n = h.discard_all_to_first_user(reason="hcm_discard_all")
    assert n == 6
    assert list(h) == [_user("Q")]
    _assert_invariant(h)
    # fidelity: discarded rounds still present in full; exactly one discard marker
    assert len(strip_markers(h.full)) == 7
    assert sum(1 for m in h.full if m.get("op") == "discard_all") == 1
    # continue appending after discard keeps the invariant
    h.append(_assistant(9)); _assert_invariant(h)
    assert [m["role"] for m in h] == ["user", "assistant"]


def test_discard_all_no_user_is_noop():
    h = ConversationHistory([{"role": "assistant", "content": "x"}])
    assert h.discard_all_to_first_user() == 0
    assert sum(1 for m in h.full if is_marker(m)) == 0


def test_copy_returns_plain_untracked_list():
    h = ConversationHistory([_user(), _assistant(0)])
    c = h.copy()
    assert type(c) is list
    n_full = len(h.full)
    c.pop()                       # mutating the copy must NOT touch the tracked log
    c.append({"role": "x"})
    assert len(h.full) == n_full


def test_blocked_bulk_and_positional_mutations_raise():
    h = ConversationHistory([_user(), _assistant(0)])
    for call in (
        lambda: h.insert(0, {}),
        lambda: h.remove(h[0]),
        lambda: h.clear(),
        lambda: h.sort(),
        lambda: h.reverse(),
        lambda: h.__setitem__(0, {}),
        lambda: h.__delitem__(0),
    ):
        with pytest.raises(NotImplementedError):
            call()


def test_non_tail_pop_raises():
    h = ConversationHistory([_user(), _assistant(0), _tool(0)])
    with pytest.raises(RuntimeError):
        h.pop(0)


def test_tool_call_id_pairing_survives_replay():
    # SFT converter (R07/R08) requires assistant.tool_calls[].id <-> tool.tool_call_id.
    h = ConversationHistory([_user()])
    h.append(_assistant(0)); h.append(_tool(0))
    h.append(_assistant(1)); h.append(_tool(1))
    h.pop(); h.pop()                         # drop the second round
    view = replay(h.full)
    asst = [m for m in view if m["role"] == "assistant"]
    tools = [m for m in view if m["role"] == "tool"]
    assert len(asst) == 1 and len(tools) == 1
    assert asst[0]["tool_calls"][0]["id"] == tools[0]["tool_call_id"] == "call_0"


def test_realistic_orchestrator_sequence():
    """Mirror a full main-agent lifecycle: turns, enforce trim, ensure_summary pop,
    HCM discard mid-way, terminal summary with a no-boxed retry."""
    h = ConversationHistory([_user("Q")])
    # 3 tool-calling turns
    for i in range(3):
        h.append(_assistant(i)); h.append(_tool(i)); _assert_invariant(h)
    # ensure_summary_context style: pop trailing tool + assistant
    h.pop(); h.pop(); _assert_invariant(h)
    # HCM discard-all reset, then keep going
    h.discard_all_to_first_user(reason="hcm"); _assert_invariant(h)
    for i in range(3, 5):
        h.append(_assistant(i)); h.append(_tool(i)); _assert_invariant(h)
    # terminal: strip trailing tool, enforce drop 1 round, append summary prompt
    h.pop()                                   # trailing tool
    h.pop(); h.pop()                          # drop_last_round (tool+assistant)... only 1 asst left
    h.append(_user("SUMMARY_PROMPT")); _assert_invariant(h)
    # attempt 1 -> no boxed -> pop assistant -> attempt 2
    h.append({"role": "assistant", "content": "no box"}); _assert_invariant(h)
    h.pop()
    h.append({"role": "assistant", "content": r"\boxed{42}"}); _assert_invariant(h)
    # visible ends with the final boxed answer; markers absent from visible
    assert list(h)[-1]["content"] == r"\boxed{42}"
    _assert_invariant(h)


@pytest.mark.parametrize("seed", range(25))
def test_fuzz_invariant_holds(seed):
    rng = random.Random(seed)
    h = ConversationHistory([_user()])
    counter = 0
    for _ in range(60):
        op = rng.choice(["append", "append", "append", "pop", "extend", "discard"])
        if op == "append":
            counter += 1
            h.append(_assistant(counter) if rng.random() < 0.5 else _tool(counter))
        elif op == "extend":
            counter += 1
            h.extend([_assistant(counter), _tool(counter)])
        elif op == "pop" and len(h) > 1:
            h.pop()
        elif op == "discard" and rng.random() < 0.15:
            h.discard_all_to_first_user()
        _assert_invariant(h)
    # full always retains >= visible real messages (never loses history)
    assert len(strip_markers(h.full)) >= len(h)
