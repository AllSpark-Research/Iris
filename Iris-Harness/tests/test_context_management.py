# Copyright (c) 2026 AllSpark Research.
# This source code is licensed under the Apache 2.0 License.
#
# Tests for the context-management strategies the report reports on.

"""The three context-management knobs are what separate the report's four
regimes, so each one gets a test that pins its contract.

* recency-K (``keep_tool_result``) folds old tool results out of the *outgoing*
  payload while leaving thinking and tool calls untouched.
* discard-all (``context_discard_threshold``) resets the visible conversation to
  the opening question and records a marker, so nothing is lost from the trace.

Both are checked against the real implementations, not reimplementations.
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pytest  # noqa: E402

from src.llm.base_client import BaseClient  # noqa: E402
from src.utils.conversation_markers import ConversationHistory, replay  # noqa: E402

FOLD_MARKER = "omitted"


class _StubLogger:
    def log_step(self, *args, **kwargs):
        pass


class _StubClient:
    """Minimal stand-in: folding only needs a logger and the tool-call mode."""

    task_log = _StubLogger()

    def __init__(self, tool_call_mode="native_fc"):
        self.tool_call_mode = tool_call_mode


def _conversation(n_rounds: int):
    """A user turn followed by ``n_rounds`` of (assistant tool call, tool result)."""
    messages = [{"role": "user", "content": "Q"}]
    for i in range(n_rounds):
        call_id = f"call-{i}"
        messages.append(
            {
                "role": "assistant",
                "content": f"thinking {i}",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "web_search", "arguments": "{}"},
                    }
                ],
            }
        )
        messages.append(
            {"role": "tool", "tool_call_id": call_id, "content": f"RESULT-{i} " * 50}
        )
    return messages


def _fold(messages, keep, tool_call_mode="native_fc"):
    return BaseClient._remove_tool_result_from_messages(
        _StubClient(tool_call_mode), [dict(m) for m in messages], keep
    )


SUMMARY_PROMPT = "Wrap the final answer in \\boxed{}."


@pytest.mark.parametrize("keep,expected_verbatim", [(-1, 3), (0, 0), (1, 1), (2, 2)])
def test_recency_k_keeps_exactly_the_last_k_tool_results(keep, expected_verbatim):
    out = _fold(_conversation(3), keep)
    tool_msgs = [m for m in out if m["role"] == "tool"]

    assert len(tool_msgs) == 3, "folding must not drop tool messages, only their bodies"
    verbatim = [m for m in tool_msgs if "RESULT-" in m["content"]]
    assert len(verbatim) == expected_verbatim

    # The surviving ones must be the most recent, not an arbitrary subset.
    if expected_verbatim:
        assert verbatim == tool_msgs[-expected_verbatim:]
    for folded in tool_msgs[: len(tool_msgs) - expected_verbatim]:
        assert FOLD_MARKER in folded["content"]


def test_recency_k_never_touches_thinking_or_tool_calls():
    original = _conversation(3)
    out = _fold(original, 1)

    before = [m for m in original if m["role"] == "assistant"]
    after = [m for m in out if m["role"] == "assistant"]
    assert after == before, "recency-K folds observations only, never the agent's own turns"


def test_keep_all_is_a_no_op():
    original = _conversation(3)
    assert _fold(original, -1) == original


def test_discard_all_resets_to_the_question_and_stays_replayable():
    history = ConversationHistory([{"role": "user", "content": "Q"}])
    for i in range(3):
        history.append({"role": "assistant", "content": f"thinking {i}"})
        history.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"R{i}"})
    assert len(history) == 7

    discarded = history.discard_all_to_first_user(reason="hcm_discard_all")

    assert discarded == 6
    assert list(history) == [{"role": "user", "content": "Q"}]
    # The audit log keeps every message plus a marker recording what happened.
    marker_ops = [
        m.get("op") for m in history.full if m.get("role") == "context_marker"
    ]
    assert "discard_all" in marker_ops
    assert len(history.full) > len(history)
    # The invariant the append-only log exists to guarantee.
    assert replay(history.full) == list(history)


def test_agent_keeps_working_after_a_discard():
    history = ConversationHistory([{"role": "user", "content": "Q"}])
    history.append({"role": "assistant", "content": "thinking"})
    history.append({"role": "tool", "tool_call_id": "c0", "content": "R0"})
    history.discard_all_to_first_user(reason="hcm_discard_all")

    history.append({"role": "assistant", "content": "fresh start"})
    assert list(history) == [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "fresh start"},
    ]
    assert replay(history.full) == list(history)


@pytest.mark.parametrize("keep", [0, 1, 2, -1])
def test_the_final_summary_prompt_is_never_folded_away(keep):
    """The summary prompt is appended as a user turn at the end of an episode.

    Folding it leaves the model staring at "Tool result is omitted to save
    tokens." instead of the instruction to produce an answer, which at
    keep_tool_result=0 silently destroyed every final summary.
    """
    messages = _conversation(3) + [{"role": "user", "content": SUMMARY_PROMPT}]
    out = _fold(messages, keep)

    assert out[-1]["content"] == SUMMARY_PROMPT
    # ...and it must not consume one of the K slots either.
    verbatim = [m for m in out if m["role"] == "tool" and "RESULT-" in m["content"]]
    assert len(verbatim) == (3 if keep == -1 else keep)


def test_mcp_xml_mode_folds_user_turns_but_spares_the_last():
    """Without a `tool` role, results arrive as user turns — but the trailing
    message is still the live instruction and must survive."""
    messages = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "t1"},
        {"role": "user", "content": "OBS-1 " * 20},
        {"role": "assistant", "content": "t2"},
        {"role": "user", "content": "OBS-2 " * 20},
        {"role": "user", "content": SUMMARY_PROMPT},
    ]
    out = _fold(messages, 0, tool_call_mode="mcp_xml")

    assert out[0]["content"] == "Q", "the task itself is never folded"
    assert out[-1]["content"] == SUMMARY_PROMPT, "the live instruction is never folded"
    assert all(FOLD_MARKER in out[i]["content"] for i in (2, 4))
