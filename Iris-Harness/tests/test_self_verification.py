# Copyright (c) 2026 AllSpark Research.
# This source code is licensed under the Apache 2.0 License.
#
# Tests for agent-side self-verification.
"""Unit tests for benchmarks/self_verification.py.

Covers (1) the 3-level verdict parser and (2) the verify→reanswer→re-verify orchestration
loop with `execute_task_pipeline` monkeypatched, so loop control flow is validated
deterministically without any live model / tools.
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

# benchmarks/ is on sys.path when the harness runs `python benchmarks/common_benchmark.py`;
# make the sibling module importable here too.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import self_verification as sv  # noqa: E402


# --------------------------------------------------------------------------- #
# parse_verdict
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"rationale":"ok","verdict":"correct"}', "correct"),
        ('pre\n{"rationale":"bad","verdict":"incorrect"}\npost', "incorrect"),
        ("Final verdict: incorrect", "incorrect"),
        ("The verdict is correct.", "correct"),
        ("My verdict is incorrect because the entity differs.", "incorrect"),
        ("verdict: correct ... but the answer is wrong", "unparseable"),  # contradictory
        ("", "unparseable"),
        ('<think>hmm</think>{"verdict":"correct","rationale":"x"}', "correct"),
        ('{"verdict":"maybe"}', "unparseable"),
        ('```json\n{"verdict":"incorrect","rationale":"y"}\n```', "incorrect"),
    ],
)
def test_parse_verdict(text, expected):
    verdict, _ = sv.parse_verdict(text)
    assert verdict == expected


# --------------------------------------------------------------------------- #
# orchestration loop (execute_task_pipeline monkeypatched)
# --------------------------------------------------------------------------- #
class _StubEvaluator:
    """Minimal stand-in; its attrs are only forwarded to the (faked) pipeline."""

    cfg = None
    main_agent_tool_manager = None
    sub_agent_tool_managers = {}
    output_formatter = None

    def get_log_dir(self):
        return "/tmp"


def _install_fake_pipeline(monkeypatch, *, verdicts, reanswers):
    """Fake execute_task_pipeline: returns scripted verdict text for sv-verify-* task_ids
    and scripted answers for sv-reanswer-* task_ids. Records the task_id call order."""
    v_iter = iter(verdicts)
    r_iter = iter(reanswers)
    calls = []

    async def fake(**kwargs):
        task_id = kwargs["task_id"]
        calls.append(task_id)
        if "sv-verify" in task_id:
            return ("resp", next(v_iter), f"/tmp/{task_id}.json", None)
        if "sv-reanswer" in task_id:
            return ("resp", next(r_iter), f"/tmp/{task_id}.json", None)
        raise AssertionError(f"unexpected task_id {task_id}")

    monkeypatch.setattr(sv, "execute_task_pipeline", fake)
    return calls


def _run(monkeypatch, *, verdicts, reanswers, initial="INIT", max_reanswer):
    calls = _install_fake_pipeline(monkeypatch, verdicts=verdicts, reanswers=reanswers)
    final, meta = asyncio.run(
        sv.run_self_verification(
            _StubEvaluator(),
            question="Q?",
            reanswer_task_description="Q? (boxed)",
            initial_answer=initial,
            task_file_path="",
            base_task_id="task_0_attempt-1",
            max_reanswer_attempts=max_reanswer,
        )
    )
    return final, meta, calls


def test_first_verify_correct_keeps_answer(monkeypatch):
    final, meta, calls = _run(
        monkeypatch, verdicts=['{"verdict":"correct"}'], reanswers=[], max_reanswer=1
    )
    assert final == "INIT"
    assert meta["verified"] is True and meta["final_verdict"] == "correct"
    assert meta["reanswer_attempts_used"] == 0 and meta["answer_changed"] is False
    assert calls == ["task_0_attempt-1_sv-verify-1"]  # exactly one verify, no reanswer


def test_incorrect_then_reanswer_then_verify_correct(monkeypatch):
    final, meta, calls = _run(
        monkeypatch,
        verdicts=['{"verdict":"incorrect"}', '{"verdict":"correct"}'],
        reanswers=["NEW_ANSWER_1"],
        max_reanswer=2,
    )
    assert final == "NEW_ANSWER_1"
    assert meta["verified"] is True and meta["reanswer_attempts_used"] == 1
    assert meta["answer_changed"] is True
    assert calls == [
        "task_0_attempt-1_sv-verify-1",
        "task_0_attempt-1_sv-reanswer-1",
        "task_0_attempt-1_sv-verify-2",
    ]


def test_budget_exhausted_keeps_last_reanswer_unverified(monkeypatch):
    final, meta, calls = _run(
        monkeypatch,
        verdicts=['{"verdict":"incorrect"}'],  # only ONE verify happens (budget=1)
        reanswers=["NEW_ANSWER_1"],
        max_reanswer=1,
    )
    assert final == "NEW_ANSWER_1"
    assert meta["verified"] is False
    assert meta["final_verdict"] == "budget_exhausted_unverified"
    assert meta["reanswer_attempts_used"] == 1
    # budget=1 => verify initial, reanswer once, then STOP (no re-verify)
    assert calls == ["task_0_attempt-1_sv-verify-1", "task_0_attempt-1_sv-reanswer-1"]


def test_verify_only_never_replaces(monkeypatch):
    # max_reanswer_attempts=0 => verify-only annotation, answer never changes
    final, meta, calls = _run(
        monkeypatch, verdicts=['{"verdict":"incorrect"}'], reanswers=[], max_reanswer=0
    )
    assert final == "INIT" and meta["answer_changed"] is False
    assert meta["reanswer_attempts_used"] == 0 and meta["final_verdict"] == "incorrect"
    assert calls == ["task_0_attempt-1_sv-verify-1"]


def test_reanswer_failure_keeps_previous(monkeypatch):
    # verify incorrect -> reanswer returns FORMAT_ERROR -> keep the initial answer
    final, meta, calls = _run(
        monkeypatch,
        verdicts=['{"verdict":"incorrect"}'],
        reanswers=[sv.FORMAT_ERROR_MESSAGE],
        max_reanswer=1,
    )
    assert final == "INIT" and meta["answer_changed"] is False
    assert meta["reanswer_attempts_used"] == 1


def test_unparseable_defaults_conservative_accept(monkeypatch):
    # verifier emits garbage -> unparseable -> default "correct" -> keep, no reanswer
    final, meta, calls = _run(
        monkeypatch, verdicts=["totally not a verdict"], reanswers=[], max_reanswer=1
    )
    assert final == "INIT" and meta["verified"] is True
    assert calls == ["task_0_attempt-1_sv-verify-1"]


# --------------------------------------------------------------------------- #
# candidate hints: rendering (no-op vs hints) + loader + threading into verify
# --------------------------------------------------------------------------- #
def test_build_verifier_task_no_hints_is_noop():
    t0 = sv.build_verifier_task("Q", "A")
    assert t0 == sv.build_verifier_task("Q", "A", None) == sv.build_verifier_task("Q", "A", [])
    assert "During its investigation" not in t0
    # byte-identical to the template with an empty candidates block => true no-op
    assert t0 == sv._VERIFIER_TASK_TEMPLATE.format(
        question="Q", candidate="A", candidates_block=""
    )


def test_build_verifier_task_with_hints_renders_section():
    t = sv.build_verifier_task("Q", "A", ["Bravo", "Charlie"])
    assert "During its investigation" in t
    assert "- Bravo" in t and "- Charlie" in t
    assert 'grounds for "incorrect"' in t  # conservative framing retained


def test_load_candidate_answers_dedup_exclude_cap(tmp_path):
    log = tmp_path / "task.json"
    log.write_text(
        json.dumps(
            {
                "trace_data": {
                    "intermediate_boxed_answers": [
                        "Foo", "foo ", "Bar", "A", "  Bar ", "Baz", "Qux", "Quux", "Corge",
                    ]
                }
            }
        )
    )
    # dedup 'foo'/'Bar', drop committed 'A', cap 5
    assert sv.load_candidate_answers(str(log), "A", cap=5) == ["Foo", "Bar", "Baz", "Qux", "Quux"]
    assert sv.load_candidate_answers(str(log), "A", cap=2) == ["Foo", "Bar"]


def test_load_candidate_answers_failopen(tmp_path):
    assert sv.load_candidate_answers("/no/such/file.json", "A") == []
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"foo": 1}))  # no trace_data
    assert sv.load_candidate_answers(str(empty), "A") == []


def test_candidate_hints_reach_verifier_task(monkeypatch):
    seen = {}

    async def fake(**kwargs):
        tid = kwargs["task_id"]
        assert "sv-verify" in tid
        seen["verify_task"] = kwargs["task_description"]
        return ("resp", '{"verdict":"correct"}', f"/tmp/{tid}.json", None)

    monkeypatch.setattr(sv, "execute_task_pipeline", fake)
    final, meta = asyncio.run(
        sv.run_self_verification(
            _StubEvaluator(),
            question="Who?",
            reanswer_task_description="Who? (boxed)",
            initial_answer="Alice",
            task_file_path="",
            base_task_id="t_a1",
            max_reanswer_attempts=0,
            candidate_answers=["Bob", "Carol"],
        )
    )
    assert "During its investigation" in seen["verify_task"]
    assert "- Bob" in seen["verify_task"] and "- Carol" in seen["verify_task"]
    assert meta["candidate_hints_used"] == 2


def test_no_candidate_hints_verifier_task_is_clean(monkeypatch):
    seen = {}

    async def fake(**kwargs):
        seen["verify_task"] = kwargs["task_description"]
        return ("resp", '{"verdict":"correct"}', f"/tmp/{kwargs['task_id']}.json", None)

    monkeypatch.setattr(sv, "execute_task_pipeline", fake)
    _final, meta = asyncio.run(
        sv.run_self_verification(
            _StubEvaluator(),
            question="Who?",
            reanswer_task_description="Who? (boxed)",
            initial_answer="Alice",
            task_file_path="",
            base_task_id="t_a1",
            max_reanswer_attempts=0,
            candidate_answers=None,
        )
    )
    assert "During its investigation" not in seen["verify_task"]
    assert meta["candidate_hints_used"] == 0
