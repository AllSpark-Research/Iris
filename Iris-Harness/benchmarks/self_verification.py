# Copyright (c) 2026 AllSpark Research.
# This source code is licensed under the Apache 2.0 License.
#
# Agent-side self-verification (verify -> reanswer -> re-verify).
"""Agent-side self-verification.

The verify -> reanswer -> re-verify shape follows AxisAgentic
(https://github.com/XYZ-AI-Lab/AxisAgentic), retargeted to this harness.

After the main agent produces a final answer, spin an independent **verifier agent** — a full
agentic run with the SAME search/scrape tools but a *verifier* system prompt — that checks the
candidate answer condition-by-condition and emits a JSON verdict. If the verdict is
``incorrect`` and there is re-answer budget left, **re-answer the original task from scratch**
and re-verify. Loop until a verdict is ``correct`` or the budget is exhausted (then keep the
current answer, un-verified).

Design notes
------------
* Both the verify run and the re-answer run are ordinary :func:`execute_task_pipeline` calls
  (each builds a fresh TaskLog + LLM client + Orchestrator, **reusing** the passed-in tool
  managers) — so there is no new agentic loop and no shared-state to snapshot. Because
  self-verification is orchestrated *outside* ``execute_task_pipeline`` it can never recurse.
* The verify run uses ``run_overrides`` to (a) inject the verifier ``system_prompt`` (bypassing
  ``compose_full_system_prompt`` and its agent-type objective lookup), (b) set
  ``answer_mode="direct"`` so the run returns the verifier's final text verbatim (the JSON
  verdict — no ``\\boxed{}`` extraction), and (c) optionally cap ``max_turns``.
* Conservative by construction: the verifier prompt only says ``incorrect`` on a clear
  condition failure, and an *unparseable* verdict defaults to ``correct`` (accept, don't
  re-answer) so a parse glitch never replaces a good answer.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from src.core.pipeline import execute_task_pipeline
from src.utils.prompt_utils import FORMAT_ERROR_MESSAGE

# --------------------------------------------------------------------------- #
# Prompts (adapted from AxisAgentic recipe/web_search/agent/orchestrator.py,
# retargeted to our tool names: web_search / scrape_website)
# --------------------------------------------------------------------------- #
VERIFIER_SYSTEM_PROMPT = """\
You are a careful answer verifier with access to structured web-search tools (web_search and \
scrape_website). Your job is to determine whether a candidate answer to the original question \
is correct.

Break the original question into its required conditions and verify those conditions one by \
one. Judge the candidate against the exact conditions in the question, not against a looser or \
different question. Actively look for contradictions and alternative answers, but use the \
verdict "incorrect" ONLY when a required condition is clearly not satisfied, clearly \
contradicted by reliable evidence, or the candidate answers a different entity/value. Do NOT \
mark the candidate incorrect merely because a detail is hard to find, evidence is incomplete, \
or another answer is possible but not clearly better. Avoid relying on search-result snippets, \
social-media pages, SEO/trivia/crossword pages, or future-dated pages as decisive evidence for \
either verdict.

Use tools when external evidence is needed. When you are done verifying, output EXACTLY one \
JSON object and no other text:
{"rationale":"<one or two sentences>","verdict":"correct"|"incorrect"}

Do not wrap the JSON in \\boxed{} and do not use markdown code fences."""

# Forced-finalization prompt: injected when the verifier hits max_turns (abnormal
# termination) so it still emits a JSON verdict instead of prose. Ported from
# AxisAgentic _SELF_VERIFICATION_VERDICT_PROMPT.
VERIFIER_SUMMARY_PROMPT = """\
You cannot call tools now. Based only on the verification work above, decide whether the \
candidate answer clearly fails any required condition in the original question. Use \
"incorrect" only for an explicit condition failure, contradiction, or different entity/value. \
If the verification is incomplete or uncertain but no obvious mismatch was found, use \
"correct". Output EXACTLY one JSON object and no other text:
{"rationale":"...","verdict":"correct"|"incorrect"}"""

_VERIFIER_TASK_TEMPLATE = """\
Verify whether the candidate answer is correct for the original question.

Original question:
{question}

Candidate answer:
{candidate}
{candidates_block}
Break the question into required conditions and verify, one by one, whether the candidate \
answer satisfies each condition. Do not only search for support for the candidate; look for \
contradictions and alternative answers. Use the verdict "incorrect" only if a required \
condition is clearly not satisfied, clearly contradicted by reliable evidence, or the \
candidate answers a different entity/value. If a detail is hard to find or remains uncertain \
but there is no obvious mismatch, do not use that uncertainty alone to reject the candidate.

Use tools if needed to verify the answer. When finished, output EXACTLY one JSON object with \
this schema and nothing else:
{{"rationale":"...","verdict":"correct"|"incorrect"}}"""

# Optional block injected into the verifier task when candidate-answer hints are enabled.
# Leading + trailing newline so that, when candidate_answers is empty, `{candidates_block}`
# collapses to "" and the rendered task is byte-for-byte identical to the no-hints version.
_CANDIDATES_SECTION = (
    "\nDuring its investigation the agent also produced these other candidate answers "
    "(for reference only — they may be wrong):\n{items}\n"
    "Consider whether the committed candidate answer above is the best-supported among "
    "these. If a different candidate is clearly better supported by reliable evidence, "
    'judge the committed answer "incorrect". Do NOT treat the mere existence of '
    'alternatives as grounds for "incorrect".\n'
)


def build_verifier_task(
    question: str,
    candidate_answer: str,
    candidate_answers: Optional[List[str]] = None,
) -> str:
    """Build the verifier task. When ``candidate_answers`` is empty/None the output is
    byte-identical to the no-hints task (so use_candidate_hints=false is a true no-op)."""
    candidates_block = ""
    if candidate_answers:
        items = "\n".join(f"- {c}" for c in candidate_answers)
        candidates_block = _CANDIDATES_SECTION.format(items=items)
    return _VERIFIER_TASK_TEMPLATE.format(
        question=question,
        candidate=candidate_answer,
        candidates_block=candidates_block,
    )


def _norm_answer(s: str) -> str:
    """Normalize a boxed answer for dedup / equality (whitespace + case insensitive)."""
    return " ".join((s or "").split()).strip().casefold()


def load_candidate_answers(
    log_file_path: str, committed_answer: str, cap: int = 5
) -> List[str]:
    """Load the main attempt's mid-loop ``\\boxed{}`` candidates from its saved task log
    (``trace_data.intermediate_boxed_answers``) for use as verifier hints.

    Dedups (whitespace/case-insensitive, first surface form kept), drops any candidate
    equal to ``committed_answer``, caps to ``cap``. Fail-open: any error → ``[]``.
    """
    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw = (data.get("trace_data") or {}).get("intermediate_boxed_answers") or []
    except Exception:
        return []
    seen = {_norm_answer(committed_answer)}
    out: List[str] = []
    for c in raw:
        if not isinstance(c, str):
            continue
        key = _norm_answer(c)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(c.strip())
        if len(out) >= cap:
            break
    return out


# --------------------------------------------------------------------------- #
# Verdict parsing — 3-level fallback (ported from AxisAgentic)
# --------------------------------------------------------------------------- #
_LOOSE_VERDICT_PATTERNS = [
    r"(?i)\bverdict\b\s*[:=\-]\s*[\"']?\s*(incorrect|correct)\b",
    r"(?i)\bverdict\b\s+(?:is|was)\s+(?:that\s+)?[\"']?(incorrect|correct)\b",
    r"(?i)\bfinal\s+verdict\b\s*[:=\-]?\s*[\"']?\s*(incorrect|correct)\b",
    r"(?i)\bjudg(?:e)?ment\b\s*[:=\-]\s*[\"']?\s*(incorrect|correct)\b",
    r"(?i)\bcandidate\s+answer\s+(?:is|was)\s+(not\s+correct|incorrect|correct|wrong|right)\b",
    r"(?i)\banswer\s+(?:is|was)\s+(not\s+correct|incorrect|correct|wrong|right)\b",
]


def _verdict_from_obj(obj: Any) -> Optional[str]:
    if not isinstance(obj, dict):
        return None
    v = str(obj.get("verdict") or "").strip().lower()
    return v if v in {"correct", "incorrect"} else None


def _scan_json_verdict(text: str) -> Optional[Dict[str, Any]]:
    """Find the first ``{...}`` JSON object carrying a valid verdict (JSON embedded in prose)."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            parsed, _end = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if _verdict_from_obj(parsed) is not None:
            return parsed
    return None


def _loose_verdict(text: str) -> Optional[Dict[str, Any]]:
    """Regex fallback; accept only when every match agrees on a single verdict."""
    verdicts = set()
    for pattern in _LOOSE_VERDICT_PATTERNS:
        for m in re.finditer(pattern, text):
            val = m.group(1).lower()
            verdicts.add("incorrect" if val in {"not correct", "incorrect", "wrong"} else "correct")
    if len(verdicts) == 1:
        return {"verdict": verdicts.pop(), "rationale": "recovered from non-JSON output"}
    return None


def parse_verdict(raw_text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Parse a verifier verdict. Returns ``(verdict, parsed)`` where verdict is
    ``"correct"`` / ``"incorrect"`` / ``"unparseable"``."""
    text = (raw_text or "").strip()
    if not text:
        return "unparseable", None
    # Defensive: strip a trailing <think> block if one leaked through.
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    # L1: whole string is JSON
    try:
        obj = json.loads(text)
        v = _verdict_from_obj(obj)
        if v is not None:
            return v, obj
    except json.JSONDecodeError:
        pass
    # L2: first embedded JSON object with a valid verdict
    obj = _scan_json_verdict(text)
    if obj is not None:
        return _verdict_from_obj(obj), obj  # type: ignore[return-value]
    # L3: loose regex (must be unambiguous)
    obj = _loose_verdict(text)
    if obj is not None:
        return obj["verdict"], obj
    return "unparseable", None


# --------------------------------------------------------------------------- #
# Sub-runs (verify / reanswer) — thin wrappers over execute_task_pipeline
# --------------------------------------------------------------------------- #
async def _run_verify(
    evaluator: Any,
    *,
    question: str,
    candidate_answer: str,
    task_file_path: str,
    task_id: str,
    verify_max_turns: Optional[int],
    unparseable_verdict: str,
    candidate_answers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    run_overrides: Dict[str, Any] = {
        "system_prompt": VERIFIER_SYSTEM_PROMPT,
        "answer_mode": "direct",
        # Force a JSON verdict even if the verifier exhausts its turns (abnormal
        # termination) — otherwise the direct-mode summary would ask for a prose answer.
        "summary_prompt": VERIFIER_SUMMARY_PROMPT,
    }
    if verify_max_turns:
        run_overrides["max_turns"] = int(verify_max_turns)

    response, verifier_text, log_file_path, _fail = await execute_task_pipeline(
        cfg=evaluator.cfg,
        task_id=task_id,
        task_description=build_verifier_task(
            question, candidate_answer, candidate_answers
        ),
        task_file_name=task_file_path,
        main_agent_tool_manager=evaluator.main_agent_tool_manager,
        sub_agent_tool_managers=evaluator.sub_agent_tool_managers,
        output_formatter=evaluator.output_formatter,
        ground_truth=None,  # the verifier must never see ground truth
        log_dir=str(evaluator.get_log_dir()),
        is_final_retry=True,
        run_overrides=run_overrides,
    )
    raw_verdict, parsed = parse_verdict(verifier_text)
    effective = raw_verdict if raw_verdict in {"correct", "incorrect"} else unparseable_verdict
    record = {
        "role": "verifier",
        "task_id": task_id,
        "log_file_path": log_file_path,
        "candidate_preview": (candidate_answer or "")[:200],
        "raw_verdict": raw_verdict,
        "effective_verdict": effective,
        "rationale": (parsed or {}).get("rationale", "")[:500] if parsed else "",
        "verifier_text_preview": (verifier_text or "")[:500],
    }
    return {"verdict": effective, "record": record}


async def _run_reanswer(
    evaluator: Any,
    *,
    reanswer_task_description: str,
    task_file_path: str,
    task_id: str,
) -> Dict[str, Any]:
    response, new_answer, log_file_path, _fail = await execute_task_pipeline(
        cfg=evaluator.cfg,
        task_id=task_id,
        task_description=reanswer_task_description,  # pristine original task (clean resample)
        task_file_name=task_file_path,
        main_agent_tool_manager=evaluator.main_agent_tool_manager,
        sub_agent_tool_managers=evaluator.sub_agent_tool_managers,
        output_formatter=evaluator.output_formatter,
        ground_truth=None,
        log_dir=str(evaluator.get_log_dir()),
        is_final_retry=True,
        run_overrides=None,  # normal agent config (same tools + system prompt as main)
    )
    record = {
        "role": "reanswer",
        "task_id": task_id,
        "log_file_path": log_file_path,
        "output_preview": (new_answer or "")[:200],
        "produced_answer": bool(new_answer and new_answer != FORMAT_ERROR_MESSAGE),
    }
    return {"answer": new_answer, "record": record}


# --------------------------------------------------------------------------- #
# Orchestration loop (mirrors AxisAgentic `_maybe_apply_self_verification`)
# --------------------------------------------------------------------------- #
async def run_self_verification(
    evaluator: Any,
    *,
    question: str,
    reanswer_task_description: str,
    initial_answer: str,
    task_file_path: str,
    base_task_id: str,
    max_reanswer_attempts: int = 1,
    verify_max_turns: Optional[int] = None,
    unparseable_verdict: str = "correct",
    candidate_answers: Optional[List[str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Run verify -> (incorrect) reanswer -> re-verify until a ``correct`` verdict or the
    re-answer budget is exhausted. Returns ``(final_answer, metadata)``.

    ``initial_answer`` is the main attempt's answer; ``question`` is used by the verifier;
    ``reanswer_task_description`` is the pristine task prompt re-run on a re-answer.
    """
    current_answer = initial_answer
    records: List[Dict[str, Any]] = []
    reanswer_used = 0
    verified = False
    final_verdict = "not_run"

    while True:
        # Budget guard at loop top: a re-answer just produced `current_answer`; if the
        # budget is spent we keep it (un-verified) rather than burn another verify round.
        if reanswer_used > 0 and reanswer_used >= max_reanswer_attempts:
            final_verdict = "budget_exhausted_unverified"
            break

        verify = await _run_verify(
            evaluator,
            question=question,
            candidate_answer=current_answer,
            task_file_path=task_file_path,
            task_id=f"{base_task_id}_sv-verify-{sum(1 for r in records if r['role'] == 'verifier') + 1}",
            verify_max_turns=verify_max_turns,
            unparseable_verdict=unparseable_verdict,
            candidate_answers=candidate_answers,
        )
        records.append(verify["record"])
        final_verdict = verify["verdict"]
        if verify["verdict"] == "correct":
            verified = True
            break
        if reanswer_used >= max_reanswer_attempts:
            break

        reanswer_used += 1
        reanswer = await _run_reanswer(
            evaluator,
            reanswer_task_description=reanswer_task_description,
            task_file_path=task_file_path,
            task_id=f"{base_task_id}_sv-reanswer-{reanswer_used}",
        )
        records.append(reanswer["record"])
        if reanswer["answer"] and reanswer["answer"] != FORMAT_ERROR_MESSAGE:
            current_answer = reanswer["answer"]
        elif reanswer_used >= max_reanswer_attempts:
            # last re-answer failed to produce an answer: keep the previous candidate
            break

    metadata = {
        "enabled": True,
        "candidate_hints_used": len(candidate_answers or []),
        "initial_answer": initial_answer,
        "final_answer": current_answer,
        "final_verdict": final_verdict,
        "verified": verified,
        "answer_changed": current_answer != initial_answer,
        "reanswer_attempts_used": reanswer_used,
        "max_reanswer_attempts": max_reanswer_attempts,
        "records": records,
    }
    return current_answer, metadata
