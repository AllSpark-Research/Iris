# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""DeepSearchQA official metrics (Fully Correct / Fully Incorrect /
Correct-with-Extraneous / F1), computed from persisted ``eval_details``.

The judge (``verify_answer_deepsearchqa`` in eval_utils.py) already writes
``num_correct / num_expected / num_excessive`` into each attempt's
``eval_details`` in ``benchmark_results.jsonl``, so these metrics are derived
WITHOUT re-running the judge.

Methodology (identical to
``benchmarks/check_progress/check_progress_deepsearchqa.py``, the established
"official Google DeepSearchQA" implementation):
  * per question, use the FIRST attempt that has eval_details (pass@1 semantics)
  * TP = num_correct, FN = num_expected - num_correct, FP = num_excessive
  * precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2PR/(P+R)  (0 on 0-denom)
  * report the macro average (mean of per-question F1) plus the 3 category rates

This module is dependency-free (only ``json``/stdlib) so it can be imported from
the eval workflow (common_benchmark.py) without pulling in the agent stack.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Stable marker so the txt section can be detected / replaced idempotently.
METRICS_HEADER = "=== DeepSearchQA Metrics (Official) ==="


def calculate_deepsearchqa_metrics(results_file: Union[str, Path]) -> Dict[str, Any]:
    """Compute the 4 official DeepSearchQA metrics from a benchmark_results.jsonl.

    Returns a dict with keys: num_valid, fully_correct, fully_incorrect,
    correct_with_extraneous, pct_fully_correct, pct_fully_incorrect,
    pct_correct_with_extraneous, avg_f1. ``num_valid`` is 0 when nothing usable.
    """
    results = []
    try:
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    except Exception as e:
        print(f"Warning: could not read {results_file}: {e}")
        return {"num_valid": 0}

    num_valid = 0
    num_total = len(results)
    num_fully_correct = 0
    num_fully_incorrect = 0
    num_correct_with_extraneous = 0
    f1_list = []

    for result in results:
        if result.get("status") != "success":
            continue
        # Use the FIRST attempt that carries eval_details (pass@1 semantics).
        for attempt in result.get("attempts", []):
            details = attempt.get("eval_details")
            if not details:
                continue

            num_correct = details.get("num_correct", 0)
            num_expected = details.get("num_expected", 0)
            num_excessive = details.get("num_excessive", 0)

            true_positives = num_correct
            false_negatives = num_expected - num_correct
            false_positives = num_excessive

            precision = 0.0
            if (true_positives + false_positives) > 0:
                precision = true_positives / (true_positives + false_positives)

            recall = 0.0
            if (true_positives + false_negatives) > 0:
                recall = true_positives / (true_positives + false_negatives)

            f1 = 0.0
            if (precision + recall) > 0:
                f1 = 2 * (precision * recall) / (precision + recall)

            f1_list.append(f1)

            all_expected_correct = num_correct == num_expected
            has_extraneous = num_excessive > 0
            if all_expected_correct and not has_extraneous:
                num_fully_correct += 1
            elif num_correct == 0:
                num_fully_incorrect += 1
            elif all_expected_correct and has_extraneous:
                num_correct_with_extraneous += 1

            num_valid += 1
            break  # Only the first attempt with details per question.

    if num_valid == 0:
        return {"num_valid": 0, "num_total": num_total}

    return {
        "num_valid": num_valid,
        "num_total": num_total,
        "fully_correct": num_fully_correct,
        "fully_incorrect": num_fully_incorrect,
        "correct_with_extraneous": num_correct_with_extraneous,
        "pct_fully_correct": num_fully_correct / num_valid,
        "pct_fully_incorrect": num_fully_incorrect / num_valid,
        "pct_correct_with_extraneous": num_correct_with_extraneous / num_valid,
        "avg_f1": sum(f1_list) / len(f1_list),
    }


def format_metrics_block(metrics: Dict[str, Any]) -> str:
    """Render the metrics dict as the txt section (starting with METRICS_HEADER)."""
    n = metrics.get("num_valid", 0)
    total = metrics.get("num_total", n)
    if n == 0:
        return (
            f"\n{METRICS_HEADER}\n"
            f"num_valid: 0 / {total} (no attempts with eval_details found)\n"
        )
    # These percentages are over num_valid, i.e. the questions the judge actually
    # returned per-item details for -- NOT over the full benchmark the way pass@k
    # is. When the two differ, questions whose judge call failed are missing from
    # F1 entirely, which inflates it; say so rather than let the numbers be
    # compared as though they shared a denominator.
    shortfall = (
        ""
        if n == total
        else (
            f"WARNING: {total - n} of {total} questions have no eval_details and are\n"
            f"         excluded from the metrics below, while pass@k counts all {total}.\n"
            "         Re-run those questions before reporting F1.\n"
        )
    )
    return (
        f"\n{METRICS_HEADER}\n"
        f"num_valid: {n} / {total}  (metrics below are over num_valid)\n"
        f"{shortfall}"
        f"Fully Correct:         {metrics['pct_fully_correct'] * 100:.2f}%  "
        f"({metrics['fully_correct']} items)\n"
        f"Fully Incorrect:       {metrics['pct_fully_incorrect'] * 100:.2f}%  "
        f"({metrics['fully_incorrect']} items)\n"
        f"Correct w/ Extraneous: {metrics['pct_correct_with_extraneous'] * 100:.2f}%  "
        f"({metrics['correct_with_extraneous']} items)\n"
        f"F1 Score:              {metrics['avg_f1'] * 100:.2f}%\n"
    )


# Matches the whole metrics section (header through the F1 line / blank), so a
# prior block can be stripped before re-appending — makes writes idempotent.
_BLOCK_RE = re.compile(
    r"\n*" + re.escape(METRICS_HEADER) + r".*?(?=\n===|\Z)",
    re.DOTALL,
)


def append_metrics_to_txt(
    results_file: Union[str, Path], accuracy_file: Union[str, Path]
) -> Optional[Dict[str, Any]]:
    """Compute metrics from ``results_file`` and append the section to
    ``accuracy_file``, idempotently (any existing section is removed first).

    Returns the metrics dict, or None if there was nothing to write / no txt.
    """
    metrics = calculate_deepsearchqa_metrics(results_file)

    acc_path = Path(accuracy_file)
    if not acc_path.exists():
        print(f"Warning: accuracy file not found, skipping: {acc_path}")
        return None

    existing = acc_path.read_text(encoding="utf-8")
    # Strip any previously appended block, then re-append the fresh one.
    cleaned = _BLOCK_RE.sub("", existing).rstrip("\n") + "\n"
    block = format_metrics_block(metrics)
    acc_path.write_text(cleaned + block, encoding="utf-8")

    if metrics.get("num_valid", 0) > 0:
        print(
            f"DeepSearchQA F1: {metrics['avg_f1'] * 100:.2f}% "
            f"(fully_correct={metrics['pct_fully_correct'] * 100:.2f}%, "
            f"n={metrics['num_valid']}) -> {acc_path.name}"
        )
    return metrics
