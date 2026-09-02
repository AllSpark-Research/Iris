#!/usr/bin/env python
# Copyright (c) 2026 AllSpark Research.
# This source code is licensed under the Apache 2.0 License.

"""
Download the four benchmarks and write them in the harness's record format.

    uv run python data/prepare_data.py                 # all four
    uv run python data/prepare_data.py browsecomp deepsearchqa   # a subset

Each benchmark lands in ``data/<name>/standardized_data.jsonl``, one JSON
object per line:

    {"task_id": str, "task_question": str, "file_name": "",
     "ground_truth": str, "metadata": {...}}

The upstream sources are the official ones. BrowseComp and BrowseComp-ZH ship
XOR-encrypted against a per-row canary so that crawlers cannot ingest them; we
decrypt with the same scheme OpenAI publishes in ``simple-evals``. HLE is a
gated dataset: accept its terms at https://huggingface.co/datasets/cais/hle and
run ``hf auth login`` first.

Row counts are asserted, so a silently changed upstream fails loudly instead of
producing a differently-sized benchmark.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Iterator

DATA_ROOT = Path(__file__).resolve().parent

# Appended to every HLE question. HLE is a closed-book exam by construction, so
# without this nudge the agent often answers from memory and never searches.
# It is part of our evaluation protocol: reproduce it or the numbers will not
# line up.
HLE_SEARCH_SUFFIX = (
    "\nPlease search for relevant background information before answering, "
    "and explore multiple possible approaches instead of rushing to a final answer."
)

BROWSECOMP_CSV = (
    "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
)


# ── canary decryption (OpenAI simple-evals scheme) ──────────────────────────
def _derive_key(password: str, length: int) -> bytes:
    key = hashlib.sha256(password.encode()).digest()
    return key * (length // len(key)) + key[: length % len(key)]


def _decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = _derive_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode()


def _load_hf(repo: str, split: str, config: str | None = None):
    try:
        from datasets import load_dataset
    except ImportError:  # pragma: no cover
        sys.exit("`datasets` is required: uv sync, or pip install datasets")
    return load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)


# ── per-benchmark builders ──────────────────────────────────────────────────
def build_browsecomp() -> Iterator[dict]:
    """OpenAI BrowseComp — 1,266 English questions, canary-encrypted CSV."""
    raw = urllib.request.urlopen(BROWSECOMP_CSV, timeout=120).read().decode()
    for i, row in enumerate(csv.DictReader(io.StringIO(raw))):
        canary = row["canary"]
        yield {
            "task_id": str(i),
            "task_question": _decrypt(row["problem"], canary),
            "file_name": "",
            "ground_truth": _decrypt(row["answer"], canary),
            "metadata": {
                "problem_topic": row.get("problem_topic", ""),
                "dataset": "browsecomp",
            },
        }


def build_browsecomp_zh() -> Iterator[dict]:
    """BrowseComp-ZH — 289 Chinese questions, canary-encrypted."""
    for i, row in enumerate(_load_hf("PALIN2018/BrowseComp-ZH", "test")):
        canary = row["canary"]
        yield {
            "task_id": i,
            # A handful of upstream rows carry stray leading/trailing spaces;
            # strip so the judge compares the answer, not the whitespace.
            "task_question": _decrypt(row["Question"], canary).strip(),
            "filename": "",
            "ground_truth": _decrypt(row["Answer"], canary).strip(),
            "metadata": {
                "topic": _decrypt(row["Topic"], canary),
                "canary": canary,
            },
        }


def build_deepsearchqa() -> Iterator[dict]:
    """DeepSearchQA — 900 questions from Google DeepMind, scored with F1."""
    for i, row in enumerate(_load_hf("google/deepsearchqa", "eval", config="deepsearchqa"), 1):
        yield {
            "task_id": f"deepsearchqa_{i}",
            "task_question": row["problem"],
            "file_name": "",
            # Four "Set Answer" questions have the empty set as their answer and
            # arrive as null; the expected response is the literal "None".
            "ground_truth": "None" if row["answer"] is None else row["answer"],
            "metadata": {
                "problem_category": row["problem_category"],
                "answer_type": row["answer_type"],
            },
        }


def build_hle_text() -> Iterator[dict]:
    """HLE text-only — the 2,158 of 2,500 questions that carry no image."""
    for row in _load_hf("cais/hle", "test"):
        if row.get("image"):
            continue
        yield {
            "task_id": row["id"],
            "task_question": row["question"] + HLE_SEARCH_SUFFIX,
            "file_name": "",
            "ground_truth": row["answer"],
            "metadata": {
                "answer_type": row.get("answer_type", ""),
                "author_name": row.get("author_name", ""),
                "rationale": row.get("rationale", ""),
                "rationale_image": row.get("rationale_image", ""),
                "raw_subject": row.get("raw_subject", ""),
                "category": row.get("category", ""),
                "canary": row.get("canary", ""),
            },
        }


BENCHMARKS: Dict[str, tuple[Callable[[], Iterator[dict]], int, str]] = {
    "browsecomp": (build_browsecomp, 1266, "openai/simple-evals"),
    "browsecomp_zh": (build_browsecomp_zh, 289, "PALIN2018/BrowseComp-ZH"),
    "deepsearchqa": (build_deepsearchqa, 900, "google/deepsearchqa"),
    "hle-text-2158": (build_hle_text, 2158, "cais/hle (gated)"),
}


def prepare(name: str, force: bool = False) -> None:
    builder, expected, source = BENCHMARKS[name]
    out_dir = DATA_ROOT / name
    out_file = out_dir / "standardized_data.jsonl"

    if out_file.exists() and not force:
        n = sum(1 for _ in out_file.open())
        print(f"  {name:16s} exists ({n} rows) — pass --force to rebuild")
        return

    print(f"  {name:16s} fetching from {source} ...")
    records = list(builder())
    if len(records) != expected:
        raise SystemExit(
            f"  {name}: got {len(records)} rows, expected {expected}. "
            "The upstream dataset changed; check before evaluating."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  {name:16s} wrote {len(records)} rows -> {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "benchmarks",
        nargs="*",
        choices=list(BENCHMARKS),
        help="which benchmarks to prepare (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="rebuild existing files")
    args = parser.parse_args()

    for name in args.benchmarks or list(BENCHMARKS):
        prepare(name, force=args.force)


if __name__ == "__main__":
    main()
