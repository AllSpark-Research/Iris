# Benchmark data

No benchmark data is committed here. BrowseComp and BrowseComp-ZH are published
canary-encrypted so that crawlers cannot ingest them, and HLE is gated —
redistributing plaintext copies would contaminate the benchmarks for everyone.
Fetch them from the official sources instead:

```bash
uv run python data/prepare_data.py                 # all four
uv run python data/prepare_data.py browsecomp      # just one
```

| Benchmark | Questions | Source | Metric |
| --- | --- | --- | --- |
| `browsecomp` | 1,266 | [openai/simple-evals](https://github.com/openai/simple-evals) | accuracy |
| `browsecomp_zh` | 289 | [PALIN2018/BrowseComp-ZH](https://huggingface.co/datasets/PALIN2018/BrowseComp-ZH) | accuracy |
| `deepsearchqa` | 900 | [google/deepsearchqa](https://huggingface.co/datasets/google/deepsearchqa) | F1 |
| `hle-text-2158` | 2,158 | [cais/hle](https://huggingface.co/datasets/cais/hle) | accuracy |

HLE is gated: accept its terms on the dataset page and run `hf auth login`
before preparing it. The 2,158 are the text-only subset — the 342 questions
that carry an image are excluded, since this harness has no vision tools.

The script asserts the row count of every benchmark, so an upstream revision
fails loudly rather than quietly changing what you are measuring.

## Two protocol details

**HLE questions get a suffix.** Every HLE question is appended with

> Please search for relevant background information before answering, and
> explore multiple possible approaches instead of rushing to a final answer.

HLE is written as a closed-book exam, and without this the agent frequently
answers from memory and never searches at all. It is applied uniformly to all
2,158 questions, and `prepare_data.py` reproduces it. The other three
benchmarks are used verbatim.

**DeepSearchQA is scored with F1.** Its answers are sets, and F1 over the set
is the metric its authors report. Grading it as exact-match accuracy produces a
substantially lower and non-comparable number.

## Bringing your own data

Any benchmark is a directory holding `standardized_data.jsonl`, one JSON object
per line, plus a `conf/benchmark/<name>.yaml` pointing at it:

```json
{"task_id": "0", "task_question": "...", "file_name": "",
 "ground_truth": "...", "metadata": {"dataset": "my-benchmark"}}
```

`file_name` must be empty — the harness evaluates text-only benchmarks and will
refuse a task that names an attachment. `metadata` is free-form and is carried
through to the trace; `answer_type` is the one key the graders read, and only
for DeepSearchQA.
