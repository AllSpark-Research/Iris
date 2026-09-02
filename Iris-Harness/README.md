# Iris-Harness

The evaluation harness behind the numbers in the [Iris](https://github.com/AllSpark-Research/Iris)
report. A single-agent ReAct loop with two tools — `web_search` and
`scrape_website` — plus the context-management strategies, the four benchmarks,
and the graders.

A published score belongs to the agent *and* its harness: turn limits, context
handling and judge all move a benchmark number by several points. This is the
harness we ran, released so those numbers can be checked and so other systems
can be measured on the same footing.

## Install

```bash
uv sync
cp .env.example .env     # then fill in the keys
```

Python ≥ 3.12. `.env` needs, at minimum, a `SERPER_API_KEY`, a summary model
for `scrape_website`, and a judge model — see the comments in `.env.example`.

## Prepare the benchmarks

```bash
uv run python data/prepare_data.py
```

Downloads BrowseComp, BrowseComp-ZH, DeepSearchQA and HLE (text-only) from
their official sources and writes them to `data/<name>/standardized_data.jsonl`.
Nothing is redistributed here: two of the four are canary-encrypted and one is
gated. See [`data/README.md`](data/README.md) — it also documents the HLE prompt
suffix and the DeepSearchQA F1 metric, both of which you need to match to
reproduce our numbers.

## Run

The harness never launches a model. Serve the weights yourself — SGLang, vLLM,
anything OpenAI-compatible — and point it at the endpoint:

```bash
bash scripts/run_eval.sh \
  --base-url http://127.0.0.1:21234/v1 \
  --llm-config iris-mini \
  --benchmarks "browsecomp:0:1" "browsecomp_zh:0:1" \
                "deepsearchqa:0:1" "hle-text-2158:0:1" \
  --context-discard-threshold 131072 \
  --max-concurrent 32
```

A benchmark spec is `name:aveM:passN`. `ave3` runs three independent pass@1
evaluations and averages them; `pass3` samples three answers per question in
one run. Set the other field to `0`. Results land in
`logs/<model-tag>/<benchmark>/`, one directory per run, with per-task traces,
an `accuracy.txt`, and the resolved Hydra config.

`bash scripts/run_eval.sh --help` lists every flag. Start with
`--max-tasks 2 --max-concurrent 2` to check your keys and endpoint before
committing to a 2,158-question run.

## Context management

Long-horizon search runs out of context before a hard question is resolved, so
every serious system carries some mechanism for it. The harness implements
three, independently switchable:

| Flag | Mechanism |
| --- | --- |
| `--context-discard-threshold N` | **discard-all** — once the model's prompt crosses *N* tokens, reset the conversation to the opening question alone and continue from there. Every later turn, including the model's own reasoning, is dropped from what it sees; the full trace is still written to disk |
| `--context-compress-limit N` | **retry** — when an episode ends without a parseable answer, restart it from scratch up to *N* times, carrying forward a short summary of what was already ruled out (`--no-retry-summary` for a clean restart). A wrong answer is never retried, only a missing one |
| `--keep-tool-result K` | **recency-K** — send only the last *K* tool results in full and fold the earlier ones into placeholders; thinking and tool calls are always kept. `-1` keeps everything, `0` folds everything |

These compose. The report's four regimes are:

| Regime | Flags |
| --- | --- |
| w/o | *(defaults — no context management)* |
| retry | `--context-compress-limit 5` |
| discard-all | `--context-discard-threshold 131072` |
| discard-all + retry | `--context-discard-threshold 131072 --context-compress-limit 5` |

Everything else is held fixed across regimes and models: a 256K context window,
one tool set, one turn cap (600, from the agent config; `--max-turns` overrides
it), and one judge. `discard-all` is the headline setting in the report even
where adding `retry` scores higher.