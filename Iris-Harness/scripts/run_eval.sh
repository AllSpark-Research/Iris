#!/usr/bin/env bash
# Run one or more benchmarks against an already-served model.
#
# The harness never launches the model: point --base-url at any
# OpenAI-compatible endpoint (SGLang, vLLM, or a hosted API) and this script
# drives benchmarks/common_benchmark.py once per benchmark, per repeat.
#
#   bash scripts/run_eval.sh \
#     --base-url http://127.0.0.1:21234/v1 \
#     --llm-config iris-mini \
#     --benchmarks "browsecomp:0:1" "hle-text-2158:3:0" \
#     --context-discard-threshold 131072
#
# See README.md for the exact settings behind every number in the report.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# ── defaults ────────────────────────────────────────────────────────────────
BASE_URL=""
LLM_CONFIG="iris-mini"
AGENT_CONFIG="serper_search_agent"
MODEL_TAG=""
BENCHMARKS=()
MAX_CONCURRENT=32
MAX_TASKS=""
MAX_CONTEXT_LENGTH=""

# context management (see README "Context management")
KEEP_TOOL_RESULT=-1            # recency-K: -1 keep all, 0 fold all, N keep last N
CONTEXT_DISCARD_THRESHOLD=0    # discard-all: token threshold, 0 disables
CONTEXT_COMPRESS_LIMIT=0       # retry: max restarts of an answerless episode, 0 disables
RETRY_WITH_SUMMARY=1           # carry a failure summary into the retry

ANSWER_MODE="boxed"
REASONING_CONTENT_MODE="context"
TOOL_CALL_MODE="native_fc"
MAX_TURNS=""                   # empty = use the agent config's value

# self-verification (off by default)
SELF_VERIFICATION=0
SV_MAX_REANSWER=1
SV_VERIFY_MAX_TURNS=""
SV_CANDIDATE_HINTS=0

OUTPUT_ROOT="logs"

usage() {
    sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
    cat <<'EOF'

Required:
  --base-url <url>                 OpenAI-compatible endpoint, e.g. http://host:21234/v1
  --benchmarks <spec> [<spec>...]  "name:aveM:passN" — aveM runs M independent
                                   pass@1 evals and averages them; passN runs a
                                   single pass@N. Set the other field to 0.
                                   Available: browsecomp, browsecomp_zh,
                                   deepsearchqa, hle-text-2158

Model / agent:
  --llm-config <name>              conf/llm/*.yaml        (default: iris-mini)
  --agent-config <name>            conf/agent/*.yaml      (default: serper_search_agent)
  --model-tag <string>             label for the output directory
  --max-context-length <N>         override the config's context limit
  --max-turns <N>                  cap the agent's tool-use turns (default: agent config)
  --answer-mode <boxed|direct>     must match how the model was trained (default: boxed)
  --reasoning-content-mode <mode>  context | preserve | log_only | discard (default: context)
  --tool-call-mode <mode>          native_fc | mcp_xml    (default: native_fc)

Context management:
  --keep-tool-result <N>           recency-K; -1 keep all, 0 fold all, N keep last N
  --context-discard-threshold <N>  discard-all trigger in tokens; 0 disables
  --context-compress-limit <N>     retry budget per question when an episode ends
                                   with no parseable answer; 0 disables
  --no-retry-summary               retry without carrying the failure summary

Execution:
  --max-concurrent <N>             concurrent tasks (default: 32)
  --max-tasks <N>                  cap tasks per benchmark (smoke tests)
  --output-root <dir>              where run directories go (default: logs)

Self-verification (inference-time, off by default):
  --self-verification              verify -> reanswer -> re-verify after the answer
  --sv-max-reanswer <N>            clean re-answers on a confident "incorrect" (default: 1)
  --sv-verify-max-turns <N>        cap the verifier's search turns
  --sv-candidate-hints             show the verifier the mid-run \boxed{} candidates
EOF
}

# ── CLI ─────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-url)                  BASE_URL="$2"; shift 2 ;;
        --llm-config)                LLM_CONFIG="$2"; shift 2 ;;
        --agent-config)              AGENT_CONFIG="$2"; shift 2 ;;
        --model-tag)                 MODEL_TAG="$2"; shift 2 ;;
        --max-context-length)        MAX_CONTEXT_LENGTH="$2"; shift 2 ;;
        --max-turns)                 MAX_TURNS="$2"; shift 2 ;;
        --answer-mode)               ANSWER_MODE="$2"; shift 2 ;;
        --reasoning-content-mode)    REASONING_CONTENT_MODE="$2"; shift 2 ;;
        --tool-call-mode)            TOOL_CALL_MODE="$2"; shift 2 ;;
        --keep-tool-result)          KEEP_TOOL_RESULT="$2"; shift 2 ;;
        --context-discard-threshold) CONTEXT_DISCARD_THRESHOLD="$2"; shift 2 ;;
        --context-compress-limit)    CONTEXT_COMPRESS_LIMIT="$2"; shift 2 ;;
        --no-retry-summary)          RETRY_WITH_SUMMARY=0; shift ;;
        --max-concurrent)            MAX_CONCURRENT="$2"; shift 2 ;;
        --max-tasks)                 MAX_TASKS="$2"; shift 2 ;;
        --output-root)               OUTPUT_ROOT="$2"; shift 2 ;;
        --self-verification)         SELF_VERIFICATION=1; shift ;;
        --sv-max-reanswer)           SV_MAX_REANSWER="$2"; shift 2 ;;
        --sv-verify-max-turns)       SV_VERIFY_MAX_TURNS="$2"; shift 2 ;;
        --sv-candidate-hints)        SV_CANDIDATE_HINTS=1; shift ;;
        --benchmarks)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do BENCHMARKS+=("$1"); shift; done ;;
        -h|--help)                   usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

[[ -z "${BASE_URL}" ]]        && { echo "ERROR: --base-url is required" >&2; exit 1; }
[[ ${#BENCHMARKS[@]} -eq 0 ]] && { echo "ERROR: --benchmarks is required" >&2; exit 1; }
[[ -z "${MODEL_TAG}" ]]       && MODEL_TAG="${LLM_CONFIG}"

RUNNER=(uv run python)
command -v uv >/dev/null 2>&1 || RUNNER=(python)

# ── optional Hydra overrides ────────────────────────────────────────────────
MAXCTX_OVERRIDE=""
[[ -n "${MAX_CONTEXT_LENGTH}" ]] && MAXCTX_OVERRIDE="llm.max_context_length=${MAX_CONTEXT_LENGTH}"
MAX_TASKS_OVERRIDE=""
[[ -n "${MAX_TASKS}" ]] && MAX_TASKS_OVERRIDE="benchmark.execution.max_tasks=${MAX_TASKS}"
MAX_TURNS_OVERRIDE=""
[[ -n "${MAX_TURNS}" ]] && MAX_TURNS_OVERRIDE="agent.main_agent.max_turns=${MAX_TURNS}"

RETRY_WITH_SUMMARY_STR=$([[ "${RETRY_WITH_SUMMARY}" == "1" ]] && echo true || echo false)

SV_ARGS=""
if [[ "${SELF_VERIFICATION}" == "1" ]]; then
    SV_ARGS="agent.self_verification.enabled=true agent.self_verification.max_reanswer_attempts=${SV_MAX_REANSWER}"
    [[ -n "${SV_VERIFY_MAX_TURNS}" ]] && SV_ARGS+=" agent.self_verification.verification_max_turns=${SV_VERIFY_MAX_TURNS}"
    [[ "${SV_CANDIDATE_HINTS}" == "1" ]] && SV_ARGS+=" agent.self_verification.use_candidate_hints=true"
fi

BASE_DIR="${OUTPUT_ROOT}/${MODEL_TAG}"
mkdir -p "${BASE_DIR}"

cat <<EOF

════════════════════════════════════════════════════════════════
  endpoint    : ${BASE_URL}
  llm / agent : ${LLM_CONFIG} / ${AGENT_CONFIG}
  benchmarks  : ${BENCHMARKS[*]}
  context mgmt: keep_tool_result=${KEEP_TOOL_RESULT} \
discard_threshold=${CONTEXT_DISCARD_THRESHOLD} \
retry=${CONTEXT_COMPRESS_LIMIT}$([[ "${RETRY_WITH_SUMMARY}" == "1" ]] && echo " (+summary)")
  answer mode : ${ANSWER_MODE}
  output      : ${BASE_DIR}
════════════════════════════════════════════════════════════════
EOF

# ── one evaluation run ──────────────────────────────────────────────────────
run_single_eval() {
    local benchmark="$1" pass_at_k="$2" run_dir="$3" log_file="$4"
    mkdir -p "$(dirname "${log_file}")"

    "${RUNNER[@]}" benchmarks/common_benchmark.py \
        llm="${LLM_CONFIG}" \
        llm.base_url="${BASE_URL}" \
        ${MAXCTX_OVERRIDE} \
        llm.reasoning_content_mode="${REASONING_CONTENT_MODE}" \
        llm.tool_call_mode="${TOOL_CALL_MODE}" \
        agent="${AGENT_CONFIG}" \
        agent.keep_tool_result="${KEEP_TOOL_RESULT}" \
        agent.context_discard_threshold="${CONTEXT_DISCARD_THRESHOLD}" \
        agent.context_compress_limit="${CONTEXT_COMPRESS_LIMIT}" \
        agent.retry_with_summary="${RETRY_WITH_SUMMARY_STR}" \
        agent.answer_mode="${ANSWER_MODE}" \
        ${MAX_TURNS_OVERRIDE} \
        ${SV_ARGS} \
        benchmark="${benchmark}" \
        benchmark.execution.max_concurrent="${MAX_CONCURRENT}" \
        benchmark.execution.pass_at_k="${pass_at_k}" \
        ${MAX_TASKS_OVERRIDE} \
        hydra.run.dir="${run_dir}" \
        2>&1 | tee "${log_file}"

    return "${PIPESTATUS[0]}"
}

# pass@N: a single run that samples N answers per question
run_pass_eval() {
    local benchmark="$1" pass_n="$2"
    local run_dir="${BASE_DIR}/${benchmark}/pass${pass_n}"
    echo ""
    echo "  ── ${benchmark} · pass@${pass_n} → ${run_dir}"
    run_single_eval "${benchmark}" "${pass_n}" "${run_dir}" "${BASE_DIR}/${benchmark}_pass${pass_n}.log"
    local rc=$?
    local result_file
    result_file="$(find "${run_dir}" -name '*accuracy.txt' 2>/dev/null | head -1)"
    [[ -n "${result_file}" ]] && { echo "  result: "; cat "${result_file}"; }
    return "${rc}"
}

# ave@M: M independent pass@1 runs, averaged
run_ave_eval() {
    local benchmark="$1" ave_m="$2"
    local ave_dir="${BASE_DIR}/${benchmark}/ave${ave_m}"
    mkdir -p "${ave_dir}"
    echo ""
    echo "  ── ${benchmark} · ave@${ave_m} (${ave_m} parallel pass@1) → ${ave_dir}"

    local pids=() all_ok=true
    for i in $(seq 1 "${ave_m}"); do
        run_single_eval "${benchmark}" 1 "${ave_dir}/run_${i}" "${ave_dir}/run_${i}.log" &
        pids+=($!)
        sleep 2
    done
    for pid in "${pids[@]}"; do wait "${pid}" || all_ok=false; done

    "${RUNNER[@]}" benchmarks/evaluators/calculate_average_score.py "${ave_dir}" "${ave_m}" || all_ok=false
    [[ "${all_ok}" == true ]]
}

# ── drive ───────────────────────────────────────────────────────────────────
FAILED=()
for spec in "${BENCHMARKS[@]}"; do
    BM_NAME="$(cut -d: -f1 <<<"${spec}")"
    BM_AVE="$(cut -d: -f2 <<<"${spec}")"
    BM_PASS="$(cut -d: -f3 <<<"${spec}")"
    [[ -z "${BM_AVE}" ]]  && BM_AVE=0
    [[ -z "${BM_PASS}" ]] && BM_PASS=0

    echo ""
    echo "════ ${BM_NAME} ════"
    if [[ "${BM_AVE}" -gt 0 ]]; then
        run_ave_eval "${BM_NAME}" "${BM_AVE}" || FAILED+=("${BM_NAME}:ave${BM_AVE}")
    fi
    if [[ "${BM_PASS}" -gt 0 ]]; then
        run_pass_eval "${BM_NAME}" "${BM_PASS}" || FAILED+=("${BM_NAME}:pass${BM_PASS}")
    fi
    if [[ "${BM_AVE}" -eq 0 && "${BM_PASS}" -eq 0 ]]; then
        echo "  skipped: both aveM and passN are 0 in '${spec}'"
    fi
done

echo ""
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "All runs finished. Results under ${BASE_DIR}/"
else
    echo "Finished with failures: ${FAILED[*]}"
    exit 1
fi
