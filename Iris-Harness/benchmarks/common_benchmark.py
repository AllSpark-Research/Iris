# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

import asyncio
import gc
import json
import os
import random
import re
from abc import ABC
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()  # Load .env before Hydra resolves ${oc.env:...} interpolations

import hydra

# Import from the new modular structure
from evaluators.eval_utils import verify_answer_for_datasets
from omegaconf import DictConfig, OmegaConf
from src.core.pipeline import (
    create_pipeline_components,
    execute_task_pipeline,
)
from src.utils.prompt_utils import (
    FAILURE_EXPERIENCE_FOOTER,
    FAILURE_EXPERIENCE_HEADER,
    FAILURE_EXPERIENCE_ITEM,
    FORMAT_ERROR_MESSAGE,
    PIPELINE_ERROR_MESSAGE,
)
from src.utils.parsing_utils import strip_think_blocks


def _task_worker(task_dict, cfg_dict, evaluator_kwargs, worker_index=0):
    """
    Worker function to run a single task in a separate process.
    This function is called by ProcessPoolExecutor and must be at module level.

    Args:
        worker_index: Sequential index of this worker (0-based). Used for
            staggered-start delay so that concurrent workers don't all hit
            external search APIs simultaneously at launch.
    """
    import asyncio
    import time

    from omegaconf import OmegaConf

    # ── Staggered start ────────────────────────────────────────────────
    # Each worker sleeps ``worker_index * stagger_seconds`` (with jitter)
    # before doing any real work.  This spreads the initial burst of API
    # requests over a ramp-up window and significantly reduces thundering-
    # herd pressure on rate-limited gateways like the Bing search proxy.
    stagger_seconds = float(os.environ.get("BENCHMARK_STAGGER_SECONDS", "1.0"))
    if worker_index > 0 and stagger_seconds > 0:
        delay = worker_index * stagger_seconds + random.uniform(0, stagger_seconds)
        time.sleep(delay)

    # Reconstruct config in this process
    cfg = OmegaConf.create(cfg_dict)

    # Reconstruct task
    task = BenchmarkTask(
        task_id=task_dict["task_id"],
        task_question=task_dict["task_question"],
        ground_truth=task_dict["ground_truth"],
        file_path=task_dict.get("file_path"),
        metadata=task_dict.get("metadata", {}),
    )

    # Create evaluator in this process
    evaluator = GenericEvaluator(
        data_dir=evaluator_kwargs["data_dir"],
        benchmark_name=evaluator_kwargs["benchmark_name"],
        cfg=cfg,
        metadata_file=evaluator_kwargs.get("metadata_file", "standardized_data.jsonl"),
        task_id_field=evaluator_kwargs.get("task_id_field", "task_id"),
        question_field=evaluator_kwargs.get("question_field", "task_question"),
        ground_truth_field=evaluator_kwargs.get("ground_truth_field", "ground_truth"),
        file_name_field=evaluator_kwargs.get("file_name_field"),
    )

    # Run task in new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Set exception handler to suppress "Task exception was never retrieved" warnings
    def exception_handler(loop, context):
        # Suppress all asyncio internal warnings for cleaner output
        pass

    loop.set_exception_handler(exception_handler)

    try:
        result = loop.run_until_complete(evaluator.run_single_task(task))
        # Convert result to dict for serialization
        return asdict(result)
    finally:
        loop.close()


@dataclass
class BenchmarkTask:
    """Generic benchmark task data structure"""

    task_id: str
    task_question: str
    ground_truth: str
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    model_boxed_answer: str = ""
    status: str = "pending"  # pending, success, failed


@dataclass
class BenchmarkResult:
    """Generic benchmark evaluation result structure"""

    task_id: str
    task_question: str
    ground_truth: str
    file_path: Optional[str]
    status: str
    model_boxed_answer: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    final_judge_result: Optional[str] = None
    judge_type: Optional[str] = None
    log_file_path: Optional[str] = None
    # Pass@K support fields
    attempts: List[Dict[str, Any]] = field(default_factory=list)  # Store all attempts
    pass_at_k_success: bool = False  # Whether task passed using pass@k evaluation
    k_value: int = 1  # The k value used for this evaluation


class BenchmarkEvaluator(ABC):
    """Abstract base class for benchmark evaluators"""

    def __init__(self, data_dir: str, benchmark_name: str, cfg: DictConfig):
        """
        Initialize benchmark evaluator

        Args:
            data_dir: Path to benchmark data directory
            benchmark_name: Name of the benchmark
            cfg: The Hydra configuration object
        """
        self.data_dir = Path(data_dir)
        self.benchmark_name = benchmark_name
        self.cfg = cfg
        self.pass_at_k = cfg.benchmark.execution.get("pass_at_k", 1)
        self.tasks: List[BenchmarkTask] = []
        self.results: List[BenchmarkResult] = []

        # Format error tracking and retry configuration
        # Read from agent config as it's part of context management
        self.context_compress_limit = cfg.agent.get("context_compress_limit", 0)

        # Get LLM provider and model from the config object
        self.llm_provider = cfg.llm.provider
        self.llm_model = cfg.llm.model_name

        # Initialize pipeline components
        print("Initializing pipeline components...")
        (
            self.main_agent_tool_manager,
            self.sub_agent_tool_managers,
            self.output_formatter,
        ) = create_pipeline_components(cfg)
        print(
            f"Pipeline components initialized successfully! Using pass@{self.pass_at_k}"
        )

    def get_log_dir(self) -> Path:
        """Get the log directory for the current benchmark and model."""
        return Path(hydra.core.hydra_config.HydraConfig.get().run.dir)

    async def run_single_task(self, task: BenchmarkTask) -> BenchmarkResult:
        """
        Run inference for a single benchmark task with pass@k support

        Args:
            task: BenchmarkTask object

        Returns:
            BenchmarkResult object
        """
        print(f"Processing task {task.task_id} with pass@{self.pass_at_k}")

        result = BenchmarkResult(
            task_id=task.task_id,
            task_question=task.task_question,
            ground_truth=task.ground_truth,
            file_path=task.file_path,
            model_boxed_answer="",
            status="pending",
            metadata=task.metadata.copy(),
            k_value=self.pass_at_k,
        )

        logs_dir = self.get_log_dir()
        found_correct_answer = False

        # Print debug info about log directory
        print(f"  Current log directory: {logs_dir}")

        try:
            # Prepare task
            task_description, task_file_path = self.prepare_task_description(task)

            # Run up to k attempts (with early stopping when correct answer found)
            for attempt in range(1, self.pass_at_k + 1):
                print(f"  Attempt {attempt}/{self.pass_at_k} for task {task.task_id}")
                format_retry_count = 0

                # Check if log file exists for this specific attempt in current directory
                log_pattern = f"task_{task.task_id}_attempt-{attempt}_*.json"
                matching_logs = []

                # Search only in current log directory
                if logs_dir.exists():
                    dir_logs = sorted(list(logs_dir.glob(log_pattern)))
                    if dir_logs:
                        matching_logs.extend(dir_logs)

                if matching_logs:
                    # Sort by timestamp in filename to get the most recent
                    def extract_timestamp(file_path):
                        filename = file_path.name
                        # Extract timestamp from filename like: task_xxx_attempt-1_format-retry-0_2025-08-13-10-13-20.json
                        # The timestamp is the last part before .json
                        if "_" in filename and filename.endswith(".json"):
                            timestamp_part = filename.split("_")[-1].replace(
                                ".json", ""
                            )
                            # Convert timestamp to datetime for proper sorting
                            from datetime import datetime

                            return datetime.strptime(
                                timestamp_part, "%Y-%m-%d-%H-%M-%S"
                            )
                        return filename

                    matching_logs = sorted(matching_logs, key=extract_timestamp)

                attempt_result = {
                    "attempt_number": attempt,
                    "model_boxed_answer": "",
                    "status": "pending",
                    "log_file_path": None,
                    "final_judge_result": None,
                    "judge_type": None,
                    "is_correct": False,
                }

                # Try to load existing result for this attempt
                if matching_logs:
                    log_file = matching_logs[-1]
                    attempt_result["log_file_path"] = str(log_file)
                    print(
                        f"    Found existing log for attempt {attempt}: {log_file.name}"
                    )

                    match = re.search(r"retry-(\d+)", os.path.basename(str(log_file)))
                    if match:
                        format_retry_count = int(match.group(1))
                    else:
                        raise ValueError(
                            f"Failed to extract retry number from log file: {log_file}"
                        )

                    try:
                        with open(log_file) as f:
                            log_data = json.loads(f.read())
                            if log_data.get("status") == "success":
                                format_retry_count += 1
                            if log_data.get("final_boxed_answer"):
                                attempt_result["model_boxed_answer"] = log_data[
                                    "final_boxed_answer"
                                ]
                                attempt_result["status"] = log_data.get("status")
                                # Check if we already have a valid judge result in log.
                                # "ERROR" means the previous judge call failed — treat
                                # it as un-judged so the judge will be retried.
                                cached_judge = log_data.get("final_judge_result", "")
                                if cached_judge and cached_judge != "ERROR":
                                    attempt_result["final_judge_result"] = cached_judge
                                    attempt_result["judge_type"] = log_data.get(
                                        "judge_type", ""
                                    )
                                    attempt_result["is_correct"] = (
                                        cached_judge == "CORRECT"
                                    )
                                    # Load evaluation details if available
                                    if log_data.get("eval_details"):
                                        attempt_result["eval_details"] = log_data[
                                            "eval_details"
                                        ]
                                print(
                                    f"    Loaded existing result: {attempt_result['model_boxed_answer']}"
                                )
                    except Exception as e:
                        print(f"    Error loading log file {log_file}: {e}")

                # Run inference if no existing result or if we have a format error
                if (
                    not attempt_result["model_boxed_answer"]
                    or attempt_result["model_boxed_answer"] == FORMAT_ERROR_MESSAGE
                ):
                    # Try to get a valid response with format retry
                    print(f"TASK ID: {task.task_id}, ATTEMPT: {attempt}")

                    max_format_retries = self.context_compress_limit

                    # Track accumulated failure experiences for this attempt
                    # Start with the original task description
                    current_task_description = task_description
                    failure_experiences = []

                    # Resume: Recover failure experiences from previous retry logs
                    if format_retry_count > 0 and logs_dir.exists():
                        print(
                            f"    Resuming from retry {format_retry_count}, recovering previous failure experiences..."
                        )
                        for prev_retry in range(format_retry_count):
                            prev_log_pattern = f"task_{task.task_id}_attempt-{attempt}_format-retry-{prev_retry}_*.json"
                            prev_logs = sorted(list(logs_dir.glob(prev_log_pattern)))
                            if prev_logs:
                                prev_log_file = prev_logs[-1]  # Get the latest one
                                try:
                                    with open(
                                        prev_log_file, "r", encoding="utf-8"
                                    ) as f:
                                        prev_log_data = json.load(f)
                                        # Extract failure experience from trace_data
                                        trace_data = prev_log_data.get("trace_data", {})
                                        prev_failure_exp = trace_data.get(
                                            "failure_experience_summary"
                                        )
                                        if prev_failure_exp:
                                            failure_experiences.append(prev_failure_exp)
                                            print(
                                                f"      Recovered failure experience from retry {prev_retry}"
                                            )
                                except Exception as e:
                                    print(
                                        f"      Warning: Failed to load previous log {prev_log_file}: {e}"
                                    )

                        # Rebuild enhanced task description with recovered failure experiences
                        if failure_experiences:
                            current_task_description += FAILURE_EXPERIENCE_HEADER
                            for idx, exp in enumerate(failure_experiences, 1):
                                current_task_description += (
                                    FAILURE_EXPERIENCE_ITEM.format(
                                        attempt_number=idx,
                                        failure_summary=strip_think_blocks(exp),
                                    )
                                )
                            current_task_description += FAILURE_EXPERIENCE_FOOTER
                            print(
                                f"    Recovered {len(failure_experiences)} failure experience(s) from previous retries"
                            )

                    while format_retry_count <= max_format_retries:
                        try:
                            # Check if this is the final retry (no more chances after this)
                            is_final_retry = format_retry_count == max_format_retries

                            (
                                response,
                                final_boxed_answer,
                                log_file_path,
                                failure_experience_summary,
                            ) = await execute_task_pipeline(
                                cfg=self.cfg,
                                task_id=f"{task.task_id}_attempt-{attempt}_format-retry-{format_retry_count}",
                                task_file_name=task_file_path,
                                task_description=current_task_description,
                                main_agent_tool_manager=self.main_agent_tool_manager,
                                sub_agent_tool_managers=self.sub_agent_tool_managers,
                                output_formatter=self.output_formatter,
                                ground_truth=task.ground_truth,
                                log_dir=str(self.get_log_dir()),
                                is_final_retry=is_final_retry,
                            )

                            attempt_result["model_boxed_answer"] = (
                                final_boxed_answer if final_boxed_answer else ""
                            )
                            attempt_result["log_file_path"] = log_file_path

                            # The episode itself failed (tool server, endpoint,
                            # unhandled exception). Retrying the format would not
                            # help, and scoring it as a wrong answer would blame
                            # the model for our outage — record it as failed so it
                            # shows up in the error count instead.
                            if (
                                attempt_result["model_boxed_answer"]
                                == PIPELINE_ERROR_MESSAGE
                            ):
                                attempt_result["status"] = "failed"
                                attempt_result["error_message"] = (
                                    final_summary or PIPELINE_ERROR_MESSAGE
                                )
                                break

                            # Check for format error
                            if (
                                attempt_result["model_boxed_answer"]
                                == FORMAT_ERROR_MESSAGE
                            ):
                                format_retry_count += 1
                                if format_retry_count <= max_format_retries:
                                    # Use the model-generated failure experience summary
                                    print(
                                        f"    Format error detected, using model-generated failure summary for retry {format_retry_count}..."
                                    )

                                    if failure_experience_summary:
                                        failure_experiences.append(
                                            failure_experience_summary
                                        )

                                        # Build enhanced task description with accumulated failure experiences
                                        # Start fresh from original task_description each time
                                        current_task_description = task_description
                                        current_task_description += (
                                            FAILURE_EXPERIENCE_HEADER
                                        )
                                        for idx, exp in enumerate(
                                            failure_experiences, 1
                                        ):
                                            current_task_description += (
                                                FAILURE_EXPERIENCE_ITEM.format(
                                                    attempt_number=idx,
                                                    failure_summary=strip_think_blocks(exp),
                                                )
                                            )
                                        current_task_description += (
                                            FAILURE_EXPERIENCE_FOOTER
                                        )

                                        print(
                                            f"    Enhanced task description with {len(failure_experiences)} failure experience(s)"
                                        )
                                    else:
                                        print(
                                            "    No failure experience summary generated, retrying without enhancement..."
                                        )
                                    continue
                                else:
                                    # Exceeded format retry limit
                                    attempt_result["status"] = "success"
                                    attempt_result["model_boxed_answer"] = (
                                        f"{FORMAT_ERROR_MESSAGE} (after {max_format_retries} retries)"
                                    )
                                    attempt_result["error_message"] = (
                                        f"Exceeded format error retry limit ({max_format_retries})"
                                    )
                                    break
                            else:
                                # Got valid response, success
                                attempt_result["status"] = "success"
                                break

                        except Exception as e:
                            attempt_result["status"] = "failed"
                            attempt_result["error_message"] = str(e)
                            print(
                                f"    Error in attempt {attempt}, format retry {format_retry_count}: {e}"
                            )
                            break

                # ── Agent-side self-verification (verify → reanswer → re-verify) ──
                # Inference-time only, runs on a FRESH (non-resumed) answer BEFORE judging.
                # Reuses execute_task_pipeline for both verify and reanswer. Fail-open: any
                # error keeps the original answer. Disabled by default (cfg.agent.self_verification).
                _sv_cfg = self.cfg.agent.get("self_verification", None)
                _sv_answer = attempt_result["model_boxed_answer"]
                if (
                    _sv_cfg is not None
                    and _sv_cfg.get("enabled", False)
                    and _sv_answer
                    and not str(_sv_answer).startswith(FORMAT_ERROR_MESSAGE)
                    and attempt_result["final_judge_result"] is None
                ):
                    try:
                        from self_verification import (
                            load_candidate_answers,
                            run_self_verification,
                        )

                        # Optional: feed the main attempt's mid-loop boxed candidates to
                        # the verifier. Gated by use_candidate_hints; load_candidate_answers
                        # is itself fail-open (returns [] on any error).
                        _cand_hints = []
                        if bool(_sv_cfg.get("use_candidate_hints", False)):
                            _cand_hints = load_candidate_answers(
                                attempt_result.get("log_file_path", "") or "",
                                _sv_answer,
                                int(_sv_cfg.get("max_candidate_hints", 5)),
                            )

                        final_answer, sv_meta = await run_self_verification(
                            self,
                            question=task.task_question,
                            reanswer_task_description=task_description,
                            initial_answer=_sv_answer,
                            task_file_path=task_file_path,
                            base_task_id=f"{task.task_id}_attempt-{attempt}",
                            max_reanswer_attempts=int(
                                _sv_cfg.get("max_reanswer_attempts", 1)
                            ),
                            verify_max_turns=_sv_cfg.get("verification_max_turns", None),
                            unparseable_verdict=_sv_cfg.get(
                                "unparseable_verdict", "correct"
                            ),
                            candidate_answers=_cand_hints,
                        )
                        attempt_result["model_boxed_answer"] = final_answer
                        attempt_result["self_verification"] = sv_meta
                        print(
                            f"    🔎 Attempt {attempt}: self-verification "
                            f"verdict={sv_meta.get('final_verdict')} "
                            f"reanswers={sv_meta.get('reanswer_attempts_used')} "
                            f"{'(answer REPLACED)' if sv_meta.get('answer_changed') else '(answer kept)'}"
                        )
                    except Exception as e:
                        print(
                            f"    Error in self-verification for attempt {attempt}: {e} "
                            f"(keeping original answer)"
                        )

                # Perform LLM verification if we have an answer and haven't verified yet
                if (
                    attempt_result["model_boxed_answer"]
                    and attempt_result["final_judge_result"] is None
                    and task.ground_truth is not None
                ):
                    print(f"    Verifying answer for attempt {attempt}...")
                    try:
                        (
                            evaluation_result,
                            judge_type,
                            eval_details,
                        ) = await verify_answer_for_datasets(
                            benchmark_name=self.benchmark_name,
                            question=task.task_question,
                            target=task.ground_truth,
                            predicted_answer=attempt_result["model_boxed_answer"],
                            metadata=task.metadata,
                        )
                        attempt_result["final_judge_result"] = evaluation_result
                        attempt_result["judge_type"] = judge_type
                        attempt_result["is_correct"] = evaluation_result == "CORRECT"

                        # Store evaluation details (e.g., for DeepSearchQA metrics)
                        if eval_details:
                            attempt_result["eval_details"] = eval_details

                        # Update the log file with verification result
                        if attempt_result["log_file_path"]:
                            self._update_log_file_with_evaluation(
                                attempt_result["model_boxed_answer"],
                                attempt_result["log_file_path"],
                                evaluation_result,
                                judge_type,
                                eval_details,  # Pass eval_details to save in log file
                            )

                        if attempt_result["is_correct"]:
                            print(f"    ✅ Attempt {attempt}: CORRECT!")
                            found_correct_answer = True
                        else:
                            print(
                                f"    ❌ Attempt {attempt}: INCORRECT ({evaluation_result})"
                            )

                    except Exception as e:
                        print(f"    Error verifying attempt {attempt}: {e}")
                        attempt_result["final_judge_result"] = "ERROR"
                        attempt_result["judge_type"] = "error"
                        attempt_result["is_correct"] = False
                        # Persist ERROR state to log file so resume won't
                        # skip re-judging this attempt.
                        if attempt_result["log_file_path"]:
                            self._update_log_file_with_evaluation(
                                attempt_result["model_boxed_answer"],
                                attempt_result["log_file_path"],
                                "ERROR",
                                "error",
                            )

                elif attempt_result["is_correct"]:
                    print(f"    ✅ Attempt {attempt}: CORRECT (cached)")
                    found_correct_answer = True

                elif attempt_result["final_judge_result"]:
                    print(
                        f"    ❌ Attempt {attempt}: INCORRECT (cached: {attempt_result['final_judge_result']})"
                    )
                else:
                    print(f"    ⚠️  Attempt {attempt}: No valid answer to verify")

                result.attempts.append(attempt_result)

                # Update main result with the first successful attempt or best attempt so far
                if attempt == 1 or (
                    attempt_result["status"] == "success"
                    and not result.model_boxed_answer
                ):
                    result.model_boxed_answer = attempt_result["model_boxed_answer"]
                    result.log_file_path = attempt_result["log_file_path"]
                    result.status = attempt_result["status"]
                    if "error_message" in attempt_result:
                        result.error_message = attempt_result["error_message"]

                # Early stopping: if we found a correct answer, we can stop
                if found_correct_answer:
                    print(
                        f"    🎯 Found correct answer! Stopping early after {attempt} attempts."
                    )
                    break

        except Exception as e:
            result.error_message = str(e)
            result.status = "failed"
            print(f"Error processing task {task.task_id}: {e}")

        finally:
            result.pass_at_k_success = found_correct_answer

            # Set main result judge result based on pass@k outcome
            if found_correct_answer:
                result.final_judge_result = "PASS_AT_K_SUCCESS"
                result.judge_type = "pass_at_k"
            else:
                if result.ground_truth is None:
                    result.final_judge_result = "TEST_SET_MODE"
                else:
                    result.final_judge_result = "PASS_AT_K_FAILED"
                result.judge_type = "pass_at_k"

            print(f"Task {task.task_id} completed with {len(result.attempts)} attempts")
            if result.ground_truth is not None:
                print(
                    f"    Pass@{self.pass_at_k} result: {'✅ SUCCESS' if found_correct_answer else '❌ FAILED'}"
                )

        gc.collect()
        return result

    def _run_single_task_sync(self, task: BenchmarkTask) -> BenchmarkResult:
        """Sync wrapper for run_single_task to be used in threads"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Set exception handler to suppress "Task exception was never retrieved" warnings
        def exception_handler(loop, context):
            # Suppress all asyncio internal warnings for cleaner output
            pass

        loop.set_exception_handler(exception_handler)

        try:
            # Direct await is simpler and cleaner than gather for single task
            return loop.run_until_complete(self.run_single_task(task))
        finally:
            loop.close()

    def run_parallel_inference(
        self, tasks: List[BenchmarkTask], max_concurrent: int = 3
    ) -> List[BenchmarkResult]:
        """Run inference on multiple tasks in parallel using multiprocessing.

        Workers are started with a staggered delay (controlled by the env var
        ``BENCHMARK_STAGGER_SECONDS``, default ``1.0`` s per worker) so that
        external API gateways (Bing, Zhipu) are not overwhelmed by a burst of
        simultaneous first-requests from all workers at once.
        """
        stagger = float(os.environ.get("BENCHMARK_STAGGER_SECONDS", "1.0"))
        ramp_up = max_concurrent * stagger
        print(
            f"Running inference on {len(tasks)} tasks with max_concurrent={max_concurrent} "
            f"(multiprocessing, stagger={stagger}s, ramp-up≈{ramp_up:.0f}s)"
        )

        # Serialize config
        cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)

        # Shuffle tasks to avoid order bias and improve balancing
        shuffled_tasks = tasks.copy()
        random.shuffle(shuffled_tasks)

        # Prepare evaluator kwargs for worker processes
        evaluator_kwargs = {
            "data_dir": str(self.data_dir),
            "benchmark_name": self.benchmark_name,
        }
        # Add GenericEvaluator specific kwargs if available
        if hasattr(self, "metadata_file"):
            evaluator_kwargs["metadata_file"] = str(self.metadata_file.name)
        if hasattr(self, "task_id_field"):
            evaluator_kwargs["task_id_field"] = self.task_id_field
        if hasattr(self, "question_field"):
            evaluator_kwargs["question_field"] = self.question_field
        if hasattr(self, "ground_truth_field"):
            evaluator_kwargs["ground_truth_field"] = self.ground_truth_field
        if hasattr(self, "file_name_field"):
            evaluator_kwargs["file_name_field"] = self.file_name_field

        # Prepare serializable arguments for worker processes.
        # ``worker_index`` is ``idx % max_concurrent`` so that only the first
        # batch of workers (0 .. max_concurrent-1) incurs a staggered delay.
        # Later batches reuse freed slots and naturally inherit temporal spread.
        worker_args = []
        for idx, task in enumerate(shuffled_tasks):
            task_dict = {
                "task_id": task.task_id,
                "task_question": task.task_question,
                "ground_truth": task.ground_truth,
                "file_path": task.file_path,
                "metadata": task.metadata,
            }
            worker_args.append(
                (task_dict, cfg_dict, evaluator_kwargs, idx % max_concurrent)
            )

        # Use ProcessPoolExecutor for true parallelism (bypasses GIL)
        processed_results = []
        task_index_map = {
            task.task_id: (i, task) for i, task in enumerate(shuffled_tasks)
        }
        results_dict = {}  # Store results by task_id to maintain order

        executor = None
        try:
            executor = ProcessPoolExecutor(max_workers=max_concurrent)
            # Submit all tasks
            future_to_task_id = {}
            for args in worker_args:
                task_dict = args[0]  # First element is task_dict
                future = executor.submit(_task_worker, *args)
                future_to_task_id[future] = task_dict["task_id"]

            # Collect results as they complete
            from concurrent.futures import as_completed

            for future in as_completed(future_to_task_id):
                task_id = future_to_task_id[future]
                try:
                    result_dict = future.result()
                    # Reconstruct BenchmarkResult from dict
                    result = BenchmarkResult(**result_dict)
                    results_dict[task_id] = result
                    completed = len(results_dict)
                    print(
                        f"Progress: {completed}/{len(shuffled_tasks)} tasks completed"
                    )
                except Exception as e:
                    print(f"Exception in task {task_id}: {e}")
                    # Get original task for error result
                    _, original_task = task_index_map[task_id]
                    error_result = BenchmarkResult(
                        task_id=original_task.task_id,
                        task_question=original_task.task_question,
                        ground_truth=original_task.ground_truth,
                        file_path=original_task.file_path,
                        model_boxed_answer="",
                        status="failed",
                        metadata=original_task.metadata.copy(),
                        error_message=str(e),
                    )
                    results_dict[task_id] = error_result
        except KeyboardInterrupt:
            print("\n⚠️  Received interrupt signal, shutting down gracefully...")
            if executor:
                print("  Cancelling pending tasks and terminating worker processes...")
                # Cancel all pending futures
                for future in future_to_task_id:
                    future.cancel()

                # Forcefully terminate worker processes
                # Access internal processes and terminate them
                if hasattr(executor, "_processes") and executor._processes:
                    for pid, process in executor._processes.items():
                        try:
                            if process.is_alive():
                                print(f"    Terminating worker process {pid}...")
                                process.terminate()
                        except Exception as e:
                            print(
                                f"    Warning: Failed to terminate process {pid}: {e}"
                            )

                    # Give processes a short time to terminate gracefully
                    import time

                    time.sleep(0.5)

                    # Force kill any remaining processes
                    for pid, process in executor._processes.items():
                        try:
                            if process.is_alive():
                                print(f"    Force killing worker process {pid}...")
                                process.kill()
                        except Exception as e:
                            print(f"    Warning: Failed to kill process {pid}: {e}")

                # Shutdown executor without waiting for pending tasks
                executor.shutdown(wait=False, cancel_futures=True)
            print("  Shutdown complete.")
            raise
        finally:
            # Ensure executor is properly cleaned up
            if executor:
                try:
                    executor.shutdown(wait=True)
                except Exception:
                    pass  # Ignore errors during cleanup

        # Reconstruct results in original task order
        processed_results = [results_dict[task.task_id] for task in shuffled_tasks]

        # Sort results to maintain original task order
        task_id_to_index = {task.task_id: i for i, task in enumerate(tasks)}
        processed_results.sort(
            key=lambda r: task_id_to_index.get(r.task_id, len(tasks))
        )

        self.results = processed_results
        return processed_results

    def save_results(self, output_file: str) -> str:
        """Save evaluation results to JSONL file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for result in self.results:
                f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

        print(f"Results saved to {output_path}")
        return str(output_path)

    def evaluate_accuracy(self) -> float:
        """Evaluate pass@k accuracy (verification already done in run_single_task)"""
        if not self.results:
            print("No results to evaluate")
            return 0.0

        print(
            f"Calculating pass@{self.pass_at_k} accuracy for {len(self.results)} results..."
        )

        correct_count = 0
        total_count = 0

        for result in self.results:
            total_count += 1

            # Display task results
            print(f"\nTask {result.task_id}:")
            print(f"  Attempts: {len(result.attempts)}")
            if result.ground_truth is not None:
                print(
                    f"  Pass@{self.pass_at_k}: {'✅ SUCCESS' if result.pass_at_k_success else '❌ FAILED'}"
                )

            print("  " + "=" * 50)
            print(f"  Reference: {result.ground_truth}")
            print("  " + "=" * 50)

            if result.pass_at_k_success:
                correct_count += 1

        pass_at_k_accuracy = correct_count / total_count if total_count > 0 else 0.0

        print(f"\nPass@{self.pass_at_k} Final Results:")
        print(f"Tasks passed: {correct_count}/{total_count}")
        print(f"Pass@{self.pass_at_k} Accuracy: {pass_at_k_accuracy:.2%}")

        return pass_at_k_accuracy

    def evaluate_accuracy_all_k(self) -> Dict[int, float]:
        """Evaluate pass@1 through pass@k accuracy for all levels.

        Returns:
            Dictionary mapping each k value (1..pass_at_k) to its accuracy.
        """
        if not self.results:
            print("No results to evaluate")
            return {}

        k = self.pass_at_k
        total_count = len(self.results)
        accuracies: Dict[int, float] = {}

        for i in range(1, k + 1):
            correct = 0
            for result in self.results:
                # Check if any of the first i attempts is correct
                for attempt in result.attempts[:i]:
                    if attempt.get("is_correct", False):
                        correct += 1
                        break
            acc = correct / total_count if total_count > 0 else 0.0
            accuracies[i] = acc

        # Print summary
        print(f"\n{'='*60}")
        print(f"Pass@1 ~ Pass@{k} Accuracy  (n={total_count})")
        print(f"{'='*60}")
        for i in range(1, k + 1):
            bar = "█" * int(accuracies[i] * 40)
            print(f"  Pass@{i}: {accuracies[i]:7.2%}  {bar}")
        print(f"{'='*60}")

        return accuracies

    def _update_log_file_with_evaluation(
        self,
        model_boxed_answer: str,
        log_file_path: str,
        evaluation_result: str,
        judge_type: str,
        eval_details: Optional[Dict[str, Any]] = None,
    ):
        """Helper method to update log file with evaluation result"""
        try:
            log_file = Path(log_file_path)
            # Read existing data
            with open(log_file, "r", encoding="utf-8") as f:
                log_data = json.load(f)

            # Update with evaluation result
            log_data["final_boxed_answer"] = model_boxed_answer
            log_data["final_judge_result"] = evaluation_result
            log_data["judge_type"] = judge_type

            # Store evaluation details (e.g., for DeepSearchQA metrics)
            if eval_details:
                log_data["eval_details"] = eval_details

            # Write to a temporary file and then atomically replace
            temp_log_file = log_file.with_suffix(f"{log_file.suffix}.tmp")
            with open(temp_log_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)

            os.replace(temp_log_file, log_file)
            print(f"    Updated log file {log_file.name} with evaluation result.")
        except Exception as e:
            print(f"    Error updating log file {log_file_path}: {e}")


class GenericEvaluator(BenchmarkEvaluator):
    """Generic benchmark evaluator for JSONL format"""

    def __init__(
        self,
        data_dir: str,
        benchmark_name: str,
        cfg: DictConfig,
        metadata_file: str = "standardized_data.jsonl",
        task_id_field: str = "task_id",
        question_field: str = "task_question",
        ground_truth_field: str = "ground_truth",
        file_name_field: Optional[str] = "file_name",
    ):
        """
        Initialize generic evaluator

        Args:
            data_dir: Path to benchmark data directory
            benchmark_name: Name of the benchmark
            cfg: The Hydra configuration object
            metadata_file: Name of the metadata file
            task_id_field: Field name for task ID in the data
            question_field: Field name for task question in the data
            ground_truth_field: Field name for ground truth answer in the data
            file_name_field: Field name for file name in the data (optional)
            pass_at_k: Pass@K value for evaluation (default: 1)
        """
        super().__init__(data_dir=data_dir, benchmark_name=benchmark_name, cfg=cfg)
        self.metadata_file = self.data_dir / metadata_file
        self.task_id_field = task_id_field
        self.question_field = question_field
        self.ground_truth_field = ground_truth_field
        self.file_name_field = file_name_field
        self.tasks: List[BenchmarkTask] = []
        self.results: List[BenchmarkResult] = []

    def load_tasks(self, limit: Optional[int] = None) -> List[BenchmarkTask]:
        """
        Load benchmark tasks from standardized_data.jsonl

        Args:
            limit: Maximum number of tasks to load (None for all)

        Returns:
            List of BenchmarkTask objects
        """
        print(f"Loading tasks from {self.metadata_file}")

        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")

        tasks = []
        unparsable: List[Tuple[int, str]] = []
        with open(self.metadata_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break

                try:
                    data = json.loads(line.strip())

                    # Extract file path if specified
                    file_path = None
                    if self.file_name_field and self.file_name_field in data:
                        file_path = data[self.file_name_field]

                    # Create metadata dict with all remaining fields
                    metadata = {
                        k: v
                        for k, v in data.items()
                        if k
                        not in [
                            self.task_id_field,
                            self.question_field,
                            self.ground_truth_field,
                            self.file_name_field,
                        ]
                    }

                    task = BenchmarkTask(
                        task_id=data[self.task_id_field],
                        task_question=data[self.question_field],
                        ground_truth=data[self.ground_truth_field],
                        file_path=file_path,
                        metadata=metadata,
                    )
                    tasks.append(task)

                except Exception as e:
                    # A skipped question silently shrinks the denominator, which
                    # turns a data problem into a wrong published score. Collect
                    # the failures and refuse to run rather than under-report.
                    unparsable.append((i + 1, str(e)))

        if unparsable:
            detail = "; ".join(f"line {n}: {msg}" for n, msg in unparsable[:5])
            more = f" (+{len(unparsable) - 5} more)" if len(unparsable) > 5 else ""
            raise ValueError(
                f"{len(unparsable)} unparsable record(s) in {self.metadata_file}: "
                f"{detail}{more}. Re-run data/prepare_data.py --force; evaluating a "
                "partially loaded benchmark would report the wrong denominator."
            )

        gc.collect()
        self.tasks = tasks
        print(f"Loaded {len(tasks)} tasks")
        return tasks

    def prepare_task_description(
        self, task: BenchmarkTask
    ) -> Tuple[str, Optional[str]]:
        """
        Prepare task description and file path for the agent

        Args:
            task: BenchmarkTask object

        Returns:
            Tuple of (task_description, task_file_path)
        """

        task_file_path = None
        if task.file_path:
            # Build complete file path: data directory + relative path
            full_file_path = self.data_dir / task.file_path
            # Convert to absolute path and resolve any symbolic links
            task_file_path = str(full_file_path.resolve())
        else:
            task_file_path = None

        # Return task question and file path
        return task.task_question, task_file_path


def _count_tool_stats_in_trace(trace_path: Path) -> tuple[int, int]:
    """Count total tool calls and tool turns in a single trace JSON file.

    Returns:
        (total_tool_calls, tool_turns) where tool_turns is the number of
        assistant messages that contain at least one tool call.
    """
    try:
        with open(trace_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return 0, 0

    total_calls = 0
    tool_turns = 0

    # main agent message history
    for msg in (
        data.get("main_agent_message_history", {}).get("message_history", [])
    ):
        if msg.get("role") == "assistant":
            tcs = msg.get("tool_calls")
            if tcs:
                total_calls += len(tcs)
                tool_turns += 1

    # sub-agent sessions
    for session in data.get("sub_agent_message_history_sessions", []):
        for msg in session.get("message_history", []):
            if msg.get("role") == "assistant":
                tcs = msg.get("tool_calls")
                if tcs:
                    total_calls += len(tcs)
                    tool_turns += 1

    return total_calls, tool_turns


def _append_tool_call_stats(results_path: Path, accuracy_file: str) -> None:
    """Read benchmark_results.jsonl, count tool calls and tool turns per
    attempt, and append avg stats (correct vs incorrect) to the accuracy file."""
    correct_calls: list[int] = []
    incorrect_calls: list[int] = []
    correct_turns: list[int] = []
    incorrect_turns: list[int] = []
    log_dir = results_path.parent

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                task = json.loads(line)
                for attempt in task.get("attempts", []):
                    log_rel = attempt.get("log_file_path")
                    if not log_rel:
                        continue
                    trace_path = log_dir / Path(log_rel).name
                    if not trace_path.exists():
                        # try the path as-is (may be relative from cwd)
                        trace_path = Path(log_rel)
                    n_calls, n_turns = _count_tool_stats_in_trace(trace_path)
                    if attempt.get("is_correct"):
                        correct_calls.append(n_calls)
                        correct_turns.append(n_turns)
                    else:
                        incorrect_calls.append(n_calls)
                        incorrect_turns.append(n_turns)
    except Exception as e:
        print(f"Warning: failed to compute tool call stats: {e}")
        return

    avg_calls_c = (
        sum(correct_calls) / len(correct_calls) if correct_calls else 0
    )
    avg_calls_i = (
        sum(incorrect_calls) / len(incorrect_calls) if incorrect_calls else 0
    )
    avg_turns_c = (
        sum(correct_turns) / len(correct_turns) if correct_turns else 0
    )
    avg_turns_i = (
        sum(incorrect_turns) / len(incorrect_turns) if incorrect_turns else 0
    )

    stats_line = (
        f"\navg_tool_calls_correct: {avg_calls_c:.2f} ({len(correct_calls)} attempts)\n"
        f"avg_tool_calls_incorrect: {avg_calls_i:.2f} ({len(incorrect_calls)} attempts)\n"
        f"avg_tool_turns_correct: {avg_turns_c:.2f}\n"
        f"avg_tool_turns_incorrect: {avg_turns_i:.2f}\n"
    )

    with open(accuracy_file, "a") as f:
        f.write(stats_line)

    print(
        f"Tool call stats: correct={avg_calls_c:.2f} ({len(correct_calls)}), "
        f"incorrect={avg_calls_i:.2f} ({len(incorrect_calls)})"
    )
    print(
        f"Tool turn stats: correct={avg_turns_c:.2f}, "
        f"incorrect={avg_turns_i:.2f}"
    )


# Authoritative step-log signal logged once per trace at main-loop exit when
# reached_max_turns is True (orchestrator: "Max Turns Reached / Context Limit
# Reached"). Its absence means the model stopped on its own (natural end).
_FORCED_TERMINATION_SIGNAL = "Max Turns Reached / Context Limit Reached"


def _detect_termination_type(trace_path: Path) -> str:
    """Classify how a trajectory terminated, from its step_logs.

    Returns one of:
      "forced_context_limit" - forced exit triggered by the context-limit break
      "forced_max_turns"     - forced exit after exhausting the turn budget
      "natural"              - the model stopped on its own (produced an answer)
      "unknown"              - trace file unreadable
    """
    try:
        with open(trace_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return "unknown"

    forced = False
    ctx_break = False
    for s in data.get("step_logs", []):
        sn = str(s.get("step_name", ""))
        if _FORCED_TERMINATION_SIGNAL in sn:
            forced = True
        # The context-limit break (ensure_summary_context -> False) logs a
        # main-agent turn-scoped "Context Limit Reached" right before breaking.
        # Distinguish it from the mid-trajectory LLM-level rollback ("🧠 LLM |
        # Context Limit Reached") by requiring the "Main Agent" + "Turn:" prefix.
        if "Context Limit Reached" in sn and "Turn:" in sn and "Main Agent" in sn:
            ctx_break = True

    if not forced:
        return "natural"
    return "forced_context_limit" if ctx_break else "forced_max_turns"


def _classify_answer_result(is_correct: Optional[bool], model_boxed_answer: str) -> str:
    """Bucket an attempt's outcome into correct / incorrect / no_boxed.

    "no_boxed" means the final summary never produced a \\boxed{} answer
    (FORMAT_ERROR_MESSAGE) — typically because the model kept calling tools
    instead of answering. Such attempts are always judged incorrect, so we
    separate them from genuine wrong answers.
    """
    ba = (model_boxed_answer or "").strip()
    if ba.startswith(FORMAT_ERROR_MESSAGE):
        return "no_boxed"
    return "correct" if is_correct else "incorrect"


def _append_error_stats(results_path: Path, accuracy_file: str) -> int:
    """Append an infrastructure-failure count to the accuracy file.

    A score is only meaningful if every question actually got an answer graded.
    Two things can break that quietly:

    * the episode failed before producing an answer (tool server, endpoint,
      unhandled exception) -- ``status == "failed"``;
    * the judge itself failed and returned ERROR / NOT_ATTEMPTED after its
      retries, which downstream is indistinguishable from "wrong".

    Both would otherwise be silently folded into the denominator and read as
    model failures. Count them here and return the total so the caller can warn.
    """
    from collections import Counter

    counts: Counter = Counter()
    total_attempts = 0
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                task = json.loads(line)
                for attempt in task.get("attempts", []):
                    total_attempts += 1
                    if attempt.get("status") == "failed":
                        counts["pipeline_error"] += 1
                    judged = str(attempt.get("final_judge_result") or "").upper()
                    if judged in ("ERROR", "NOT_ATTEMPTED"):
                        counts[f"judge_{judged.lower()}"] += 1
    except Exception as e:  # never let reporting break a finished run
        print(f"Warning: could not compute error stats: {e}")
        return 0

    n_bad = sum(counts.values())
    lines = ["", "=== Infrastructure Errors ===", f"total_attempts: {total_attempts}"]
    for key in ("pipeline_error", "judge_error", "judge_not_attempted"):
        lines.append(f"{key}: {counts.get(key, 0)}")
    if n_bad:
        lines.append(
            "WARNING: these attempts were graded as incorrect but never actually "
            "produced or received a verdict. Treat the score above as a lower bound "
            "and re-run them before reporting."
        )
    with open(accuracy_file, "a") as f:
        f.write("\n".join(lines) + "\n")
    return n_bad


def _append_termination_stats(results_path: Path, accuracy_file: str) -> None:
    """Append termination analysis to the accuracy file:

    1. Ratio of natural-end vs forced-summary attempts.
    2. For forced-summary attempts, the correct / incorrect / no_boxed split
       (no_boxed = model kept calling tools -> no \\boxed{} answer).

    A symmetric breakdown for natural-end attempts is also emitted for context.
    """
    log_dir = results_path.parent
    # term_type -> Counter of result category
    from collections import Counter

    term_counter: Counter = Counter()          # natural / forced_* / unknown
    breakdown: Dict[str, Counter] = {
        "natural": Counter(),
        "forced": Counter(),
        "unknown": Counter(),
    }

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                task = json.loads(line)
                for attempt in task.get("attempts", []):
                    log_rel = attempt.get("log_file_path")
                    trace_path = None
                    if log_rel:
                        trace_path = log_dir / Path(log_rel).name
                        if not trace_path.exists():
                            trace_path = Path(log_rel)
                    term = (
                        _detect_termination_type(trace_path)
                        if trace_path is not None
                        else "unknown"
                    )
                    term_counter[term] += 1

                    result = _classify_answer_result(
                        attempt.get("is_correct"),
                        attempt.get("model_boxed_answer", ""),
                    )
                    if term.startswith("forced"):
                        breakdown["forced"][result] += 1
                    elif term == "natural":
                        breakdown["natural"][result] += 1
                    else:
                        breakdown["unknown"][result] += 1
    except Exception as e:
        print(f"Warning: failed to compute termination stats: {e}")
        return

    total = sum(term_counter.values())
    if total == 0:
        return

    n_natural = term_counter.get("natural", 0)
    n_forced = (
        term_counter.get("forced_max_turns", 0)
        + term_counter.get("forced_context_limit", 0)
    )
    n_unknown = term_counter.get("unknown", 0)

    def _pct(n: int, d: int) -> str:
        return f"{(n / d * 100):.2f}%" if d else "0.00%"

    lines = ["\n=== Termination Analysis ===\n"]
    lines.append(f"total_attempts: {total}\n")
    lines.append(f"natural_end: {n_natural} ({_pct(n_natural, total)})\n")
    lines.append(f"forced_summary: {n_forced} ({_pct(n_forced, total)})\n")
    lines.append(
        f"  - by max_turns: {term_counter.get('forced_max_turns', 0)}\n"
    )
    lines.append(
        f"  - by context_limit: {term_counter.get('forced_context_limit', 0)}\n"
    )
    if n_unknown:
        lines.append(f"unknown_termination: {n_unknown} ({_pct(n_unknown, total)})\n")

    fb = breakdown["forced"]
    lines.append(f"\nforced_summary breakdown ({n_forced}):\n")
    lines.append(
        f"  - correct: {fb.get('correct', 0)} ({_pct(fb.get('correct', 0), n_forced)})\n"
    )
    lines.append(
        f"  - incorrect (wrong boxed answer): {fb.get('incorrect', 0)} "
        f"({_pct(fb.get('incorrect', 0), n_forced)})\n"
    )
    lines.append(
        f"  - no_boxed (kept calling tools): {fb.get('no_boxed', 0)} "
        f"({_pct(fb.get('no_boxed', 0), n_forced)})\n"
    )

    nb = breakdown["natural"]
    lines.append(f"\nnatural_end breakdown ({n_natural}):\n")
    lines.append(
        f"  - correct: {nb.get('correct', 0)} ({_pct(nb.get('correct', 0), n_natural)})\n"
    )
    lines.append(
        f"  - incorrect: {nb.get('incorrect', 0)} ({_pct(nb.get('incorrect', 0), n_natural)})\n"
    )
    lines.append(
        f"  - no_boxed: {nb.get('no_boxed', 0)} ({_pct(nb.get('no_boxed', 0), n_natural)})\n"
    )

    with open(accuracy_file, "a") as f:
        f.write("".join(lines))

    print(
        f"Termination stats: natural={n_natural} ({_pct(n_natural, total)}), "
        f"forced={n_forced} ({_pct(n_forced, total)}); "
        f"forced no_boxed={fb.get('no_boxed', 0)}, "
        f"forced incorrect={fb.get('incorrect', 0)}, "
        f"forced correct={fb.get('correct', 0)}"
    )


class CommonBenchmark:
    """Main class to run a benchmark"""

    def __init__(self, cfg: DictConfig):
        """
        Initialize the benchmark run

        Args:
            cfg: Hydra configuration object
        """
        self.cfg = cfg
        self.benchmark_name = cfg.benchmark.name
        evaluator_kwargs = cfg.benchmark.get("evaluator_kwargs", OmegaConf.create({}))
        # Support for legacy config structure
        if "metadata_file" in cfg.benchmark.data:
            evaluator_kwargs["metadata_file"] = cfg.benchmark.data.metadata_file
        if "field_mapping" in cfg.benchmark.data:
            mapping = cfg.benchmark.data.field_mapping
            if "task_id_field" in mapping:
                evaluator_kwargs["task_id_field"] = mapping.task_id_field
            if "task_question_field" in mapping:
                evaluator_kwargs["question_field"] = mapping.task_question_field
            if "ground_truth_field" in mapping:
                evaluator_kwargs["ground_truth_field"] = mapping.ground_truth_field
            if "file_name_field" in mapping:
                evaluator_kwargs["file_name_field"] = mapping.file_name_field

        self.evaluator = GenericEvaluator(
            data_dir=cfg.benchmark.data.data_dir,
            benchmark_name=self.benchmark_name,
            cfg=cfg,
            **evaluator_kwargs,
        )

    def run_evaluation(self) -> float:
        """
        Run the full benchmark evaluation process
        """
        print(f"Starting evaluation for benchmark: {self.benchmark_name}")
        print(f"LLM Provider: {self.evaluator.llm_provider}")
        print(f"LLM Model: {self.evaluator.llm_model}")

        # Load tasks
        self.evaluator.load_tasks(limit=self.cfg.benchmark.execution.max_tasks)
        if not self.evaluator.tasks:
            print("No tasks loaded. Exiting.")
            return 0.0

        # Run inference
        print(
            f"\nStarting parallel inference with {self.cfg.benchmark.execution.max_concurrent} concurrent tasks..."
        )
        print(f"Using pass@{self.evaluator.pass_at_k} evaluation...")

        self.evaluator.run_parallel_inference(
            self.evaluator.tasks,
            max_concurrent=self.cfg.benchmark.execution.max_concurrent,
        )

        # Evaluate accuracy
        print("Evaluating accuracy...")
        accuracy = self.evaluator.evaluate_accuracy()
        print(f"\nOverall pass@{self.evaluator.pass_at_k} accuracy: {accuracy:.2%}")

        # Evaluate pass@1 ~ pass@k accuracy for all levels
        all_accuracies = self.evaluator.evaluate_accuracy_all_k()

        # Save results
        # Construct the full path in the correct log directory
        log_dir = self.evaluator.get_log_dir()
        results_path = log_dir / "benchmark_results.jsonl"

        self.evaluator.save_results(str(results_path))
        print(f"\nEvaluation completed! Results saved to {results_path}")

        # Save combined accuracy file with all pass@1 ~ pass@k results
        k = self.evaluator.pass_at_k
        accuracy_file = str(results_path).replace(
            ".jsonl", f"_pass_at_{k}_accuracy.txt"
        )
        with open(accuracy_file, "w") as f:
            for i in range(1, k + 1):
                acc = all_accuracies.get(i, 0.0)
                f.write(f"pass@{i}: {acc:.2%}\n")

        print(f"Accuracy file saved: {accuracy_file}")

        # Append average tool call stats (correct vs incorrect) to accuracy file
        _append_tool_call_stats(results_path, accuracy_file)

        # Append termination analysis (natural vs forced summary, and the
        # correct/incorrect/no_boxed split among forced-summary attempts).
        _append_termination_stats(results_path, accuracy_file)

        # Append the count of attempts that never produced or received a verdict.
        n_errors = _append_error_stats(results_path, accuracy_file)
        if n_errors:
            print(
                f"\n*** WARNING: {n_errors} attempt(s) failed for infrastructure "
                "reasons and were counted as incorrect. See the 'Infrastructure "
                "Errors' section of the accuracy file; the reported score is a "
                "lower bound. ***\n"
            )

        # For DeepSearchQA, append the official 4 metrics (Fully Correct /
        # Fully Incorrect / Correct-with-Extraneous / F1) computed from the
        # persisted eval_details — no judge re-run needed.
        if self.benchmark_name.startswith("deepsearchqa"):
            from evaluators.deepsearchqa_metrics import append_metrics_to_txt

            append_metrics_to_txt(results_path, accuracy_file)

        return accuracy


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def run_benchmark(cfg: DictConfig) -> None:
    """
    Main entry point for running benchmarks with Hydra.
    """
    print("Benchmark configuration:\n", OmegaConf.to_yaml(cfg.benchmark))

    # A score is only comparable against runs graded by the same judge, so put
    # the resolved judge on the record at the top of every run -- and refuse to
    # start if there isn't one, rather than grade 2,158 questions against a model
    # the endpoint does not serve and report the resulting near-zero as a result.
    from evaluators.eval_utils import JUDGE_BASE_URL, JUDGE_MODEL_LABEL

    if JUDGE_MODEL_LABEL == "unknown" or not JUDGE_BASE_URL:
        raise SystemExit(
            "No judge configured. Set JUDGE_BASE_URL and JUDGE_MODEL_NAME (or "
            "OPENAI_BASE_URL / OPENAI_API_KEY as the fallback) in your .env. "
            "Gateways that route by URL path and take an empty model name should "
            "set JUDGE_PROVIDER=maas. See .env.example."
        )
    print(f"Judge: {JUDGE_MODEL_LABEL}\n")

    benchmark = CommonBenchmark(cfg)
    benchmark.run_evaluation()


if __name__ == "__main__":
    run_benchmark()
