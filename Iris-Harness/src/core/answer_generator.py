# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Answer generator module for final answer generation and context management.

This module provides the AnswerGenerator class that handles:
- LLM call processing
- Failure summary generation for context compression
- Final answer generation with retries
- Context management fallback strategies
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig

from ..io.output_formatter import OutputFormatter
from ..llm.base_client import BaseClient
from ..logging.task_logger import TaskLog
from ..utils.parsing_utils import (
    extract_failure_experience_summary,
    is_usable_failure_summary,
)
from ..utils.prompt_utils import (
    FAILURE_SUMMARY_ASSISTANT_PREFIX,
    FAILURE_SUMMARY_FALLBACK,
    FAILURE_SUMMARY_PROMPT,
    FAILURE_SUMMARY_RETRY_PROMPT,
    FAILURE_SUMMARY_THINK_CONTENT,
    FORMAT_ERROR_MESSAGE,
    generate_agent_summarize_prompt,
    generate_direct_summarize_prompt,
)
from ..utils.wrapper_utils import ErrorBox, ResponseBox
from .stream_handler import StreamHandler

logger = logging.getLogger(__name__)

# Safety limits for retry loops
DEFAULT_MAX_FINAL_ANSWER_RETRIES = 3


class AnswerGenerator:
    """
    Generator for final answers with context management support.

    Handles the generation of final answers, failure summaries for retry,
    and various fallback strategies based on context management settings.
    """

    def __init__(
        self,
        llm_client: BaseClient,
        output_formatter: OutputFormatter,
        task_log: TaskLog,
        stream_handler: StreamHandler,
        cfg: DictConfig,
        intermediate_boxed_answers: List[str],
    ):
        """
        Initialize the answer generator.

        Args:
            llm_client: The LLM client for API calls
            output_formatter: Formatter for output processing
            task_log: Logger for task execution
            stream_handler: Handler for streaming events
            cfg: Configuration object
            intermediate_boxed_answers: List to track intermediate answers
        """
        self.llm_client = llm_client
        self.output_formatter = output_formatter
        self.task_log = task_log
        self.stream = stream_handler
        self.cfg = cfg
        self.intermediate_boxed_answers = intermediate_boxed_answers

        # Context management settings
        self.context_compress_limit = cfg.agent.get("context_compress_limit", 0)
        # Always allow multiple final-answer retries.  Previously this was
        # reduced to 1 when keep_tool_result != -1 (reasoning: filtered
        # context makes retry useless).  However, transient API failures
        # (e.g. empty content from rate-limited endpoints) benefit greatly
        # from retrying — the context quality is irrelevant in that case.
        self.max_final_answer_retries = DEFAULT_MAX_FINAL_ANSWER_RETRIES
        self.retry_with_summary = cfg.agent.get("retry_with_summary", True)

    async def handle_llm_call(
        self,
        system_prompt: str,
        message_history: List[Dict[str, Any]],
        tool_definitions: List[Dict],
        step_id: int,
        purpose: str = "",
        agent_type: str = "main",
    ) -> Tuple[Optional[str], bool, Optional[Any], List[Dict[str, Any]]]:
        """
        Unified LLM call and logging processing.

        Args:
            system_prompt: System prompt for the LLM
            message_history: Conversation history
            tool_definitions: Available tool definitions
            step_id: Current step ID for logging
            purpose: Description of the call purpose
            agent_type: Type of agent making the call

        Returns:
            Tuple of (response_text, should_break, tool_calls_info, message_history)
        """
        original_message_history = message_history
        try:
            response, message_history = await self.llm_client.create_message(
                system_prompt=system_prompt,
                message_history=message_history,
                tool_definitions=tool_definitions,
                keep_tool_result=self.cfg.agent.keep_tool_result,
                step_id=step_id,
                task_log=self.task_log,
                agent_type=agent_type,
            )

            if ErrorBox.is_error_box(response):
                await self.stream.show_error(str(response))
                response = None

            if ResponseBox.is_response_box(response):
                if response.has_extra_info():
                    extra_info = response.get_extra_info()
                    if extra_info.get("warning_msg"):
                        await self.stream.show_error(
                            extra_info.get("warning_msg", "Empty warning message")
                        )
                response = response.get_response()

            # Check if response is None (indicating an error occurred)
            if response is None:
                self.task_log.log_step(
                    "error",
                    f"{purpose} | LLM Call Failed",
                    f"{purpose} failed - no response received",
                )
                return "", False, None, original_message_history

            # Use client's response processing method
            assistant_response_text, should_break, message_history = (
                self.llm_client.process_llm_response(
                    response, message_history, agent_type
                )
            )

            # Use client's tool call information extraction method
            tool_calls_info = self.llm_client.extract_tool_calls_info(
                response, assistant_response_text
            )

            self.task_log.log_step(
                "info",
                f"{purpose} | LLM Call",
                "completed successfully",
            )
            return (
                assistant_response_text,
                should_break,
                tool_calls_info,
                message_history,
            )

        except Exception as e:
            self.task_log.log_step(
                "error",
                f"{purpose} | LLM Call ERROR",
                f"{purpose} error: {str(e)}",
            )
            # Return empty response with should_break=False, need to retry
            return "", False, None, original_message_history

    async def generate_failure_summary(
        self,
        system_prompt: str,
        message_history: List[Dict[str, Any]],
        tool_definitions: List[Dict],
        turn_count: int,
    ) -> Optional[str]:
        """
        Generate a failure experience summary for context compression.

        This is the core of the context management mechanism. When a task attempt fails
        (i.e., the task is not completed within the given turns and context window),
        we compress the entire conversation history into a structured summary containing:
        - Failure type: incomplete / blocked / misdirected / format_missed
        - What happened: the approach taken and why a final answer was not reached
        - Useful findings: facts, intermediate results, or conclusions to be reused

        Args:
            system_prompt: The system prompt used in the conversation
            message_history: The full conversation history to be compressed
            tool_definitions: Available tool definitions
            turn_count: Current turn count for step ID

        Returns:
            The compressed failure experience summary, or None if generation failed
        """
        self.task_log.log_step(
            "info",
            "Main Agent | Failure Summary",
            "Generating failure experience summary for potential retry...",
        )

        # Build failure summary history
        failure_summary_history = message_history.copy()
        # Remove trailing tool result messages (role="user" in mcp_xml, role="tool" in native_fc)
        while failure_summary_history and failure_summary_history[-1]["role"] in ("user", "tool"):
            failure_summary_history.pop()
            if not failure_summary_history:
                break

        # Cheap pre-trim so the failure-summary request fits the hard context
        # window (this runs on the outer context-compress retry path, which can
        # be triggered on an already very long trajectory). Reserve room for the
        # system prompt (prepended at send time), the failure-summary prompt, and
        # the response budget.
        failure_prompt_reserve = self.llm_client._estimate_tokens(FAILURE_SUMMARY_PROMPT)
        system_reserve = self.llm_client._estimate_tokens(system_prompt)
        failure_summary_history = self.llm_client.enforce_context_budget(
            failure_summary_history,
            reserve_tokens=self.llm_client.max_tokens
            + failure_prompt_reserve
            + system_reserve
            + 1000,
        )

        # Generate the post-mortem with a bounded  validate -> corrective re-ask
        # -> deterministic fallback  flow. SFT search models frequently ignore
        # "summarize, don't call tools" and emit tool-call TEXT here; that garbage
        # must never be injected into the next attempt's task description, and the
        # recovery must stay bounded (no loops) to protect eval throughput.
        base_history = failure_summary_history

        async def _run_summary_attempt(prompt: str, step_id: int) -> str:
            # Append the summary prompt + an assistant prefix that primes the
            # structured format. In "preserve" mode the priming goes to the
            # dedicated ``_reasoning_content`` field (lifted to reasoning_content
            # by ``_create_message``); otherwise it is a ``<think>`` content prefix.
            attempt_history = base_history + [{"role": "user", "content": prompt}]
            if self.llm_client.reasoning_content_mode == "preserve":
                attempt_history.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "_reasoning_content": FAILURE_SUMMARY_THINK_CONTENT,
                    }
                )
            else:
                attempt_history.append(
                    {"role": "assistant", "content": FAILURE_SUMMARY_ASSISTANT_PREFIX}
                )
            # Empty tool list -> no tool schema in the request -> the model can
            # only produce text (it may still emit tool-call TEXT — handled below).
            (text, _, _, _) = await self.handle_llm_call(
                system_prompt,
                attempt_history,
                [],
                step_id,
                "Main Agent | Failure Experience Summary",
                agent_type="main",
            )
            if not text:
                return ""
            # Extract the structured summary: strip the model's own <think> CoT
            # and any tool-call XML it emitted as literal text. We do NOT prepend
            # the canned priming seed — doing so let the seed text leak in as a
            # bogus "summary" whenever the model produced only tool calls.
            return extract_failure_experience_summary(text)

        summary = await _run_summary_attempt(FAILURE_SUMMARY_PROMPT, turn_count + 10)
        source = "model"
        if not is_usable_failure_summary(summary):
            # First reply was a tool call / not a post-mortem: ONE corrective
            # re-ask (bounded — never loops, protects eval efficiency).
            self.task_log.log_step(
                "warning",
                "Main Agent | Failure Summary",
                "First failure-summary reply was unusable (tool-call or empty); "
                "issuing one corrective re-ask.",
            )
            summary = await _run_summary_attempt(
                FAILURE_SUMMARY_RETRY_PROMPT, turn_count + 11
            )
            if not is_usable_failure_summary(summary):
                # Still unusable -> inject a clean deterministic note instead of
                # ever letting raw tool-call text reach the next task description.
                summary = FAILURE_SUMMARY_FALLBACK
                source = "fallback"

        log_preview = summary[:500] + ("..." if len(summary) > 500 else "")
        self.task_log.log_step(
            "info",
            "Main Agent | Failure Summary",
            f"Failure experience summary ({source}):\n{log_preview}",
        )
        return summary

    async def generate_final_answer_with_retries(
        self,
        system_prompt: str,
        message_history: List[Dict[str, Any]],
        tool_definitions: List[Dict],
        turn_count: int,
        task_description: str,
        answer_mode: str = "boxed",
        summary_prompt_override: Optional[str] = None,
    ) -> Tuple[Optional[str], str, Optional[str], str, List[Dict[str, Any]]]:
        """
        Generate final answer with retry mechanism.

        Args:
            system_prompt: System prompt for the LLM
            message_history: Conversation history
            tool_definitions: Available tool definitions
            turn_count: Current turn count
            task_description: Original task description
            answer_mode: "boxed" or "direct"

        Returns:
            Tuple of (final_answer_text, final_summary, final_boxed_answer, usage_log, message_history)
        """
        # Generate summary prompt based on answer_mode. A run-level override (used by
        # self-verification) forces a specific summary prompt so the verifier still emits
        # a JSON verdict even when it hits max_turns (abnormal termination).
        if summary_prompt_override:
            summary_prompt = summary_prompt_override
        elif answer_mode == "direct":
            summary_prompt = generate_direct_summarize_prompt(task_description)
        else:
            benchmark_name = ""
            if self.cfg is not None:
                benchmark_name = self.cfg.get("benchmark", {}).get("name", "") or ""
            summary_prompt = generate_agent_summarize_prompt(
                task_description,
                agent_type="main",
                benchmark_name=benchmark_name,
            )

        # Remove trailing tool result messages (role="user" in mcp_xml, role="tool" in native_fc)
        while message_history and message_history[-1]["role"] in ("user", "tool"):
            message_history.pop(-1)
        # Cheap pre-trim so the summary request is close to the hard context
        # window (the pre-send guard only runs inside the main loop). This drops
        # the near-minimum; the loop below then drops one more round per REAL
        # context-length 400 until the summary actually goes through. Reserve room
        # for the system prompt (prepended at send time, not in message_history),
        # the summary prompt, and the response budget.
        summary_prompt_reserve = self.llm_client._estimate_tokens(summary_prompt)
        system_reserve = self.llm_client._estimate_tokens(system_prompt)
        message_history = self.llm_client.enforce_context_budget(
            message_history,
            reserve_tokens=self.llm_client.max_tokens
            + summary_prompt_reserve
            + system_reserve
            + 1000,
        )
        message_history.append({"role": "user", "content": summary_prompt})

        final_answer_text = None
        final_boxed_answer = None
        final_summary = ""
        usage_log = ""

        # Direct mode: single attempt, no boxed retry needed
        max_retries = 1 if answer_mode == "direct" else self.max_final_answer_retries

        # Separate, bounded budget for "shrink until the summary request fits".
        # If the summary call is rejected for context length, drop ONE more
        # trailing round (the one just before the summary prompt) and retry —
        # driven by the REAL 400 (the server is the oracle), so we remove the true
        # minimum. A context-fit retry does NOT consume a format retry.
        max_context_drops = 64
        context_drops = 0
        retry_idx = 0
        while retry_idx < max_retries:
            (
                final_answer_text,
                should_break,
                tool_calls_info,
                message_history,
            ) = await self.handle_llm_call(
                system_prompt,
                message_history,
                # Force-summary must NOT call tools. Passing an empty tool list
                # means the API request carries no tool schema (all providers
                # guard `if tools:` before adding the param + tool_choice), so
                # the model is structurally unable to emit a tool call and must
                # answer from the existing context. A text-only instruction is
                # not enough to stop SFT models that were trained to keep calling
                # tools whenever tools are available.
                [],
                turn_count + 1 + retry_idx + context_drops,
                f"Main agent | Final Summary (attempt {retry_idx + 1}/{max_retries})",
                agent_type="main",
            )

            # If the summary request itself exceeded the context window, drop one
            # more trailing round (before the summary prompt) and retry until it
            # fits or there is nothing left to drop. Does NOT consume a format retry.
            if (
                not final_answer_text
                and getattr(self.llm_client, "context_length_exceeded", False)
                and context_drops < max_context_drops
            ):
                summary_msg = (
                    message_history.pop()
                    if message_history
                    and message_history[-1].get("role") == "user"
                    else None
                )
                dropped = self.llm_client.drop_last_round(message_history)
                if summary_msg is not None:
                    message_history.append(summary_msg)
                if dropped:
                    context_drops += 1
                    self.task_log.log_step(
                        "warning",
                        "Main Agent | Final Summary",
                        f"Summary request exceeded the context window; dropped 1 "
                        f"more round (total {context_drops}) and retrying.",
                    )
                    continue  # retry this format attempt; do not consume a retry
                # nothing left to drop -> fall through to failure handling below

            if final_answer_text:
                final_summary, final_boxed_answer, usage_log = (
                    self.output_formatter.format_final_summary_and_log(
                        final_answer_text, self.llm_client, answer_mode=answer_mode
                    )
                )

                if answer_mode == "direct":
                    # Direct mode: content is the answer, always valid
                    self.task_log.log_step(
                        "info",
                        "Main Agent | Final Answer (direct)",
                        f"Direct answer extracted (length={len(final_boxed_answer)})",
                    )
                    break
                elif final_boxed_answer != FORMAT_ERROR_MESSAGE:
                    self.task_log.log_step(
                        "info",
                        "Main Agent | Final Answer",
                        f"Boxed answer found on attempt {retry_idx + 1}",
                    )
                    break
                else:
                    self.task_log.log_step(
                        "warning",
                        "Main Agent | Final Answer",
                        f"No boxed answer on attempt {retry_idx + 1}, retrying...",
                    )
                    if retry_idx < max_retries - 1:
                        if (
                            message_history
                            and message_history[-1]["role"] == "assistant"
                        ):
                            message_history.pop()
            else:
                self.task_log.log_step(
                    "warning",
                    "Main Agent | Final Answer",
                    f"Failed to generate answer on attempt {retry_idx + 1}",
                )
                if retry_idx < max_retries - 1:
                    if message_history and message_history[-1]["role"] == "assistant":
                        message_history.pop()

            retry_idx += 1

        # Ensure final_boxed_answer is never None
        if final_boxed_answer is None:
            final_boxed_answer = FORMAT_ERROR_MESSAGE

        return (
            final_answer_text,
            final_summary,
            final_boxed_answer,
            usage_log,
            message_history,
        )

    def handle_no_context_management_fallback(
        self,
        final_answer_text: Optional[str],
        final_summary: str,
        final_boxed_answer: Optional[str],
    ) -> Tuple[str, str, str]:
        """
        Handle fallback when context_compress_limit == 0 (no context management).

        In this mode, the model has only one chance to answer.
        We should try to use intermediate answers as fallback to maximize accuracy.

        Args:
            final_answer_text: The generated final answer text
            final_summary: The final summary
            final_boxed_answer: The extracted boxed answer

        Returns:
            Tuple of (final_answer_text, final_summary, final_boxed_answer)
        """
        # Validate final_answer_text
        if not final_answer_text:
            final_answer_text = "No final answer generated."
            final_summary = final_answer_text
            final_boxed_answer = FORMAT_ERROR_MESSAGE
            self.task_log.log_step(
                "error",
                "Main Agent | Final Answer",
                "Unable to generate final answer after all retries",
            )
        else:
            self.task_log.log_step(
                "info",
                "Main Agent | Final Answer",
                f"Final answer content:\n\n{final_answer_text}",
            )

        # Fallback to intermediate answer if no valid boxed answer
        if (
            final_boxed_answer == FORMAT_ERROR_MESSAGE or final_boxed_answer is None
        ) and self.intermediate_boxed_answers:
            final_boxed_answer = self.intermediate_boxed_answers[-1]
            self.task_log.log_step(
                "info",
                "Main Agent | Final Answer (No Context Management)",
                f"Using intermediate boxed answer as fallback: {final_boxed_answer}",
            )

        # Ensure final_boxed_answer is never None
        if final_boxed_answer is None:
            final_boxed_answer = FORMAT_ERROR_MESSAGE

        return final_answer_text, final_summary, final_boxed_answer

    def handle_context_management_no_fallback(
        self,
        final_answer_text: Optional[str],
        final_summary: str,
        final_boxed_answer: Optional[str],
    ) -> Tuple[str, str, str]:
        """
        Handle failure when context_compress_limit > 0 (context management enabled).

        In this mode, the model has multiple chances to retry with context management.
        We should NOT guess or use intermediate answers, because:
        - A wrong guess can reduce accuracy
        - The model will have another chance to answer with failure experience

        Args:
            final_answer_text: The generated final answer text
            final_summary: The final summary
            final_boxed_answer: The extracted boxed answer

        Returns:
            Tuple of (final_answer_text, final_summary, final_boxed_answer)
        """
        # Validate final_answer_text
        if not final_answer_text:
            final_answer_text = "No final answer generated."
            final_summary = final_answer_text
            final_boxed_answer = FORMAT_ERROR_MESSAGE
            self.task_log.log_step(
                "error",
                "Main Agent | Final Answer",
                "Unable to generate final answer after all retries",
            )
        else:
            self.task_log.log_step(
                "info",
                "Main Agent | Final Answer",
                f"Final answer content:\n\n{final_answer_text}",
            )

        # Ensure final_boxed_answer is never None
        if final_boxed_answer is None:
            final_boxed_answer = FORMAT_ERROR_MESSAGE

        # With context management, do NOT fallback to intermediate answers
        if final_boxed_answer == FORMAT_ERROR_MESSAGE:
            self.task_log.log_step(
                "info",
                "Main Agent | Final Answer (Context Management Mode)",
                "No valid boxed answer found. Not using intermediate fallback - will generate failure summary for retry.",
            )

        return final_answer_text, final_summary, final_boxed_answer

    async def generate_and_finalize_answer(
        self,
        system_prompt: str,
        message_history: List[Dict[str, Any]],
        tool_definitions: List[Dict],
        turn_count: int,
        task_description: str,
        reached_max_turns: bool = False,
        is_final_retry: bool = False,
        save_callback=None,
        answer_mode: str = "boxed",
        last_assistant_content: Optional[str] = None,
        summary_prompt_override: Optional[str] = None,
    ) -> Tuple[str, str, Optional[str], str, List[Dict[str, Any]]]:
        """
        Generate final answer and handle fallback based on context management settings.

        Context Management (context_compress_limit > 0) is essentially a context compression
        mechanism that enables multi-attempt problem solving.

        Decision table for **boxed** mode (unchanged from previous behavior):

        | Context Management | Reached Max Turns | Behavior                                    |
        |--------------------|-------------------|---------------------------------------------|
        | OFF (limit=0)      | No                | Generate answer → fallback to intermediate  |
        | OFF (limit=0)      | Yes               | Generate answer → fallback to intermediate  |
        | ON  (limit>0)      | No                | Generate answer → no fallback, fail summary |
        | ON  (limit>0)      | Yes               | SKIP generation → fail summary directly     |

        Decision table for **direct** mode:

        | Reached Max Turns | Behavior                                              |
        |-------------------|-------------------------------------------------------|
        | No (normal end)   | Use last_assistant_content directly — no summary call |
        | Yes (abnormal)    | Trigger lightweight summary (natural language)        |

        Args:
            system_prompt: System prompt for the LLM
            message_history: Conversation history
            tool_definitions: Available tool definitions
            turn_count: Current turn count
            task_description: Original task description
            reached_max_turns: Whether the main loop ended due to reaching max turns
            is_final_retry: Whether this is the last retry opportunity
            save_callback: Optional callback to save message history
            answer_mode: "boxed" (default, existing behavior) or "direct" (new mode)
            last_assistant_content: Last assistant response content (used in direct mode
                for normal completion — avoids an extra LLM call)

        Returns:
            Tuple of (final_summary, final_boxed_answer, failure_experience_summary, usage_log, message_history)
        """
        # ── Direct mode ──────────────────────────────────────────────────
        if answer_mode == "direct":
            failure_experience_summary = None
            usage_log = ""

            if not reached_max_turns and last_assistant_content:
                # Normal completion: take last assistant content directly
                self.task_log.log_step(
                    "info",
                    "Main Agent | Final Answer (direct, normal)",
                    f"Using last assistant content as answer (length={len(last_assistant_content)})",
                )
                final_answer = last_assistant_content.strip()
                final_summary, final_boxed_answer, usage_log = (
                    self.output_formatter.format_final_summary_and_log(
                        final_answer, self.llm_client, answer_mode="direct"
                    )
                )

                if save_callback:
                    save_callback(system_prompt, message_history)

                return (
                    final_summary,
                    final_boxed_answer,
                    None,
                    usage_log,
                    message_history,
                )
            else:
                # Abnormal termination: trigger lightweight summary
                self.task_log.log_step(
                    "info",
                    "Main Agent | Final Answer (direct, abnormal)",
                    "Abnormal termination — triggering direct summary prompt",
                )
                (
                    final_answer_text,
                    final_summary,
                    final_boxed_answer,
                    usage_log,
                    message_history,
                ) = await self.generate_final_answer_with_retries(
                    system_prompt=system_prompt,
                    message_history=message_history,
                    tool_definitions=tool_definitions,
                    turn_count=turn_count,
                    task_description=task_description,
                    answer_mode="direct",
                    summary_prompt_override=summary_prompt_override,
                )

                if save_callback:
                    save_callback(system_prompt, message_history)

                # In direct mode with context management, still generate failure summary for retry
                if (
                    self.context_compress_limit > 0
                    and reached_max_turns
                    and not is_final_retry
                    and self.retry_with_summary
                ):
                    failure_experience_summary = await self.generate_failure_summary(
                        system_prompt, message_history, tool_definitions, turn_count
                    )

                return (
                    final_summary,
                    final_boxed_answer,
                    failure_experience_summary,
                    usage_log,
                    message_history,
                )

        # ── Boxed mode (unchanged existing behavior) ─────────────────────
        context_management_enabled = self.context_compress_limit > 0
        failure_experience_summary = None
        usage_log = ""

        # CASE: Context management ON + reached max turns + NOT final retry
        # Skip answer generation entirely - any answer would be a blind guess
        # But if this is the final retry, we still try to generate an answer (last chance)
        if context_management_enabled and reached_max_turns and not is_final_retry:
            self.task_log.log_step(
                "info",
                "Main Agent | Final Answer (Context Management Mode)",
                "Reached max turns. Skipping answer generation to avoid blind guessing.",
            )

            if save_callback:
                save_callback(system_prompt, message_history)

            if self.retry_with_summary:
                failure_experience_summary = await self.generate_failure_summary(
                    system_prompt, message_history, tool_definitions, turn_count
                )

            return (
                "Task incomplete - reached maximum turns. Will retry with failure experience.",
                FORMAT_ERROR_MESSAGE,
                failure_experience_summary,
                usage_log,
                message_history,
            )

        # ALL OTHER CASES: Generate final answer first
        # (including final retry with reached_max_turns - last chance to get an answer)
        (
            final_answer_text,
            final_summary,
            final_boxed_answer,
            usage_log,
            message_history,
        ) = await self.generate_final_answer_with_retries(
            system_prompt=system_prompt,
            message_history=message_history,
            tool_definitions=tool_definitions,
            turn_count=turn_count,
            task_description=task_description,
            answer_mode="boxed",
            summary_prompt_override=summary_prompt_override,
        )

        if save_callback:
            save_callback(system_prompt, message_history)

        # CASE: Context management OFF or final retry
        # Try to use intermediate answers as fallback to maximize accuracy
        # For final retry, there's no more retry opportunity, so we use fallback
        if not context_management_enabled or is_final_retry:
            final_answer_text, final_summary, final_boxed_answer = (
                self.handle_no_context_management_fallback(
                    final_answer_text, final_summary, final_boxed_answer
                )
            )
            if is_final_retry:
                self.task_log.log_step(
                    "info",
                    "Main Agent | Final Answer (Final Retry)",
                    "This is the final retry. Using intermediate fallback if available.",
                )
            return (
                final_summary,
                final_boxed_answer,
                None,
                usage_log,
                message_history,
            )

        # CASE: Context management ON + normal completion (not reached max turns, not final retry)
        # Don't use fallback - wrong guess would reduce accuracy
        final_answer_text, final_summary, final_boxed_answer = (
            self.handle_context_management_no_fallback(
                final_answer_text, final_summary, final_boxed_answer
            )
        )

        if final_boxed_answer == FORMAT_ERROR_MESSAGE and self.retry_with_summary:
            failure_experience_summary = await self.generate_failure_summary(
                system_prompt, message_history, tool_definitions, turn_count
            )

        return (
            final_summary,
            final_boxed_answer,
            failure_experience_summary,
            usage_log,
            message_history,
        )
