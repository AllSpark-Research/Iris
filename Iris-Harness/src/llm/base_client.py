# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Base client module for LLM providers.

This module defines the abstract base class and common utilities for LLM clients,
supporting both OpenAI and Anthropic API formats.
"""

import asyncio
import copy
import dataclasses
from abc import ABC
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    TypedDict,
)

from omegaconf import DictConfig, OmegaConf

from ..logging.task_logger import TaskLog
from .util import with_timeout

# Default timeout for LLM API calls (10 minutes)
DEFAULT_LLM_TIMEOUT_SECONDS = 600


class TokenUsage(TypedDict, total=True):
    """
    Unified token usage tracking across different LLM providers.

    We unify OpenAI and Anthropic formats. There are four usage types:
    - input/output tokens: Standard input and output token counts
    - cache write/read tokens: Tokens involved in caching operations

    Provider-specific notes:
    - OpenAI: Cache write is free, cache read is cheaper
    - Anthropic: Cache write has a small cost, cache read is cheaper
    """

    total_input_tokens: int
    total_output_tokens: int
    total_cache_read_input_tokens: int
    total_cache_write_input_tokens: int


@dataclasses.dataclass
class BaseClient(ABC):
    """
    Abstract base class for LLM provider clients.

    This class provides the common interface and utilities for interacting with
    different LLM providers (OpenAI, Anthropic, etc.). Concrete implementations
    should override _create_client() and provider-specific methods.

    Attributes:
        task_id: Unique identifier for the current task (used for tracking)
        cfg: Hydra configuration containing LLM settings
        task_log: Optional logger for recording task execution details
    """

    # Required arguments (no default value)
    task_id: str
    cfg: DictConfig

    # Optional arguments (with default value)
    task_log: Optional["TaskLog"] = None

    # Initialized in __post_init__
    client: Any = dataclasses.field(init=False)
    token_usage: TokenUsage = dataclasses.field(init=False)
    last_call_tokens: Dict[str, int] = dataclasses.field(init=False)

    def __post_init__(self):
        # Initialize last_call_tokens before other operations
        self.last_call_tokens: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

        # Set True by the provider create-loop when a request is rejected for
        # exceeding the model's hard context window (HTTP 400 "maximum context
        # length …" / "requested token count exceeds …"). The orchestrator reads
        # it to END the loop and summarize instead of blindly re-sending the
        # identical over-limit request (which would 400 forever). It is reset at
        # the start of every _create_message call, so it always reflects only the
        # most recent call. Per-task safe: one client instance per task.
        self.context_length_exceeded: bool = False

        # Explicitly assign from cfg object
        self.provider: str = self.cfg.llm.provider
        self.model_name: str = self.cfg.llm.model_name
        self.temperature: float = self.cfg.llm.temperature
        self.top_p: float = self.cfg.llm.top_p
        self.min_p: float = self.cfg.llm.min_p
        self.top_k: int = self.cfg.llm.top_k
        self.max_context_length: int = self.cfg.llm.max_context_length
        self.max_tokens: int = self.cfg.llm.max_tokens
        self.async_client: bool = self.cfg.llm.async_client
        self.keep_tool_result: int = self.cfg.agent.keep_tool_result
        self.api_key: Optional[str] = self.cfg.llm.get("api_key")
        self.base_url: Optional[str] = self.cfg.llm.get("base_url")
        self.api_version: Optional[str] = self.cfg.llm.get("api_version")
        self.use_tool_calls: Optional[bool] = self.cfg.llm.get("use_tool_calls")
        self.repetition_penalty: float = self.cfg.llm.get("repetition_penalty", 1.0)
        self.presence_penalty: float = self.cfg.llm.get("presence_penalty", 0.0)
        self.tool_call_mode: str = self.cfg.llm.get("tool_call_mode", "mcp_xml")

        # Extended thinking / reasoning parameters
        self.thinking_enabled: bool = self.cfg.llm.get("thinking_enabled", False)
        self.thinking_budget_tokens: int = self.cfg.llm.get("thinking_budget_tokens", 10000)
        self.reasoning_effort: Optional[str] = self.cfg.llm.get("reasoning_effort", None)

        # Native "preserve thinking" switch — an endpoint-specific extra_body
        # fragment that is deep-merged into every OpenAI-compatible request.
        # Declaring it per-config avoids hardcoding model-name branches in code.
        # Examples:
        #   Kimi MaaS (official):  {"thinking": {"type": "enabled", "keep": "all"}}
        #   Kimi/Qwen SGLang/vLLM: {"chat_template_kwargs": {"thinking": true, "preserve_thinking": true}}
        _teb = self.cfg.llm.get("thinking_extra_body", None)
        self.thinking_extra_body: Dict[str, Any] = (
            OmegaConf.to_container(_teb, resolve=True) if _teb else {}
        )

        # Length-truncation retry budget (INDEPENDENT of transient-error retry).
        # When a single response is cut off by max_tokens (finish_reason
        # "length" / stop_reason "max_tokens" / finishReason "MAX_TOKENS"), the
        # client re-issues the SAME request with a larger max_tokens (×1.5,
        # capped at max_context_length - prompt_tokens) up to this many times
        # before giving up and returning the truncated response — at which point
        # the orchestrator turns it into a final summary (the legacy behaviour).
        self.length_retry_max: int = int(self.cfg.llm.get("length_retry_max", 5))

        # Reasoning content handling mode: "discard" | "log_only" | "context" | "preserve"
        # Backward compatible: old `save_reasoning_content` bool is mapped automatically.
        _VALID_REASONING_MODES = {"discard", "log_only", "context", "preserve"}
        _rcm = self.cfg.llm.get("reasoning_content_mode", None)
        if _rcm is not None:
            if _rcm not in _VALID_REASONING_MODES:
                raise ValueError(
                    f"Invalid reasoning_content_mode: {_rcm!r}. "
                    f"Must be one of: {sorted(_VALID_REASONING_MODES)}"
                )
            self.reasoning_content_mode: str = _rcm
        else:
            # Fallback: map old bool config → new string mode
            _old = self.cfg.llm.get("save_reasoning_content", False)
            self.reasoning_content_mode: str = "log_only" if _old else "discard"


        self.token_usage = self._reset_token_usage()
        self.client = self._create_client()

        self.task_log.log_step(
            "info",
            "LLM | Initialization",
            f"LLMClient {self.provider} {self.model_name} initialization completed.",
        )

    def _reset_token_usage(self) -> TokenUsage:
        """
        Reset token usage counter to zero.

        Returns:
            A new TokenUsage dict with all counters set to zero.
        """
        return TokenUsage(
            total_input_tokens=0,
            total_output_tokens=0,
            total_cache_write_input_tokens=0,
            total_cache_read_input_tokens=0,
        )

    def _remove_tool_result_from_messages(
        self, messages, keep_tool_result
    ) -> List[Dict]:
        """Remove tool results from messages

        Args:
            messages: List of message dictionaries
            keep_tool_result: Number of tool results to keep. -1 means keep all.

        Returns:
            List of messages with tool results filtered according to keep_tool_result
        """
        messages_copy = [m.copy() for m in messages]

        if keep_tool_result == -1:
            # No processing needed, keep all messages
            return messages_copy

        # Which messages are actually tool observations?
        #
        # In native_fc the server returns them as role="tool", and every
        # role="user" message is a real instruction -- the task itself, or the
        # summary prompt appended when an episode ends. Folding a user message
        # there would delete the instruction the model is being asked to follow,
        # which at keep_tool_result=0 silently destroyed every final summary.
        #
        # In mcp_xml there is no "tool" role: results come back as user turns, so
        # we fall back to "every user message after the first", and exempt the
        # trailing one for the same reason.
        last_idx = len(messages_copy) - 1
        if self.tool_call_mode == "native_fc":
            foldable = [
                i for i, m in enumerate(messages_copy) if m.get("role") == "tool"
            ]
        else:
            user_indices = [
                i for i, m in enumerate(messages_copy) if m.get("role") == "user"
            ]
            foldable = [i for i in user_indices[1:] if i != last_idx]

        if not foldable:
            self.task_log.log_step(
                "info",
                "LLM | Message Retention",
                "No tool results to fold.",
            )
            return messages_copy

        # Keep the most recent `keep_tool_result` of them verbatim; 0 folds all.
        n_keep = 0 if keep_tool_result == 0 else min(keep_tool_result, len(foldable))
        indices_to_fold = set(foldable[: len(foldable) - n_keep])

        self.task_log.log_step(
            "info",
            "LLM | Message Retention",
            f"Tool results: {len(foldable)} | kept verbatim: {len(foldable) - len(indices_to_fold)} "
            f"| folded: {len(indices_to_fold)}",
        )

        for i in indices_to_fold:
            msg = messages_copy[i]
            if isinstance(msg.get("content"), list):
                # Anthropic-style content blocks
                msg["content"] = [
                    {"type": "text", "text": "Tool result is omitted to save tokens."}
                ]
            else:
                msg["content"] = "Tool result is omitted to save tokens."

        return messages_copy

    # Safety margin (tokens) kept free below max_context_length when growing
    # max_tokens on a length-truncation retry, so the larger output budget does
    # not itself push the request over the context window.
    _LENGTH_RETRY_SAFETY_MARGIN = 512

    def _grow_max_tokens_for_length(
        self, current_max_tokens: int, prompt_tokens: int
    ) -> int:
        """Compute the next ``max_tokens`` for a length-truncation retry.

        Grows the budget ×1.5, capped at
        ``max_context_length - prompt_tokens - SAFETY``. Returns a value
        ``<= current_max_tokens`` when there is no headroom left to grow, which
        the caller uses as the signal to STOP retrying and return the truncated
        response.
        """
        headroom = (
            int(self.max_context_length)
            - int(prompt_tokens or 0)
            - self._LENGTH_RETRY_SAFETY_MARGIN
        )
        grown = int(current_max_tokens * 1.5)
        return min(grown, headroom)

    def discard_all_tool_history(self, messages: List[Dict]) -> List[Dict]:
        """
        Discard all tool-call history, keeping only the initial task message.

        This implements the 'Discard-all' reset from GLM-5's Hierarchical Context
        Management (HCM) strategy. When the context length exceeds a threshold T
        during keep-recent-k inference, the entire tool-call history is discarded
        and the agent restarts with a fresh context containing only the original
        question, while continuing to apply keep-recent-k going forward.

        Args:
            messages: Full message history list

        Returns:
            A new list containing only the first user message (initial task)
        """
        if not messages:
            return messages

        # Find the first user message (initial task).
        # Use deepcopy to avoid shared references with the original
        # message_history — downstream operations like _apply_cache_control
        # mutate content blocks, and shallow copy would corrupt logged data.
        first_user_msg = None
        for msg in messages:
            if msg.get("role") == "user":
                first_user_msg = copy.deepcopy(msg)
                break

        if first_user_msg is None:
            self.task_log.log_step(
                "warning",
                "LLM | HCM Discard",
                "No user message found in history, returning empty list.",
            )
            return []

        discarded_count = len(messages) - 1
        self.task_log.log_step(
            "info",
            "LLM | HCM Discard-All",
            f"Discarded {discarded_count} messages, restarting with fresh context "
            f"(initial task only).",
        )

        return [first_user_msg]

    @staticmethod
    def is_context_length_error(err: object) -> bool:
        """Return True if ``err`` (an exception or its string form) is a hard
        context-window overflow rejection from the model server.

        Providers phrase this differently; we match all known forms so the caller
        can react (end the loop / summarize) instead of mistaking it for a
        transient error and retrying the identical request forever:
          * OpenAI / vLLM: "... is longer than the model's context length ..."
                           "This model's maximum context length is N tokens ..."
          * SGLang:        "Requested token count exceeds the model's maximum
                           context length of N ..."
          * generic:       "context length exceeded" / "context window" /
                           "reduce the length of the messages"
        """
        s = str(err).lower()
        markers = (
            "context length",
            "context window",
            "maximum context",
            "longer than the model",
            "requested token count exceeds",
            "reduce the length of the messages",
            "exceeds the maximum number of tokens",
        )
        return any(m in s for m in markers)

    def drop_last_round(self, message_history: List[Dict], min_len: int = 1) -> bool:
        """Pop ONE complete trailing tool-call round from ``message_history`` in
        place: the trailing run of tool-result messages (``role`` in
        {"tool", "user"}) plus the single ``assistant`` that produced them.

        Never pops below ``min_len`` (used to protect a leading system message
        and/or the first user/task message). Returns True iff anything was
        removed.
        """
        removed = False
        while (
            len(message_history) > min_len
            and message_history[-1].get("role") in ("tool", "user")
        ):
            message_history.pop()
            removed = True
        if (
            len(message_history) > min_len
            and message_history[-1].get("role") == "assistant"
        ):
            message_history.pop()
            removed = True
        return removed

    # Flat safety margin (tokens) kept free below max_context_length when
    # estimating whether a summary request fits. A *flat* margin (not a
    # multiplier) covers tokenizer drift (tiktoken vs the server tokenizer) and
    # chat-template overhead without over-dropping: an inflated multiplier here
    # would discard far more history than physically necessary.
    _CONTEXT_BUDGET_SAFETY_MARGIN = 4096

    def enforce_context_budget(
        self, message_history: List[Dict], reserve_tokens: Optional[int] = None
    ) -> List[Dict]:
        """Trim the history just enough to fit the model's hard context window.

        Pops the **minimum** number of complete trailing rounds so that
        ``estimated_prompt + reserve_tokens + SAFETY < max_context_length``. The
        estimate is deliberately NOT inflated by a multiplier (tiktoken already
        counts the real text); only a small flat safety margin is applied. This
        is the cheap pre-trim that gets the summary request close to the window;
        the terminal-summary caller additionally retries on a real context-length
        400, dropping one more round at a time, so the true minimum is removed
        even if this estimate is slightly off.

        The leading system message (if any) and the first user message (the task)
        are always preserved. Works in every context-management mode.

        Args:
            message_history: history to shrink IN PLACE (also returned).
            reserve_tokens: tokens to keep free for prompt overhead + completion;
                defaults to ``max_tokens + 1000``.
        """
        if not message_history:
            return message_history
        if reserve_tokens is None:
            reserve_tokens = int(self.max_tokens) + 1000

        def _estimate_history(hist: List[Dict]) -> int:
            total = 0
            for m in hist:
                total += self._estimate_tokens(str(m.get("content", "")))
                # Assistant messages may carry preserved thinking that is sent
                # back to the server (preserve mode) — count it too.
                rc = m.get("_reasoning_content") or m.get("reasoning_content")
                if rc:
                    total += self._estimate_tokens(str(rc))
            return total

        # Never pop the leading system message or the first user message (task).
        min_len = 1
        for i, m in enumerate(message_history):
            if m.get("role") == "user":
                min_len = i + 1
                break

        budget = (
            int(self.max_context_length)
            - int(reserve_tokens)
            - self._CONTEXT_BUDGET_SAFETY_MARGIN
        )
        popped_rounds = 0
        while (
            _estimate_history(message_history) > budget
            and len(message_history) > min_len
        ):
            if not self.drop_last_round(message_history, min_len=min_len):
                break
            popped_rounds += 1

        if popped_rounds:
            self.task_log.log_step(
                "warning",
                "LLM | Context Budget Enforced",
                f"Dropped {popped_rounds} trailing tool-call round(s) to fit the "
                f"hard context window ({self.max_context_length} tokens) before "
                f"summarizing; history length now {len(message_history)}.",
            )
        return message_history

    @with_timeout(DEFAULT_LLM_TIMEOUT_SECONDS)
    async def create_message(
        self,
        system_prompt: str,
        message_history: List[Dict],
        tool_definitions: List[Dict],
        keep_tool_result: int = -1,
        step_id: int = 1,
        task_log: Optional["TaskLog"] = None,
        agent_type: str = "main",
    ) -> Tuple[Any, List[Dict]]:
        """
        Call LLM to generate a response with optional tool call support.

        This is the main entry point for LLM interactions. It handles:
        - Message history management
        - Tool result filtering based on keep_tool_result
        - Error handling and logging

        Args:
            system_prompt: System prompt to guide the LLM's behavior
            message_history: List of previous messages in the conversation
            tool_definitions: List of available tool definitions
            keep_tool_result: Number of recent tool results to keep (-1 = keep all)
            step_id: Current step identifier for logging
            task_log: Optional logger for task execution
            agent_type: Type of agent making the call ("main" or sub-agent name)

        Returns:
            Tuple of (response, updated_message_history)
        """
        # Unified LLM call processing
        try:
            response, message_history = await self._create_message(
                system_prompt,
                message_history,
                tool_definitions,
                keep_tool_result=keep_tool_result,
            )

        except Exception as e:
            self.task_log.log_step(
                "error",
                f"FATAL ERROR | {agent_type} | LLM Call ERROR",
                f"{agent_type} failed: {str(e)}",
            )
            response = None

        return response, message_history

    @staticmethod
    async def convert_tool_definition_to_tool_call(tools_definitions):
        """
        Convert MCP tool definitions to OpenAI function call format.

        Transforms the internal tool definition format used by MCP servers into
        the format expected by OpenAI's function calling API.

        Args:
            tools_definitions: List of server definitions, each containing a 'name'
                and 'tools' list with tool specifications.

        Returns:
            List of tool definitions in OpenAI function call format, where each
            tool name is prefixed with its server name (e.g., "server-name-tool-name").
        """
        tool_list = []
        for server in tools_definitions:
            if "tools" in server and len(server["tools"]) > 0:
                for tool in server["tools"]:
                    tool_def = dict(
                        type="function",
                        function=dict(
                            name=f"{server['name']}-{tool['name']}",
                            description=tool["description"],
                            parameters=tool["schema"],
                        ),
                    )
                    tool_list.append(tool_def)
        return tool_list

    def close(self):
        """Close client connection.

        Note: For async clients (AsyncOpenAI, AsyncAnthropic), the connection
        will be closed when the client object is garbage collected.
        For proper async cleanup, use `await client.aclose()` in an async context.
        """
        if hasattr(self.client, "close"):
            if asyncio.iscoroutinefunction(self.client.close):
                # For async clients, we cannot call close() synchronously.
                # The async HTTP client will be closed when garbage collected.
                # For explicit async cleanup, call aclose() from an async context.
                if hasattr(self.client, "_client"):
                    # Try to close the underlying httpx client if available
                    try:
                        self.client._client.close()
                    except Exception:
                        pass  # Ignore errors during cleanup
            else:
                self.client.close()
        elif hasattr(self.client, "_client") and hasattr(self.client._client, "close"):
            # Some clients may have internal _client attribute
            self.client._client.close()

    def _format_response_for_log(self, response) -> Dict:
        """Format response for logging"""
        if not response:
            return {}

        # Basic response information
        formatted = {
            "response_type": type(response).__name__,
        }

        # Anthropic response
        if hasattr(response, "content"):
            formatted["content"] = []
            for block in response.content:
                if hasattr(block, "type"):
                    if block.type == "text":
                        formatted["content"].append(
                            {
                                "type": "text",
                                "text": block.text[:500] + "..."
                                if len(block.text) > 500
                                else block.text,
                            }
                        )
                    elif block.type == "tool_use":
                        formatted["content"].append(
                            {
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": str(block.input)[:200] + "..."
                                if len(str(block.input)) > 200
                                else str(block.input),
                            }
                        )

        # OpenAI response
        if hasattr(response, "choices"):
            formatted["choices"] = []
            for choice in response.choices:
                choice_data = {"finish_reason": choice.finish_reason}
                if hasattr(choice, "message"):
                    message = choice.message
                    choice_data["message"] = {
                        "role": message.role,
                        "content": message.content[:500] + "..."
                        if message.content and len(message.content) > 500
                        else message.content,
                    }
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        choice_data["message"]["tool_calls_count"] = len(
                            message.tool_calls
                        )
                formatted["choices"].append(choice_data)

        return formatted
