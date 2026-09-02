# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
OpenAI-compatible LLM client implementation.

This module provides the OpenAIClient class for interacting with OpenAI's API
and OpenAI-compatible endpoints (such as vLLM, Qwen, DeepSeek, etc.).

Features:
- Async and sync API support
- Automatic retry with exponential backoff
- Token usage tracking and context length management
- MCP tool call parsing and response processing
- Native function calling support (tool_call_mode="native_fc")
"""

import asyncio
import dataclasses
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import tiktoken
from openai import AsyncOpenAI, DefaultAsyncHttpxClient, DefaultHttpxClient, OpenAI

from ...utils.prompt_utils import generate_mcp_system_prompt
from ..base_client import BaseClient

logger = logging.getLogger("miroflow_agent")

# ---------------------------------------------------------------------------
# System prompt for native function calling mode (no XML tool descriptions)
# ---------------------------------------------------------------------------
_NATIVE_FC_SYSTEM_PROMPT_TEMPLATE = """In this environment you have access to a set of tools you can use to answer the user's question.

You only have access to the tools provided. You can use one or more tools per message, and will receive the results in the next response. You use tools step-by-step to accomplish a given task, with each tool-use informed by the result of the previous tool-use. Today is: {date}

# General Objective

You accomplish a given task iteratively, breaking it down into clear steps and working through them methodically.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``src`` into ``dst`` (in place) and return ``dst``.

    Nested dicts are merged key-by-key so that, e.g., merging
    ``{"chat_template_kwargs": {"thinking": True}}`` into an ``extra_body``
    that already holds ``{"chat_template_kwargs": {"foo": 1}}`` keeps both
    inner keys instead of clobbering the whole sub-dict.
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


# ---------------------------------------------------------------------------
# MCP tools → OpenAI function calling format converter
# ---------------------------------------------------------------------------

def _convert_mcp_tools_to_openai_fc(tools_definitions: List[Dict]) -> List[Dict]:
    """Convert MCP server tool definitions to OpenAI function calling format.

    MCP format (input):
        [{"name": "server-name", "tools": [{"name": "tool", "description": "...", "schema": {...}}]}]

    OpenAI function calling format (output):
        [{"type": "function", "function": {"name": "server-name__tool", "description": "...", "parameters": {...}}}]

    We use double-underscore ``__`` as separator between server_name and
    tool_name so that we can reliably split later (consistent with the
    Anthropic client convention).
    """
    fc_tools: List[Dict] = []
    if not tools_definitions:
        return fc_tools

    for server in tools_definitions:
        server_name = server.get("name", "unknown")
        for tool in server.get("tools", []):
            # Skip tools that failed to load (they only have 'error' key)
            if "error" in tool and "name" not in tool:
                continue
            tool_name = tool.get("name", "unknown")
            schema = tool.get("schema", {"type": "object", "properties": {}})
            # Ensure schema has required top-level keys for OpenAI
            if "type" not in schema:
                schema["type"] = "object"
            fc_tools.append({
                "type": "function",
                "function": {
                    "name": f"{server_name}__{tool_name}",
                    "description": tool.get("description", ""),
                    "parameters": schema,
                },
            })

    return fc_tools


@dataclasses.dataclass
class OpenAIClient(BaseClient):
    def _create_client(self) -> Union[AsyncOpenAI, OpenAI]:
        """Create LLM client.

        Supports both standard OpenAI and Azure-style OpenAI endpoints.
        When provider is 'azure', uses standard OpenAI client with custom
        headers (api-key instead of Authorization: Bearer) and passes
        api-version as httpx query parameter.
        """
        custom_headers = {"x-upstream-session-id": self.task_id}
        http_client_kwargs: Dict[str, Any] = {"headers": custom_headers}

        if self.provider == "azure":
            # Azure-style API uses "api-key" header instead of "Authorization: Bearer"
            custom_headers["api-key"] = self.api_key
            # Pass api-version as httpx query parameter so it's appended correctly
            # (embedding it in base_url breaks because the SDK appends a trailing slash)
            if self.api_version:
                http_client_kwargs["params"] = {"api-version": self.api_version}
            # Use a dummy api_key to satisfy OpenAI SDK validation
            # (actual auth is via the api-key header above)
            api_key = "azure-api-key-in-header"
        else:
            api_key = self.api_key

        if self.async_client:
            return AsyncOpenAI(
                api_key=api_key,
                base_url=self.base_url,
                http_client=DefaultAsyncHttpxClient(**http_client_kwargs),
            )
        else:
            return OpenAI(
                api_key=api_key,
                base_url=self.base_url,
                http_client=DefaultHttpxClient(**http_client_kwargs),
            )

    @property
    def _is_native_fc(self) -> bool:
        """Check if native function calling mode is enabled."""
        return self.tool_call_mode == "native_fc"

    def _update_token_usage(self, usage_data: Any) -> None:
        """Update cumulative token usage"""
        if usage_data:
            input_tokens = getattr(usage_data, "prompt_tokens", 0)
            output_tokens = getattr(usage_data, "completion_tokens", 0)
            prompt_tokens_details = getattr(usage_data, "prompt_tokens_details", None)
            if prompt_tokens_details:
                cached_tokens = (
                    getattr(prompt_tokens_details, "cached_tokens", None) or 0
                )
            else:
                cached_tokens = 0

            # Record token usage for the most recent call
            self.last_call_tokens = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
            }

            # OpenAI does not provide cache_creation_input_tokens
            self.token_usage["total_input_tokens"] += input_tokens
            self.token_usage["total_output_tokens"] += output_tokens
            self.token_usage["total_cache_read_input_tokens"] += cached_tokens

            self.task_log.log_step(
                "info",
                "LLM | Token Usage",
                f"Input: {self.token_usage['total_input_tokens']}, "
                f"Output: {self.token_usage['total_output_tokens']}",
            )

    async def _create_message(
        self,
        system_prompt: str,
        messages_history: List[Dict[str, Any]],
        tools_definitions,
        keep_tool_result: int = -1,
    ):
        """
        Send message to OpenAI API.

        In native_fc mode, tools are passed via the `tools` API parameter
        and the model returns structured `tool_calls` in the response.

        In mcp_xml mode (default), tools are described in the system prompt
        and the model outputs XML tags that are parsed by regex.
        """

        # Reset the per-call context-overflow flag: it must reflect ONLY this
        # call so the orchestrator can tell a context-window rejection (never
        # retry the identical request) apart from a transient error (do retry).
        self.context_length_exceeded = False

        # Create a copy for sending to LLM (to avoid modifying the original).
        # Filter out metadata fields (prefixed with "_") that are stored in
        # message_history for logging but must NOT be sent to the API.
        # e.g., "_reasoning_content" stores thinking traces for distillation.
        #
        # Two ways the prior turn's thinking is carried back to the model:
        #
        #   * "context" mode — the thinking is already embedded inside
        #     ``content`` as a canonical ``<think>...</think>`` block (see
        #     ``_apply_reasoning_to_content``), so we deliberately do NOT also
        #     lift ``_reasoning_content`` to ``reasoning_content`` — doing that
        #     would make Kimi/Qwen chat templates render the same thinking twice.
        #
        #   * "preserve" mode — native preserve_thinking (Kimi K2.6 / Qwen):
        #     ``content`` is kept clean (no ``<think>``) and the thinking is
        #     handed back via the dedicated ``reasoning_content`` field, exactly
        #     as the official API expects. The server chat template re-renders
        #     ``<think>`` from that field. Requires the matching server switch,
        #     supplied via ``thinking_extra_body`` (merged below).
        if self.reasoning_content_mode == "preserve":
            messages_for_llm = []
            for m in messages_history:
                new_m = {k: v for k, v in m.items() if not k.startswith("_")}
                _rc = m.get("_reasoning_content")
                if _rc and new_m.get("role") == "assistant":
                    new_m["reasoning_content"] = _rc
                messages_for_llm.append(new_m)
        else:
            messages_for_llm = [
                {k: v for k, v in m.items() if not k.startswith("_")}
                for m in messages_history
            ]

        # put the system prompt in the first message since OpenAI API does not support system prompt in
        if system_prompt:
            # Check if there's already a system or developer message
            if messages_for_llm and messages_for_llm[0]["role"] in [
                "system",
                "developer",
            ]:
                messages_for_llm[0] = {
                    "role": "system",
                    "content": system_prompt,
                }

            else:
                messages_for_llm.insert(
                    0,
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                )

        # Filter tool results to save tokens (only affects messages sent to LLM)
        messages_for_llm = self._remove_tool_result_from_messages(
            messages_for_llm, keep_tool_result
        )

        # Length-truncation retry loop with an INDEPENDENT budget.
        #
        # When a single response is cut off by max_tokens (finish_reason ==
        # "length") we re-issue the SAME request with a larger max_tokens
        # (×1.5, capped at max_context_length - prompt_tokens) up to
        # ``self.length_retry_max`` times. Only after exhausting that budget
        # (or running out of context headroom) do we return the truncated
        # response, which the orchestrator turns into a final summary.
        #
        # Transient errors (timeout / API error) and repeat-degeneration are
        # intentionally NOT retried in-client (they raise / return immediately,
        # same as before — the orchestrator handles turn-level retries).
        current_max_tokens = self.max_tokens
        length_attempt = 0
        # Track the last truncated response so we can return it if a
        # subsequent retry is killed by the outer @with_timeout or
        # CancelledError.  Without this, the timeout raises through
        # create_message → response=None → orchestrator retries the
        # turn with turn_count-=1 → identical request → infinite loop.
        _last_truncated_response = None

        while True:
            params = {
                "model": self.model_name,
                "temperature": self.temperature,
                "messages": messages_for_llm,
                "stream": False,
                "top_p": self.top_p,
                "presence_penalty": self.presence_penalty,
                "extra_body": {},
            }
            # Check if the model is GPT-5, and adjust the parameter accordingly
            if "gpt-5" in self.model_name:
                # Use 'max_completion_tokens' for GPT-5
                params["max_completion_tokens"] = current_max_tokens
            else:
                # Use 'max_tokens' for GPT-4 and other models
                params["max_tokens"] = current_max_tokens

            # Add repetition_penalty if it's not the default value
            if self.repetition_penalty != 1.0:
                params["extra_body"]["repetition_penalty"] = self.repetition_penalty

            # Add top_k via extra_body (supported by SGLang/vLLM/MaaS, not standard OpenAI)
            if self.top_k > 0:
                params["extra_body"]["top_k"] = self.top_k

            # Add min_p via extra_body (supported by SGLang/vLLM/MaaS, not standard OpenAI)
            if self.min_p > 0.0:
                params["extra_body"]["min_p"] = self.min_p

            if "deepseek-v3-1" in self.model_name:
                params["extra_body"]["thinking"] = {"type": "enabled"}

            # DeepSeek-V4-Pro: thinking is toggled by a top-level
            # extra_body.enable_thinking=true; the response then carries the
            # thinking text in message.reasoning_content and a token count in
            # usage.reasoning_tokens.
            if "deepseek-v4" in self.model_name and self.thinking_enabled:
                params["extra_body"]["enable_thinking"] = True

            # Enable (and, in "preserve" mode, retain) the model's thinking.
            # The exact switch is endpoint-specific and declared per-config via
            # ``thinking_extra_body`` (deep-merged so we don't clobber sibling
            # extra_body keys). Examples:
            #   Kimi MaaS:        {"thinking": {"type": "enabled", "keep": "all"}}
            #   Kimi/Qwen SGLang: {"chat_template_kwargs": {"thinking": true,
            #                                                "preserve_thinking": true}}
            # Falls back to the legacy qwen3 default when no config is supplied,
            # to preserve existing behaviour for configs not yet migrated.
            if self.thinking_extra_body:
                _deep_merge(params["extra_body"], self.thinking_extra_body)
            elif "qwen3" in self.model_name.lower():
                thinking_kwargs = {"enable_thinking": True}
                # Only set preserve_thinking when mode is "preserve" — this
                # routes thinking to the dedicated reasoning_content field for
                # the server chat template to re-render.  For other modes
                # (context/log_only/discard), leave thinking in content as
                # <think>...</think> tags, avoiding the roundtrip
                # extract→re-embed and potential double-wrapping risk.
                if self.reasoning_content_mode == "preserve":
                    thinking_kwargs["preserve_thinking"] = True
                params["extra_body"].setdefault("chat_template_kwargs", {}).update(
                    thinking_kwargs
                )

            # GPT-5 reasoning effort — controls internal reasoning depth
            # Only applied when reasoning_effort is explicitly set (not None)
            if self.reasoning_effort and "gpt-5" in self.model_name:
                params["reasoning_effort"] = self.reasoning_effort

            # --- Native function calling: pass tools via API parameter ---
            if self._is_native_fc and tools_definitions:
                fc_tools = _convert_mcp_tools_to_openai_fc(tools_definitions)
                if fc_tools:
                    params["tools"] = fc_tools
                    # Let the model decide whether to use tools or respond directly
                    params["tool_choice"] = "auto"

            # auto-detect if we need to continue from the last assistant message
            # Note: continue_final_message / add_generation_prompt are SGLang/vLLM
            # specific parameters; cloud APIs (DirectLLM, OpenAI, etc.) don't support them
            is_local_server = self.base_url and (
                "localhost" in self.base_url or "127.0.0.1" in self.base_url
            )
            if (
                is_local_server
                and messages_for_llm
                and messages_for_llm[-1].get("role") == "assistant"
            ):
                params["extra_body"]["continue_final_message"] = True
                params["extra_body"]["add_generation_prompt"] = False

            try:
                if self.async_client:
                    response = await self.client.chat.completions.create(**params)
                else:
                    response = self.client.chat.completions.create(**params)

                # Guard against malformed responses (e.g., proxy returning
                # an error JSON with HTTP 200 that SDK parses into a
                # ChatCompletion with choices=None or empty choices)
                if not response.choices:
                    raw_resp = str(getattr(response, "model_extra", None) or response)
                    raise RuntimeError(
                        f"API returned empty choices (possible proxy error). Raw response: {raw_resp[:500]}"
                    )

                # Update token count
                self._update_token_usage(getattr(response, "usage", None))
                self.task_log.log_step(
                    "info",
                    "LLM | Response Status",
                    f"{getattr(response.choices[0], 'finish_reason', 'N/A')}",
                )

                # Check if response was truncated due to length limit.
                # INDEPENDENT length-retry budget: re-issue with a larger
                # max_tokens (×1.5, capped at max_context_length - prompt_tokens)
                # up to self.length_retry_max times before returning truncated.
                finish_reason = getattr(response.choices[0], "finish_reason", None)
                if finish_reason == "length":
                    usage = getattr(response, "usage", None)
                    prompt_tokens = (
                        getattr(usage, "prompt_tokens", 0) if usage is not None else 0
                    ) or 0
                    # Measure truncated output length for diagnostics
                    _trunc_content = getattr(
                        getattr(response.choices[0], "message", None), "content", None
                    ) or ""
                    _trunc_len = len(_trunc_content)
                    next_max_tokens = self._grow_max_tokens_for_length(
                        current_max_tokens, prompt_tokens
                    )
                    if (
                        length_attempt < self.length_retry_max
                        and next_max_tokens > current_max_tokens
                    ):
                        # Save the truncated response before retrying — if the
                        # next retry is killed by @with_timeout or CancelledError
                        # we return this instead of None (which would cause the
                        # orchestrator to infinitely retry the same turn).
                        _last_truncated_response = response
                        length_attempt += 1
                        self.task_log.log_step(
                            "warning",
                            "LLM | Length Limit Reached",
                            f"Output truncated (finish_reason=length, "
                            f"truncated_output_chars={_trunc_len}). Length retry "
                            f"{length_attempt}/{self.length_retry_max}: max_tokens "
                            f"{current_max_tokens} -> {next_max_tokens} "
                            f"(prompt_tokens={prompt_tokens}). Re-generating...",
                        )
                        current_max_tokens = next_max_tokens
                        await asyncio.sleep(2)
                        continue
                    else:
                        reason = (
                            "no context headroom to grow max_tokens"
                            if next_max_tokens <= current_max_tokens
                            else f"exhausted {self.length_retry_max} length retries"
                        )
                        self.task_log.log_step(
                            "warning",
                            "LLM | Length Limit Reached - Returning Truncated Response",
                            f"Returning truncated response ({reason}); the "
                            f"orchestrator will turn it into a final summary.",
                        )
                        return response, messages_history

                # Check if the last 50 characters of the response appear more than 5 times.
                # Repeat-degeneration is logged but NOT retried in-client (preserves
                # the prior max_retries=1 behaviour); we return the response anyway.
                resp_content = ""
                if hasattr(response.choices[0], "message") and hasattr(
                    response.choices[0].message, "content"
                ):
                    resp_content = response.choices[0].message.content or ""
                else:
                    resp_content = getattr(response.choices[0], "text", "")

                if resp_content and len(resp_content) >= 50:
                    tail_50 = resp_content[-50:]
                    repeat_count = resp_content.count(tail_50)
                    if repeat_count > 5:
                        self.task_log.log_step(
                            "warning",
                            "LLM | Repeat Detected - Returning Anyway",
                            "Severe repeat detected (last 50 chars appeared over 5 "
                            "times). Returning response anyway.",
                        )

                # Success - return the original messages_history (not the filtered copy)
                # This ensures that the complete conversation history is preserved in logs
                return response, messages_history

            except (asyncio.TimeoutError, asyncio.CancelledError) as e:
                err_type = "Timeout" if isinstance(e, asyncio.TimeoutError) else "Cancelled"
                if _last_truncated_response is not None:
                    # A previous length-retry produced a valid (truncated)
                    # response.  Return it so the orchestrator can turn it
                    # into a final summary instead of retrying infinitely.
                    self.task_log.log_step(
                        "warning",
                        f"LLM | {err_type} During Length Retry - Returning Last Truncated",
                        f"Request {err_type.lower()} during length retry "
                        f"{length_attempt}/{self.length_retry_max}. Returning the "
                        f"last truncated response for the orchestrator to summarize.",
                    )
                    return _last_truncated_response, messages_history
                self.task_log.log_step(
                    "error",
                    f"LLM | {err_type} Error",
                    f"{err_type} error: {str(e)}",
                )
                raise e
            except Exception as e:
                if self.is_context_length_error(e):
                    # Hard context-window overflow. Flag it so the orchestrator
                    # ENDS the loop and summarizes instead of blindly re-sending
                    # the identical over-limit request (which 400s forever). We
                    # still raise → the create_message wrapper returns response=
                    # None → handle_llm_call returns empty → the orchestrator's
                    # empty-response branch inspects the flag.
                    self.context_length_exceeded = True
                    self.task_log.log_step(
                        "error",
                        "LLM | Context Length Error",
                        f"Error: {str(e)}",
                    )
                else:
                    self.task_log.log_step(
                        "error",
                        "LLM | API Error",
                        f"Error: {str(e)}",
                    )
                raise e

    @staticmethod
    def _strip_thinking_content(text: str) -> str:
        """Strip thinking/reasoning content from model response.

        Models like Qwen3.5 and Kimi-K2.5 may output thinking content wrapped
        in <think>...</think> tags. When no --reasoning-parser is configured in
        SGLang/vLLM, these tags appear directly in the message content field.

        Handles patterns:
        - "thinking text</think>actual response"  (no opening <think>)
        - "<think>thinking text</think>actual response"  (with opening tag)

        If no </think> tag is found, the text is returned unchanged.
        """
        if not text:
            return text
        think_end_idx = text.rfind("</think>")
        if think_end_idx >= 0:
            return text[think_end_idx + len("</think>") :].strip()
        return text

    def _extract_reasoning_content(self, message: Any) -> Optional[str]:
        """Extract reasoning/thinking content from the API response message.

        SGLang with --reasoning-parser separates thinking into a dedicated
        `reasoning_content` field. Some providers use model_extra for
        non-standard fields. Also handles <think>...</think> tags in the
        raw content when no reasoning parser is configured.

        Returns the reasoning text if found, otherwise None.
        Controlled by reasoning_content_mode: returns None in "discard" mode.
        """
        if self.reasoning_content_mode == "discard":
            return None

        # 1. Direct attribute (SGLang / DeepSeek native)
        #    - SGLang uses "reasoning_content"
        #    - vLLM with --reasoning-parser uses "reasoning"
        reasoning = getattr(message, "reasoning_content", None)
        if not reasoning:
            reasoning = getattr(message, "reasoning", None)
        if reasoning:
            return reasoning

        # 2. model_extra dict (pydantic v2 overflow fields)
        extras = getattr(message, "model_extra", None) or {}
        reasoning = extras.get("reasoning_content") or extras.get("reasoning")
        if reasoning:
            return reasoning

        # 3. Fallback: extract from <think>...</think> in raw content
        #    (when no --reasoning-parser is configured on the server)
        raw_content = message.content or ""
        think_end = raw_content.rfind("</think>")
        if think_end >= 0:
            think_start = raw_content.find("<think>")
            if think_start >= 0:
                return raw_content[think_start + len("<think>"):think_end].strip()
            else:
                # No opening tag — everything before </think> is thinking
                return raw_content[:think_end].strip()

        return None

    def _apply_reasoning_to_content(
        self, text: str, reasoning: Optional[str], raw_content: str
    ) -> str:
        """Apply reasoning_content_mode to the assistant response text.

        In "discard" / "log_only" / "preserve" mode: strip <think> tags from
        content. ("preserve" keeps the thinking clean in content because it is
        carried back to the model via the separate ``reasoning_content`` field
        — see ``_create_message`` — so embedding <think> here would duplicate it.)
        In "context" mode: keep thinking in content so the LLM can see its
        own prior reasoning in subsequent turns (GLM-5 ri-style trajectory).
        If reasoning was extracted from a separate API field (e.g. SGLang
        --reasoning-parser), reconstruct <think> tags in content.

        IMPORTANT (context mode):
          The output is always normalised to the canonical form
          ``<think>\\n{reasoning}\\n</think>\\n\\n{rest}`` so downstream
          chat templates (Kimi, Qwen) can render it correctly. Specifically
          we handle the common case where SGLang has already injected a
          ``<think>`` prefix via ``add_generation_prompt``, so the model
          only emits ``{thinking}</think>{rest}`` (missing the opening tag).
          Returning that raw string here would cause the next chat-template
          render to wrap it in another ``<think></think>`` and treat the
          previous-turn thinking as content (Kimi) or leave a dangling
          ``</think>`` token in the conversation.
        """
        if self.reasoning_content_mode != "context":
            # discard / log_only: always strip
            return self._strip_thinking_content(text)

        # --- context mode: keep thinking in content ---
        if not reasoning:
            # No reasoning extracted; nothing to reconstruct.
            # Still strip stray tags just in case.
            return text

        # Strip *any* existing <think>...</think> region from the text so we
        # can rebuild it canonically. This makes the output deterministic
        # regardless of which of the following SGLang/vLLM serving variants
        # produced the raw content:
        #   1) "<think>{thinking}</think>{rest}"  — full tags, ideal case
        #   2) "{thinking}</think>{rest}"         — missing opening tag
        #                                            (chat_template injected
        #                                             <think> via
        #                                             add_generation_prompt)
        #   3) "{rest}" + reasoning in a separate field (true parser path)
        rest = text
        if "</think>" in rest:
            rest = rest.split("</think>", 1)[1]
        if "<think>" in rest:
            # Defensive: in case multiple <think>...</think> regions appear
            rest = rest.split("</think>")[-1]
        rest = rest.lstrip("\n")

        cleaned_reasoning = reasoning.strip("\n")
        if not rest:
            return f"<think>\n{cleaned_reasoning}\n</think>"
        return f"<think>\n{cleaned_reasoning}\n</think>\n\n{rest}"

    def _attach_reasoning(self, msg: Dict, reasoning: Optional[str]) -> None:
        """Attach reasoning content to a message dict as metadata.

        The "_reasoning_content" key uses underscore prefix convention:
        fields starting with "_" are filtered out when building API
        requests (see _create_message), but preserved in message_history
        for log serialization → distillation training data.
        """
        if reasoning:
            msg["_reasoning_content"] = reasoning

    def process_llm_response(
        self, llm_response: Any, message_history: List[Dict], agent_type: str = "main"
    ) -> tuple[str, bool, List[Dict]]:
        """Process LLM response.

        In native_fc mode, handles finish_reason="tool_calls" by storing
        the assistant message with structured tool_calls in the history.

        Reasoning/thinking content is handled according to reasoning_content_mode:
        - "discard": strip thinking, don't save metadata
        - "log_only": strip thinking from content, save to "_reasoning_content"
        - "context": keep thinking in content (GLM-5 ri-style), also save metadata
        The "_reasoning_content" metadata field is NOT sent back to the API
        (filtered by underscore prefix convention in _create_message) but IS
        preserved in the saved log JSON files for distillation.
        """
        if not llm_response or not llm_response.choices:
            error_msg = "LLM did not return a valid response."
            self.task_log.log_step(
                "error", "LLM | Response Error", f"Error: {error_msg}"
            )
            return "", True, message_history  # Exit loop, return message_history

        # Extract LLM response text
        from ...utils.parsing_utils import fix_server_name_in_text

        finish_reason = llm_response.choices[0].finish_reason

        # Normalize finish_reason: some proxies (e.g., Anthropic via Azure)
        # may return "end_turn" instead of the OpenAI-standard "stop"
        if finish_reason == "end_turn":
            finish_reason = "stop"

        message = llm_response.choices[0].message
        raw_content = message.content or ""

        # Extract reasoning content (thinking traces)
        # Returns None in "discard" mode; actual content in "log_only"/"context".
        reasoning = self._extract_reasoning_content(message)

        # --- Native function calling: handle "tool_calls" finish reason ---
        if self._is_native_fc and finish_reason == "tool_calls":
            assistant_response_text = self._apply_reasoning_to_content(
                raw_content, reasoning, raw_content
            )

            # Separate the two roles of assistant_response_text:
            #   1. message_history["content"] → sent back to the API next turn.
            #      Store the REAL content (empty when model returned only tool_calls).
            #      The OpenAI API accepts content="" or null for tool-calling messages.
            #   2. Return value → orchestrator flow control.
            #      Must be non-empty so the orchestrator proceeds to tool execution
            #      (empty string triggers the `else: continue` retry branch).
            history_content = assistant_response_text
            if not assistant_response_text:
                assistant_response_text = "[Calling tools]\n"

            # Build assistant message with tool_calls for message history
            # (required by OpenAI API for subsequent tool result messages)
            assistant_msg = {"role": "assistant", "content": history_content}
            self._attach_reasoning(assistant_msg, reasoning)
            if message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]
            message_history.append(assistant_msg)

            # Return should_break=False because the model wants to call tools
            return assistant_response_text, False, message_history

        # --- Also handle "stop" with tool_calls present (some providers) ---
        if self._is_native_fc and finish_reason == "stop" and message.tool_calls:
            assistant_response_text = self._apply_reasoning_to_content(
                raw_content, reasoning, raw_content
            )

            history_content = assistant_response_text
            if not assistant_response_text:
                assistant_response_text = "[Calling tools]\n"

            assistant_msg = {"role": "assistant", "content": history_content}
            self._attach_reasoning(assistant_msg, reasoning)
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
            message_history.append(assistant_msg)

            return assistant_response_text, False, message_history

        # --- Standard text responses (stop / length) ---
        if finish_reason == "stop":
            assistant_response_text = raw_content
            if not self._is_native_fc:
                assistant_response_text = fix_server_name_in_text(assistant_response_text)
            assistant_response_text = self._apply_reasoning_to_content(
                assistant_response_text, reasoning, raw_content
            )

            assistant_msg = {"role": "assistant", "content": assistant_response_text}
            self._attach_reasoning(assistant_msg, reasoning)
            message_history.append(assistant_msg)

        elif finish_reason == "length":
            assistant_response_text = raw_content
            if not self._is_native_fc:
                assistant_response_text = fix_server_name_in_text(assistant_response_text)
            assistant_response_text = self._apply_reasoning_to_content(
                assistant_response_text, reasoning, raw_content
            )
            if assistant_response_text == "":
                assistant_response_text = "LLM response is empty."
            elif "Context length exceeded" in assistant_response_text:
                # This is the case where context length is exceeded, needs special handling
                self.task_log.log_step(
                    "warning",
                    "LLM | Context Length",
                    "Detected context length exceeded, returning error status",
                )
                assistant_msg = {"role": "assistant", "content": assistant_response_text}
                self._attach_reasoning(assistant_msg, reasoning)
                message_history.append(assistant_msg)
                return (
                    assistant_response_text,
                    True,
                    message_history,
                )  # Return True to indicate need to exit loop

            # Add assistant response to history
            assistant_msg = {"role": "assistant", "content": assistant_response_text}
            self._attach_reasoning(assistant_msg, reasoning)
            message_history.append(assistant_msg)

        else:
            raise ValueError(
                f"Unsupported finish reason: {finish_reason}"
            )

        return assistant_response_text, False, message_history

    def extract_tool_calls_info(
        self, llm_response: Any, assistant_response_text: str
    ) -> List[Dict]:
        """Extract tool call information from LLM response.

        In native_fc mode, extracts tool calls from the structured
        response.choices[0].message.tool_calls field.

        In mcp_xml mode, parses XML tags from the text response.
        """
        # --- Native function calling: extract from structured tool_calls ---
        if self._is_native_fc and llm_response and llm_response.choices:
            message = llm_response.choices[0].message
            if message.tool_calls:
                tool_calls = []
                for tc in message.tool_calls:
                    combined_name = tc.function.name
                    arguments_str = tc.function.arguments

                    # Split "server_name__tool_name" (consistent with Anthropic convention)
                    if "__" in combined_name:
                        server_name, tool_name = combined_name.split("__", 1)
                    else:
                        server_name = "unknown"
                        tool_name = combined_name

                    # Parse arguments JSON
                    try:
                        arguments = json.loads(arguments_str) if arguments_str else {}
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Failed to parse tool arguments: {arguments_str}"
                        )
                        try:
                            # Try fixing common issues
                            fixed = (
                                arguments_str.replace("'", '"')
                                .replace("None", "null")
                                .replace("True", "true")
                                .replace("False", "false")
                            )
                            arguments = json.loads(fixed)
                        except json.JSONDecodeError:
                            arguments = {
                                "error": "Failed to parse arguments",
                                "raw": arguments_str,
                            }

                    # Unwrap nested "arguments" key produced by some models
                    # (e.g. DeepSeek-V4-Pro via vLLM may return
                    #  {"arguments": "{\"q\": \"...\", \"count\": \"10\"}"})
                    if (
                        isinstance(arguments, dict)
                        and len(arguments) == 1
                        and "arguments" in arguments
                    ):
                        inner = arguments["arguments"]
                        if isinstance(inner, str):
                            try:
                                arguments = json.loads(inner)
                            except json.JSONDecodeError:
                                pass  # keep original
                        elif isinstance(inner, dict):
                            arguments = inner

                    tool_calls.append({
                        "server_name": server_name,
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "id": tc.id,
                    })

                if tool_calls:
                    return tool_calls

        # --- MCP XML mode: parse from text ---
        from ...utils.parsing_utils import parse_llm_response_for_tool_calls
        return parse_llm_response_for_tool_calls(assistant_response_text)

    def update_message_history(
        self, message_history: List[Dict], all_tool_results_content_with_id: List[Tuple]
    ) -> List[Dict]:
        """Update message history with tool results.

        In native_fc mode, each tool result is sent as a separate message
        with role="tool" and the corresponding tool_call_id, as required
        by the OpenAI API.

        In mcp_xml mode, all tool results are merged into a single "user"
        message (backward compatible).
        """
        if self._is_native_fc:
            # --- Native function calling: use role="tool" messages ---
            # Get tool_call IDs from the last assistant message
            last_assistant_msg = None
            for msg in reversed(message_history):
                if msg.get("role") == "assistant":
                    last_assistant_msg = msg
                    break

            tool_call_ids = []
            if last_assistant_msg and "tool_calls" in last_assistant_msg:
                tool_call_ids = [
                    tc["id"] for tc in last_assistant_msg["tool_calls"]
                ]

            for i, (call_id, result_content) in enumerate(all_tool_results_content_with_id):
                # Match tool_call_id: prefer the call_id from execution,
                # fall back to the assistant message's tool_call IDs by position
                if call_id and any(call_id == tid for tid in tool_call_ids):
                    use_id = call_id
                elif i < len(tool_call_ids):
                    use_id = tool_call_ids[i]
                else:
                    use_id = call_id or f"call_{i}"

                result_text = (
                    result_content.get("text", "")
                    if isinstance(result_content, dict)
                    else str(result_content)
                )

                message_history.append({
                    "role": "tool",
                    "tool_call_id": use_id,
                    "content": result_text,
                })

            return message_history

        # --- MCP XML mode: merge into a single user message ---
        merged_text = "\n".join(
            [
                item[1]["text"]
                for item in all_tool_results_content_with_id
                if item[1]["type"] == "text"
            ]
        )

        message_history.append(
            {
                "role": "user",
                "content": merged_text,
            }
        )

        return message_history

    def generate_agent_system_prompt(self, date: Any, mcp_servers: List[Dict]) -> str:
        """Generate the system prompt for the agent.

        In native_fc mode, uses a simplified system prompt that does NOT
        include XML tool-use formatting instructions — because tools are
        provided via the API `tools` parameter instead.

        In mcp_xml mode, uses the full MCP system prompt with tool
        definitions and XML formatting instructions.
        """
        from ...utils.parsing_utils import set_tool_server_mapping

        if self._is_native_fc:
            # Still call set_tool_server_mapping for server name correction in logs
            full_mcp_prompt = generate_mcp_system_prompt(date, mcp_servers)
            set_tool_server_mapping(full_mcp_prompt)

            # Use a simpler prompt without XML tool descriptions
            if mcp_servers:
                formatted_date = (
                    date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
                )
                return _NATIVE_FC_SYSTEM_PROMPT_TEMPLATE.format(date=formatted_date)
            else:
                return full_mcp_prompt
        else:
            # Original MCP XML mode
            prompt = generate_mcp_system_prompt(date, mcp_servers)
            set_tool_server_mapping(prompt)
            return prompt

    def _estimate_tokens(self, text: str) -> int:
        """Use tiktoken to estimate the number of tokens in text"""
        if not hasattr(self, "encoding"):
            # Initialize tiktoken encoder
            try:
                self.encoding = tiktoken.get_encoding("o200k_base")
            except Exception:
                # If o200k_base is not available, use cl100k_base as fallback
                self.encoding = tiktoken.get_encoding("cl100k_base")

        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            # If encoding fails, use simple estimation: approximately 1 token per 4 characters
            self.task_log.log_step(
                "error",
                "LLM | Token Estimation Error",
                f"Error: {str(e)}",
            )
            return len(text) // 4

    def ensure_summary_context(
        self, message_history: list, summary_prompt: str
    ) -> tuple[bool, list]:
        """
        Check if current message_history + summary_prompt will exceed context
        If it will exceed, remove the last assistant-user pair and return False
        Return True to continue, False if messages have been rolled back
        """
        # Get token usage from the last LLM call
        last_prompt_tokens = self.last_call_tokens.get("prompt_tokens", 0)
        last_completion_tokens = self.last_call_tokens.get("completion_tokens", 0)
        buffer_factor = 1.5

        # Calculate token count for summary prompt
        summary_tokens = int(self._estimate_tokens(summary_prompt) * buffer_factor)

        # Sum tokens for EVERY message appended since the last LLM call — i.e.
        # every trailing tool/user message up to (but not including) the last
        # assistant message. This is the whole set of tool results produced by
        # the previous turn.
        #
        # ⚠ Why summing all of them matters: in native function-calling the model
        # can emit MANY tool calls in a single turn, and each one comes back as a
        # separate role="tool" message. Counting only ``message_history[-1]`` (one
        # tool result) badly under-estimates the next prompt, so the guard fails
        # to fire and the next request blows past the context window (400 "maximum
        # context length exceeded"). Walking back to the last assistant sums the
        # complete new round while never double-counting earlier turns, which are
        # already reflected in ``last_prompt_tokens``.
        new_content_tokens = 0
        for msg in reversed(message_history):
            if msg.get("role") in ("user", "tool"):
                new_content_tokens += int(
                    self._estimate_tokens(str(msg.get("content", ""))) * buffer_factor
                )
            else:
                # Reached the assistant/system message that opened this round —
                # everything from here back is already in last_prompt_tokens.
                break

        # Calculate total token count: last prompt + completion + this turn's new
        # tool results + summary + reserved response space
        estimated_total = (
            last_prompt_tokens
            + last_completion_tokens
            + new_content_tokens
            + summary_tokens
            + self.max_tokens
            + 1000  # Add 1000 tokens as buffer
        )

        if estimated_total >= self.max_context_length:
            self.task_log.log_step(
                "info",
                "LLM | Context Limit Reached",
                "Context limit reached, proceeding to step back and summarize the conversation",
            )

            # In native_fc mode, we need to remove all tool result messages
            # from the last tool call round, plus the assistant message
            if self._is_native_fc:
                # Remove trailing tool messages
                while message_history and message_history[-1].get("role") == "tool":
                    message_history.pop()
                # Remove the assistant message with tool_calls
                if message_history and message_history[-1].get("role") == "assistant":
                    message_history.pop()
            else:
                # MCP XML mode: remove the last user message (tool call results)
                if message_history and message_history[-1]["role"] == "user":
                    message_history.pop()
                # Remove the second-to-last assistant message (tool call request)
                if message_history and message_history[-1]["role"] == "assistant":
                    message_history.pop()

            self.task_log.log_step(
                "info",
                "LLM | Context Limit Reached",
                f"Removed the last assistant-tool pair, current message_history length: {len(message_history)}",
            )

            return False, message_history

        self.task_log.log_step(
            "info",
            "LLM | Context Limit Not Reached",
            f"{estimated_total}/{self.max_context_length}",
        )
        return True, message_history

    def format_token_usage_summary(self) -> tuple[List[str], str]:
        """Format token usage statistics, return summary_lines for format_final_summary and log string"""
        token_usage = self.get_token_usage()

        total_input = token_usage.get("total_input_tokens", 0)
        total_output = token_usage.get("total_output_tokens", 0)
        cache_input = token_usage.get("total_cache_input_tokens", 0)

        summary_lines = []
        summary_lines.append("\n" + "-" * 20 + " Token Usage " + "-" * 20)
        summary_lines.append(f"Total Input Tokens: {total_input}")
        summary_lines.append(f"Total Cache Input Tokens: {cache_input}")
        summary_lines.append(f"Total Output Tokens: {total_output}")
        summary_lines.append("-" * (40 + len(" Token Usage ")))
        summary_lines.append("Pricing is disabled - no cost information available")
        summary_lines.append("-" * (40 + len(" Token Usage ")))

        # Generate log string
        log_string = (
            f"[{self.model_name}] Total Input: {total_input}, "
            f"Cache Input: {cache_input}, "
            f"Output: {total_output}"
        )

        return summary_lines, log_string

    def get_token_usage(self):
        return self.token_usage.copy()
