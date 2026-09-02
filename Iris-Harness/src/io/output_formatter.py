# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Output formatting utilities for agent responses."""

import re
from typing import Tuple

from ..utils.prompt_utils import FORMAT_ERROR_MESSAGE

# Maximum length for tool results before truncation (100k chars ≈ 25k tokens)
TOOL_RESULT_MAX_LENGTH = 100_000


class OutputFormatter:
    """Formatter for processing and formatting agent outputs."""

    def _extract_boxed_content(self, text: str) -> str:
        r"""
        Extract the content of the last \boxed{...} occurrence in the given text.

        Supports:
          - Arbitrary levels of nested braces
          - Escaped braces (\{ and \})
          - Whitespace between \boxed and the opening brace
          - Empty content inside braces
          - Incomplete boxed expressions (extracts to end of string as fallback)

        Args:
            text: Input text that may contain \boxed{...} expressions

        Returns:
            The extracted boxed content, or empty string if no match is found.
        """
        if not text:
            return ""

        _BOXED_RE = re.compile(r"\\boxed\b", re.DOTALL)

        last_result = None  # Track the last boxed content (complete or incomplete)
        i = 0
        n = len(text)

        while True:
            # Find the next \boxed occurrence
            m = _BOXED_RE.search(text, i)
            if not m:
                break
            j = m.end()

            # Skip any whitespace after \boxed
            while j < n and text[j].isspace():
                j += 1

            # Require that the next character is '{'
            if j >= n or text[j] != "{":
                i = j
                continue

            # Parse the brace content manually to handle nesting and escapes
            depth = 0
            k = j
            escaped = False
            found_closing = False
            while k < n:
                ch = text[k]
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    # When depth returns to zero, the boxed content ends
                    if depth == 0:
                        last_result = text[j + 1 : k]
                        i = k + 1
                        found_closing = True
                        break
                k += 1

            # If we didn't find a closing brace, this is an incomplete boxed
            # Store it as the last result (will be overwritten if we find more boxed later)
            if not found_closing and depth > 0:
                last_result = text[j + 1 : n]
                i = k  # Continue from where we stopped
            elif not found_closing:
                i = j + 1  # Move past this invalid boxed

        # Return the last boxed content found (complete or incomplete).
        #
        # A \boxed{} holding only a non-answer ("?", "...", "unknown") is the
        # model declining, not answering; treat it as no answer so it triggers a
        # format retry instead of being sent to the judge as a wrong answer.
        # Compare after stripping, or "\boxed{ unknown }" would slip through.
        NON_ANSWERS = {"?", "??", "???", "？", "……", "…", "...", "unknown"}
        if last_result is None:
            return ""
        stripped = last_result.strip()
        return "" if stripped.lower() in NON_ANSWERS else stripped

    def format_tool_result_for_user(self, tool_call_execution_result: dict) -> dict:
        """
        Format tool execution results to be fed back to LLM as user messages.

        Only includes necessary information (results or errors). Long results
        are truncated to TOOL_RESULT_MAX_LENGTH to prevent context overflow.

        Args:
            tool_call_execution_result: Dict containing server_name, tool_name,
                and either 'result' or 'error'.

        Returns:
            Dict with 'type' and 'text' keys suitable for LLM message content.
        """
        server_name = tool_call_execution_result["server_name"]
        tool_name = tool_call_execution_result["tool_name"]

        if "error" in tool_call_execution_result:
            # Provide concise error information to LLM
            content = f"Tool call to {tool_name} on {server_name} failed. Error: {tool_call_execution_result['error']}"
        elif "result" in tool_call_execution_result:
            # Provide the original output result of the tool
            content = tool_call_execution_result["result"]
            # Truncate overly long results to prevent context overflow
            if len(content) > TOOL_RESULT_MAX_LENGTH:
                content = content[:TOOL_RESULT_MAX_LENGTH] + "\n... [Result truncated]"
        else:
            content = f"Tool call to {tool_name} on {server_name} completed, but produced no specific output or result."

        return {"type": "text", "text": content}

    def format_final_summary_and_log(
        self, final_answer_text: str, client=None, answer_mode: str = "boxed"
    ) -> Tuple[str, str, str]:
        """
        Format final summary information, including answers and token statistics.

        Args:
            final_answer_text: The final answer text from the agent
            client: Optional LLM client for token usage statistics
            answer_mode: "boxed" (extract \\boxed{} content) or "direct" (use text as-is)

        Returns:
            Tuple of (summary_text, extracted_answer, usage_log)
        """
        summary_lines = []
        summary_lines.append("\n" + "=" * 30 + " Final Answer " + "=" * 30)
        summary_lines.append(final_answer_text)

        if answer_mode == "direct":
            # Direct mode: use the content text as the answer (no boxed parsing).
            # Strip <think>...</think> blocks that may be embedded when
            # reasoning_content_mode="context" (GLM, Qwen, DeepSeek, etc.).
            if final_answer_text:
                text = final_answer_text.strip()
                think_end = text.rfind("</think>")
                if think_end >= 0:
                    text = text[think_end + len("</think>"):].strip()
                extracted_answer = text if text else FORMAT_ERROR_MESSAGE
            else:
                extracted_answer = FORMAT_ERROR_MESSAGE
            summary_lines.append("\n" + "-" * 20 + " Extracted Result (direct) " + "-" * 20)
            # Show a preview in the summary log
            preview = extracted_answer[:200] + ("..." if len(extracted_answer) > 200 else "")
            summary_lines.append(preview)
        else:
            # Boxed mode: extract \boxed{} content (existing behavior)
            extracted_answer = self._extract_boxed_content(final_answer_text)

            summary_lines.append("\n" + "-" * 20 + " Extracted Result " + "-" * 20)

            if extracted_answer:
                summary_lines.append(extracted_answer)
            elif final_answer_text:
                summary_lines.append("No \\boxed{} content found.")
                extracted_answer = FORMAT_ERROR_MESSAGE

        # Token usage statistics and cost estimation - use client method
        if client and hasattr(client, "format_token_usage_summary"):
            token_summary_lines, log_string = client.format_token_usage_summary()
            summary_lines.extend(token_summary_lines)
        else:
            # If no client or client doesn't support it, use default format
            summary_lines.append("\n" + "-" * 20 + " Token Usage & Cost " + "-" * 20)
            summary_lines.append("Token usage information not available.")
            summary_lines.append("-" * (40 + len(" Token Usage & Cost ")))
            log_string = "Token usage information not available."

        return "\n".join(summary_lines), extracted_answer, log_string
