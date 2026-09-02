# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Parsing utilities for LLM responses and tool calls.

This module provides functions for:
- Parsing tool calls from LLM responses (both OpenAI and MCP formats)
- Extracting text content from responses
- Safe JSON parsing with automatic repair
- Failure experience summary extraction
"""

import json
import logging
import re
from typing import Any, Dict, List, Union

from json_repair import repair_json

logger = logging.getLogger("miroflow_agent")


def parse_tool_server_mapping(system_prompt: str) -> dict:
    """
    Parse system prompt to extract tool_name → server_name mapping.

    Parses patterns like:
        ## Server name: tool-python
        ### Tool name: run_python_code

    Only extracts mappings for the tools that models commonly attribute to the
    wrong server. Both shipped tools are listed, since an agent config can put
    web_search and scrape_website on different servers (see
    conf/agent/serper_jina_search_agent.yaml).

    Args:
        system_prompt: The system prompt containing MCP tool definitions

    Returns:
        Dict mapping tool_name to correct server_name, e.g.
        {"web_search": "tool-serper-search", "scrape_website": "tool-jina-scrape"}
    """
    TARGET_TOOLS = {"web_search", "scrape_website"}
    mapping = {}
    current_server = None
    for line in system_prompt.split("\n"):
        server_match = re.match(r"## Server name:\s*(.+)", line)
        if server_match:
            current_server = server_match.group(1).strip()
            continue
        tool_match = re.match(r"### Tool name:\s*(.+)", line)
        if tool_match and current_server:
            tool_name = tool_match.group(1).strip()
            if tool_name in TARGET_TOOLS:
                mapping[tool_name] = current_server
    return mapping


# Module-level cache for tool_server_mapping
_tool_server_mapping: dict = {}


def set_tool_server_mapping(system_prompt: str) -> None:
    """
    Parse system prompt and cache the tool_name → server_name mapping.

    Should be called once when system prompt is available.

    Args:
        system_prompt: The system prompt containing MCP tool definitions
    """
    global _tool_server_mapping
    _tool_server_mapping = parse_tool_server_mapping(system_prompt)


def fix_server_name_in_text(text: str) -> str:
    """
    Fix incorrect server_name and tool_name in MCP XML tool calls.

    Uses the cached tool_server_mapping (parsed from the system prompt) to
    determine the correct server_name for each tool, for the tools in
    TARGET_TOOLS. mcp_xml mode only -- native_fc gets the server name from the
    API and never goes through here.

    Args:
        text: The LLM response text containing MCP tool calls

    Returns:
        Text with corrected server_name and tool_name if needed
    """
    if not isinstance(text, str):
        return text

    mapping = _tool_server_mapping
    if not mapping:
        return text

    # Legacy alias repair: some models emit tool_name=python / python_code.
    # Inert unless a run_python_code tool is actually exposed.
    if "run_python_code" in mapping:
        for wrong_name in ("python", "python_code"):
            tag = f"<tool_name>{wrong_name}</tool_name>"
            if tag in text:
                text = text.replace(tag, "<tool_name>run_python_code</tool_name>")

    # Fix server_name for each target tool using the mapping from system prompt
    for tool_name, correct_server in mapping.items():
        tool_tag = f"<tool_name>{tool_name}</tool_name>"
        if tool_tag not in text:
            continue
        correct_server_tag = f"<server_name>{correct_server}</server_name>"
        if correct_server_tag in text:
            continue
        text = re.sub(
            r"<server_name>[^<]+</server_name>(\s*" + re.escape(tool_tag) + r")",
            correct_server_tag + r"\1",
            text,
        )

    return text


def filter_none_values(arguments: Union[Dict, Any]) -> Union[Dict, Any]:
    """
    Filter out keys with None values from arguments dictionary.

    Args:
        arguments: A dictionary to filter, or any other value

    Returns:
        The filtered dictionary, or the original value if not a dict
    """
    if not isinstance(arguments, dict):
        return arguments
    return {k: v for k, v in arguments.items() if v is not None}


def _fix_backslash_escapes(json_str: str) -> str:
    """
    Fix common backslash escape issues in JSON strings.
    This handles cases where backslashes in string values are not properly escaped.

    Common issues:
    - Unescaped backslashes before non-escape characters

    Note: This is a conservative fix that preserves valid escape sequences
    (backslash, quote, slash, b, f, n, r, t) and only fixes clearly problematic cases.
    """
    fixed_str = json_str

    # Fix backslashes that are not part of valid escape sequences
    # Valid JSON escape sequences: \\, \", \/, \b, \f, \n, \r, \t, \uXXXX
    # Pattern: backslash not followed by a valid escape character
    # This regex matches \ followed by anything except valid escape chars
    # But we need to be careful not to match already-escaped backslashes (\\)

    # Strategy: Find all backslashes, but skip those that are:
    # 1. Already escaped (\\)
    # 2. Part of valid escape sequences (\", \/, \b, \f, \n, \r, \t, \u)

    # More conservative approach: Only fix backslashes before uppercase letters
    # (common in Windows paths) and other clearly problematic patterns
    # This avoids breaking valid JSON escape sequences

    # Fix backslashes before uppercase letters (Windows paths like C:\Users)
    fixed_str = re.sub(
        r"(?<!\\)\\([A-Z])",  # Backslash before uppercase letter, not already escaped
        r"\\\\\1",
        fixed_str,
    )

    # Fix backslashes before digits (common in paths like \1, \2)
    fixed_str = re.sub(
        r"(?<!\\)\\([0-9])",  # Backslash before digit, not already escaped
        r"\\\\\1",
        fixed_str,
    )

    # Fix other unescaped backslashes that are not part of valid escape sequences
    # This is more aggressive but should be safe after json_repair fails
    # Valid escape chars: \\, ", /, b, f, n, r, t, u
    # Use a capturing group to preserve the character after backslash
    fixed_str = re.sub(
        r'(?<!\\)\\([^\\"/bfnrtu])',  # Backslash followed by invalid escape char
        r"\\\\\1",  # Escape it and preserve the character
        fixed_str,
    )

    return fixed_str


def safe_json_loads(arguments_str: str) -> Dict[str, Any]:
    """
    Safely parse a JSON string with multiple fallbacks.

    Parsing strategy:
    1. Try standard json.loads()
    2. If it fails, try json_repair to fix common issues
    3. If all attempts fail, return an error object

    Args:
        arguments_str: JSON string to parse

    Returns:
        Parsed dictionary, or error dict with 'error' and 'raw' keys
    """
    # Step 1: Try standard JSON parsing
    try:
        return json.loads(arguments_str)
    except json.JSONDecodeError:
        pass

    # Step 2: Try json_repair to fix common issues
    try:
        repaired = repair_json(arguments_str, ensure_ascii=False)
        return json.loads(repaired)
    except Exception:
        logger.warning(f"Unable to parse JSON: {arguments_str}")

    # Step 3: Give up and return error information
    return {
        "error": "Failed to parse arguments",
        "raw": arguments_str,
    }


_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>")
_THINK_OPEN_RE = re.compile(r"<think>[\s\S]*$")

# Native function-calling tool-call syntax that reasoning/SFT models sometimes
# emit as LITERAL TEXT even when no tools are offered (Qwen-style
# ``<tool_call><function=NAME><parameter=k>v</parameter></function></tool_call>``).
# Both complete blocks and a trailing *unterminated* one (truncated output) are
# stripped so they never leak into a re-injected failure-experience summary.
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>[\s\S]*?</tool_call>", re.IGNORECASE)
_FUNCTION_BLOCK_RE = re.compile(r"<function=[\s\S]*?</function>", re.IGNORECASE)
_TOOL_CALL_OPEN_RE = re.compile(r"<tool_call>[\s\S]*$", re.IGNORECASE)
_FUNCTION_OPEN_RE = re.compile(r"<function=[\s\S]*$", re.IGNORECASE)
_MCP_TOOL_OPEN_RE = re.compile(r"<use_mcp_tool>[\s\S]*$", re.IGNORECASE)
# Residual bare tags left after block removal (stray </parameter>, <parameter=..>, ...)
_TOOL_RESIDUE_RE = re.compile(
    r"</?(?:tool_call|function|parameter|tool_response)\b[^>]*>", re.IGNORECASE
)


def strip_think_blocks(text: str) -> str:
    """Remove every ``<think>...</think>`` block (chain-of-thought) from text.

    Reasoning models (Qwen / Kimi / GLM, ...) emit ``<think>`` CoT inside
    ``content`` whenever ``reasoning_content_mode`` is not "preserve". That raw
    CoT must never leak into a re-injected failure-experience summary, so we
    strip ALL complete think blocks — not just the first — plus any trailing
    *unterminated* ``<think>`` left by a truncated response. Returns the
    remainder, stripped.
    """
    if not text:
        return ""
    text = _THINK_BLOCK_RE.sub("", text)  # all complete <think>...</think>
    text = _THINK_OPEN_RE.sub("", text)  # a dangling, unclosed <think>...
    # A prefill-continued think block can leave an ORPHAN close tag with no
    # matching open (e.g. "<continued reasoning></think>\n\nsummary" when a local
    # server continued the seeded prefix): drop everything up to and including it.
    orphan_close = text.find("</think>")
    if orphan_close != -1:
        text = text[orphan_close + len("</think>"):]
    return text.strip()


def strip_tool_call_blocks(text: str) -> str:
    """Remove tool-call syntax the model emits as LITERAL TEXT.

    Handles the native function-calling form
    (``<tool_call><function=NAME><parameter=k>v</parameter></function></tool_call>``)
    and the mcp_xml form (``<use_mcp_tool>...``), including complete blocks and a
    trailing *unterminated* one from a truncated response, plus any residual bare
    tags. Returns the remainder, stripped. Keeps a re-injected failure-experience
    summary free of the model's would-be tool calls.
    """
    if not text:
        return ""
    text = _TOOL_CALL_BLOCK_RE.sub("", text)  # complete <tool_call>...</tool_call>
    text = _FUNCTION_BLOCK_RE.sub("", text)  # complete <function=...>...</function>
    text = _TOOL_CALL_OPEN_RE.sub("", text)  # trailing/truncated <tool_call>...
    text = _FUNCTION_OPEN_RE.sub("", text)  # trailing/truncated <function=...
    text = _MCP_TOOL_OPEN_RE.sub("", text)  # trailing <use_mcp_tool>...
    text = _TOOL_RESIDUE_RE.sub("", text)  # stray bare tags
    return text.strip()


def extract_failure_experience_summary(text: str) -> str:
    """
    Extract the structured failure-experience summary from an LLM response.

    The raw ``text`` is ``FAILURE_SUMMARY_ASSISTANT_PREFIX`` (a canned
    ``<think>...</think>`` seed) followed by the model's output. For a reasoning
    model in a non-"preserve" mode the output itself contains a SECOND
    ``<think>...</think>`` block (the model's own CoT) before the structured
    summary. Possible shapes:

        "<think>{canned}</think>\n\n<think>{model CoT}</think>\n\n{summary}"
        "<think>{canned}</think>\n\n{summary}"           (preserve mode)
        "{summary}"                                        (no think block)
        "<think>{canned}</think>\n\n{summary}\n\n<use_mcp_tool>..."

    Every ``<think>`` block (canned seed AND model CoT) is stripped so ONLY the
    structured summary is returned — no chain-of-thought leaks into the summary
    re-injected into the next retry's task description. Think blocks are removed
    BEFORE the ``<use_mcp_tool>`` trailer is trimmed, so a tool mention inside a
    think block cannot truncate the real summary.

    Returns:
        - The structured content, with all think blocks and any ``<use_mcp_tool>``
          trailer removed.
        - If nothing survives (model put everything inside ``<think>``), falls
          back to the inner text of the LAST think block, so we never return "".
    """
    if not text:
        return ""

    # Inner text of complete think blocks — used only as a last-resort fallback.
    think_inner = re.findall(r"<think>([\s\S]*?)</think>", text)

    # Strip all think blocks first (so a <use_mcp_tool> mentioned inside a think
    # block cannot truncate the summary), then drop any tool-call trailer.
    without_think = strip_think_blocks(text)
    # Also strip native function-calling tool-call XML the model sometimes emits
    # as literal text (Qwen ``<tool_call><function=...>``) plus any mcp_xml
    # ``<use_mcp_tool>`` trailer — a "summary" that is really the model's next
    # search query must never be injected into the retry task description.
    content = strip_tool_call_blocks(without_think).strip()

    if content:
        return content
    # Structured summary empty — best-effort fallback to the last think block,
    # itself cleaned of any tool-call residue.
    if think_inner:
        return strip_tool_call_blocks(think_inner[-1]).strip()
    return ""


def is_usable_failure_summary(text: str) -> bool:
    """Return True if ``text`` is a real post-mortem summary rather than tool-call
    residue, empty, or degenerate.

    Usable = non-trivial after cleaning AND free of any tool-call / invocation
    syntax. It does NOT require exact section headers, so a free-form post-mortem
    (e.g. a ``## Post-Mortem`` block) still passes, while a response that was only
    ``<tool_call>...`` search queries — which cleans down to near-empty — is
    rejected.
    """
    if not text:
        return False
    t = text.strip()
    if len(t) < 25:
        return False
    if re.search(
        r"<tool_call>|<function=|<parameter=|<use_mcp_tool>", t, re.IGNORECASE
    ):
        return False
    return True


def extract_llm_response_text(llm_response: Union[str, Dict]) -> str:
    """
    Extract text from LLM response, excluding <use_mcp_tool> tags.

    Stops immediately when <use_mcp_tool> tag is encountered, returning
    only the content before it.

    Args:
        llm_response: Either a string or a dict with 'content' key

    Returns:
        Extracted text content, stripped of trailing whitespace
    """
    # If it's a dictionary type, extract the content field
    if isinstance(llm_response, dict):
        content = llm_response.get("content", "")
    else:
        # If it's a string type, use directly
        content = str(llm_response)

    # Find the position of <use_mcp_tool> tag
    tool_start_pattern = r"<use_mcp_tool>"
    match = re.search(tool_start_pattern, content)

    if match:
        # If <use_mcp_tool> tag is found, only return content before the tag
        return content[: match.start()].strip()
    else:
        # If no tag is found, return the complete content
        return content.strip()


def parse_llm_response_for_tool_calls(
    llm_response_content_text: Union[str, Dict, List],
) -> List[Dict[str, Any]]:
    """
    Parse tool calls from LLM response content.

    Supports multiple formats:
    - OpenAI Response API format (dict with 'output' containing function_call items)
    - OpenAI Completion API format (list of tool_call objects)
    - MCP format (<use_mcp_tool> XML tags in text)

    Args:
        llm_response_content_text: Response content in any supported format

    Returns:
        List of tool call dicts with keys: server_name, tool_name, arguments, id
    """
    # tool_calls or MCP reponse are handled differently
    # for openai response api, the tool_calls are in the response text
    if isinstance(llm_response_content_text, dict):
        tool_calls = []
        for item in llm_response_content_text.get("output") or []:
            if item.get("type") == "function_call":
                name = item.get("name", "")
                if "-" in name:
                    server_name, tool_name = name.rsplit("-", maxsplit=1)
                else:
                    server_name = "unknown"
                    tool_name = name
                arguments_str = item.get("arguments")
                arguments = safe_json_loads(arguments_str)
                arguments = filter_none_values(arguments)
                tool_calls.append(
                    dict(
                        server_name=server_name,
                        tool_name=tool_name,
                        arguments=arguments,
                        id=item.get("call_id"),
                    )
                )
        return tool_calls

    # for openai completion api, the tool_calls are in the response text
    if isinstance(llm_response_content_text, list):
        tool_calls = []
        for tool_call in llm_response_content_text:
            name = tool_call.function.name
            if "-" in name:
                server_name, tool_name = name.rsplit("-", maxsplit=1)
            else:
                server_name = "unknown"
                tool_name = name
            arguments_str = tool_call.function.arguments

            # Parse JSON string to dictionary
            try:
                # Try to handle possible newlines and escape characters
                arguments = json.loads(arguments_str)
            except json.JSONDecodeError:
                logger.info(
                    f"Warning: Unable to parse tool arguments JSON: {arguments_str}"
                )
                # Try more lenient parsing or log error
                try:
                    # Try to replace some common error formats, such as Python dict strings
                    arguments_str_fixed = (
                        arguments_str.replace("'", '"')
                        .replace("None", "null")
                        .replace("True", "true")
                        .replace("False", "false")
                    )
                    arguments = json.loads(arguments_str_fixed)
                    logger.info(
                        "Info: Successfully parsed arguments after attempting to fix."
                    )
                except json.JSONDecodeError:
                    logger.info(
                        f"Error: Still unable to parse tool arguments JSON after fixing: {arguments_str}"
                    )
                    arguments = {
                        "error": "Failed to parse arguments",
                        "raw": arguments_str,
                    }

            arguments = filter_none_values(arguments)
            tool_calls.append(
                dict(
                    server_name=server_name,
                    tool_name=tool_name,
                    arguments=arguments,
                    id=tool_call.id,
                )
            )
        return tool_calls

    # for other clients, such as qwen and anthropic, we use MCP instead of tool calls
    tool_calls = []
    # Find all <use_mcp_tool> tags
    tool_call_patterns = re.findall(
        r"<use_mcp_tool>\s*<server_name>(.*?)</server_name>\s*<tool_name>(.*?)</tool_name>\s*<arguments>\s*([\s\S]*?)\s*</arguments>\s*</use_mcp_tool>",
        llm_response_content_text,
        re.DOTALL,
    )

    for match in tool_call_patterns:
        server_name = match[0].strip()
        tool_name = match[1].strip()
        arguments_str = match[2].strip()

        # Parse JSON string to dictionary
        arguments = safe_json_loads(arguments_str)
        arguments = filter_none_values(arguments)

        tool_calls.append(
            {
                "server_name": server_name,
                "tool_name": tool_name,
                "arguments": arguments,
                "id": None,
            }
        )

    return tool_calls
