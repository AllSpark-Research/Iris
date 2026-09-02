# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Prompt templates and utilities for agent system prompts.

This module provides:
- System prompt generation for MCP tool usage
- Agent-specific prompt generation (main agent, browsing agent)
- Summary prompt templates for final answer generation
- Failure experience templates for retry mechanisms
"""

# ============================================================================
# Format Error Messages
# ============================================================================

FORMAT_ERROR_MESSAGE = "No \\boxed{} content found in the final answer."

# Returned in place of an answer when the episode itself blew up (tool server
# failed to spawn, endpoint unreachable, unhandled exception). It must stay
# distinct from FORMAT_ERROR_MESSAGE: a format error is the model's failure and
# is worth retrying, whereas this is ours and must never be scored as a wrong
# answer.
PIPELINE_ERROR_MESSAGE = "Task execution failed before an answer was produced."

# ============================================================================
# Failure Experience Templates (for format error retry)
# ============================================================================

# Header that appears once before all failure experiences
FAILURE_EXPERIENCE_HEADER = """

=== Previous Attempts Analysis ===
The following summarizes what was tried before and why it didn't work. Use this to guide a NEW approach.

"""

# Template for each individual failure experience (used multiple times)
FAILURE_EXPERIENCE_ITEM = """[Attempt {attempt_number}]
{failure_summary}

"""

# Footer that appears once after all failure experiences
FAILURE_EXPERIENCE_FOOTER = """=== End of Analysis ===

Based on the above, you should try a different strategy this time.
"""

FAILURE_SUMMARY_PROMPT = """The task attempt has ended. Write a brief post-mortem so the NEXT attempt can try a better strategy.

STRICT OUTPUT RULES (read carefully):
- You MUST NOT call any tool, run any search/scrape/code, or output ANY tool-call syntax. Do NOT emit <tool_call>, <function=...>, <parameter=...>, <use_mcp_tool>, or JSON tool arguments. Emitting a tool call here is a mistake and the reply will be discarded.
- Answer ONLY from the conversation above and your own knowledge.
- Reply with ONLY the three sections below, as plain prose. Begin your reply immediately with the literal text "Failure type:".

Failure type: [incomplete / blocked / misdirected / format_missed]
  - incomplete: ran out of turns before finishing
  - blocked: got stuck due to tool failure or missing information
  - misdirected: went down the wrong path
  - format_missed: found the answer but forgot to use \\boxed{}
What happened: [1-3 sentences: the approach taken and why a final answer was not reached]
Useful findings: [concrete facts, URLs, candidate answers, or dead-ends discovered that the next attempt should reuse or avoid; write "None" if nothing useful]"""

# Assistant prefix for failure summary generation (primes the structured format and
# explicitly forbids tool-call syntax, which SFT search models tend to emit here).
FAILURE_SUMMARY_THINK_CONTENT = """I must write a structured post-mortem now. I will NOT call any tool and will NOT output any tool-call syntax (no <tool_call>, <function=...>, <parameter=...>, <use_mcp_tool>). I will answer only from the conversation above, in plain prose, using exactly these three sections and starting with "Failure type:":

* Failure type: one of incomplete / blocked / misdirected / format_missed
* What happened: the approach taken and why it didn't reach a final answer
* Useful findings: concrete facts, URLs, candidate answers, or dead-ends to reuse or avoid"""

FAILURE_SUMMARY_ASSISTANT_PREFIX = (
    f"<think>\n{FAILURE_SUMMARY_THINK_CONTENT}\n</think>\n\n"
)

# Corrective re-ask used ONCE when the first failure-summary reply was a tool call
# or otherwise not a usable post-mortem. Bounded (single retry) to protect eval
# efficiency — never loops.
FAILURE_SUMMARY_RETRY_PROMPT = """Your previous reply was invalid: it attempted to call a tool or did not contain a post-mortem. You CANNOT call tools now and MUST NOT output any tool-call syntax (no <tool_call>, <function=...>, <parameter=...>, <use_mcp_tool>).

Reply with ONLY these three sections, as plain prose, beginning immediately with "Failure type:":
Failure type: [incomplete / blocked / misdirected / format_missed]
What happened: [1-3 sentences]
Useful findings: [concrete facts / URLs / candidate answers / dead-ends, or "None"]"""

# Deterministic fallback injected when the model still cannot produce a usable
# summary after the corrective re-ask. Guarantees the next attempt's task
# description never receives raw tool-call text.
FAILURE_SUMMARY_FALLBACK = (
    "Failure type: incomplete\n"
    "What happened: The previous attempt ended without a usable post-mortem "
    "(the model did not produce a structured summary).\n"
    "Useful findings: None recorded. Try a substantially different search strategy "
    "and different query phrasings this time."
)

# ============================================================================
# MCP Tags for Parsing
# ============================================================================

mcp_tags = [
    "<use_mcp_tool>",
    "</use_mcp_tool>",
    "<server_name>",
    "</server_name>",
    "<arguments>",
    "</arguments>",
]

refusal_keywords = [
    "time constraint",
    "I’m sorry, but I can’t",
    "I'm sorry, I cannot solve",
]


def generate_mcp_system_prompt(date, mcp_servers):
    """
    Generate the MCP (Model Context Protocol) system prompt for LLM.

    Creates a structured prompt that instructs the LLM on how to use available
    MCP tools. Includes tool definitions, XML formatting instructions, and
    general task-solving guidelines.

    Args:
        date: Current date object for timestamp inclusion
        mcp_servers: List of server definitions, each containing 'name' and 'tools'

    Returns:
        Complete system prompt string with tool definitions and usage instructions
    """
    formatted_date = date.strftime("%Y-%m-%d")

    # Start building the template, now follows https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview#tool-use-system-prompt
    template = f"""In this environment you have access to a set of tools you can use to answer the user's question. 

You only have access to the tools provided below. You can only use one tool per message, and will receive the result of that tool in the user's next response. You use tools step-by-step to accomplish a given task, with each tool-use informed by the result of the previous tool-use. Today is: {formatted_date}

# Tool-Use Formatting Instructions 

Tool-use is formatted using XML-style tags. The tool-use is enclosed in <use_mcp_tool></use_mcp_tool> and each parameter is similarly enclosed within its own set of tags.

The Model Context Protocol (MCP) connects to servers that provide additional tools and resources to extend your capabilities. You can use the server's tools via the `use_mcp_tool`.

Description: 
Request to use a tool provided by a MCP server. Each MCP server can provide multiple tools with different capabilities. Tools have defined input schemas that specify required and optional parameters.

Parameters:
- server_name: (required) The name of the MCP server providing the tool
- tool_name: (required) The name of the tool to execute
- arguments: (required) A JSON object containing the tool's input parameters, following the tool's input schema, quotes within string must be properly escaped, ensure it's valid JSON

Usage:
<use_mcp_tool>
<server_name>server name here</server_name>
<tool_name>tool name here</tool_name>
<arguments>
{{
"param1": "value1",
"param2": "value2 \\"escaped string\\""
}}
</arguments>
</use_mcp_tool>

Important Notes:
- Tool-use must be placed **at the end** of your response, **top-level**, and not nested within other tags.
- Always adhere to this format for the tool use to ensure proper parsing and execution.

String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.
Here are the functions available in JSONSchema format:

"""

    # Add MCP servers section
    if mcp_servers and len(mcp_servers) > 0:
        for server in mcp_servers:
            template += f"\n## Server name: {server['name']}\n"

            if "tools" in server and len(server["tools"]) > 0:
                for tool in server["tools"]:
                    # Skip tools that failed to load (they only have 'error' key)
                    if "error" in tool and "name" not in tool:
                        continue
                    template += f"### Tool name: {tool['name']}\n"
                    template += f"Description: {tool['description']}\n"
                    template += f"Input JSON schema: {tool['schema']}\n"

    # Add the full objective system prompt
    template += """
# General Objective

You accomplish a given task iteratively, breaking it down into clear steps and working through them methodically.

"""

    return template


def generate_no_mcp_system_prompt(date):
    """
    Generate a minimal system prompt without MCP tool definitions.

    Used when no tools are available or when running in tool-less mode.

    Args:
        date: Current date object for timestamp inclusion

    Returns:
        Basic system prompt string without tool definitions
    """
    formatted_date = date.strftime("%Y-%m-%d")

    # Start building the template, now follows https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview#tool-use-system-prompt
    template = """In this environment you have access to a set of tools you can use to answer the user's question. """

    template += f" Today is: {formatted_date}\n"

    template += """
Important Notes:
- Tool-use must be placed **at the end** of your response, **top-level**, and not nested within other tags.
- Always adhere to this format for the tool use to ensure proper parsing and execution.

String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.
"""

    # Add the full objective system prompt
    template += """
# General Objective

You accomplish a given task iteratively, breaking it down into clear steps and working through them methodically.

"""
    return template


def generate_agent_specific_system_prompt(agent_type=""):
    """
    Generate agent-specific objective prompts based on agent type.

    Different agent types have different objectives:
    - main: Task-solving agent that uses tools to answer questions
    - agent-browsing: Web search and browsing agent for information retrieval

    Args:
        agent_type: Type of agent ("main", "agent-browsing", or "browsing-agent")

    Returns:
        Agent-specific objective prompt string
    """
    if agent_type == "main":
        system_prompt = """\n
# Agent Specific Objective

You are a task-solving agent that uses tools step-by-step to answer the user's question. Your goal is to provide complete, accurate and well-reasoned answers using additional tools.

"""
    elif agent_type == "agent-browsing" or agent_type == "browsing-agent":
        system_prompt = """# Agent Specific Objective

You are an agent that performs the task of searching and browsing the web for specific information and generating the desired answer. Your task is to retrieve reliable, factual, and verifiable information that fills in knowledge gaps.
Do not infer, speculate, summarize broadly, or attempt to fill in missing parts yourself. Only return factual content.
"""
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return system_prompt.strip()


def render_objective_prompt(cfg_prompt, date, agent_type="main"):
    """
    Render the "objective" segment of the system prompt based on Hydra config.

    The full system prompt has two parts:
        1) header    -- injected by llm_client.generate_agent_system_prompt(),
                        contains tool schemas / date / format instructions.
                        NOT controlled by this function.
        2) objective -- task-behavior / role instructions. Selected here.

    This function lets us swap the objective via `prompt=<preset>` on the CLI
    (Hydra config group `conf/prompt/`) without touching code:

        - style: "default"  -> equivalent to the hardcoded
                               generate_agent_specific_system_prompt(agent_type).
        - style: "custom"   -> use cfg_prompt.template with {date} interpolation.

    To limit the custom prompt to the main agent (and keep sub-agents on the
    default objective so retrieval / browsing behavior is unchanged), set
    `apply_to: "main"` in the yaml. Set `apply_to: "all"` to override sub-agents
    too.

    Args:
        cfg_prompt: OmegaConf node (or dict-like) from cfg.prompt. May be None,
            in which case the function falls back to the default behavior.
        date: Current date object (must support .strftime).
        agent_type: "main" / "agent-browsing" / etc.

    Returns:
        Objective prompt string (already stripped).
    """
    # Defensive fallback: if cfg.prompt was never injected (e.g. an old
    # config.yaml without `- prompt: default` in its defaults list), behave
    # exactly like the legacy code path.
    if cfg_prompt is None:
        return generate_agent_specific_system_prompt(agent_type=agent_type)

    # OmegaConf nodes support .get(); plain dicts do too.
    style = cfg_prompt.get("style", "default") if hasattr(cfg_prompt, "get") else "default"

    if style == "default":
        return generate_agent_specific_system_prompt(agent_type=agent_type)

    if style == "custom":
        apply_to = cfg_prompt.get("apply_to", "main")
        # Sub-agent is excluded unless apply_to == "all".
        if apply_to != "all" and agent_type != "main":
            return generate_agent_specific_system_prompt(agent_type=agent_type)

        template = cfg_prompt.get("template", "")
        if not template:
            # Empty template => fall back to default rather than emit a blank section.
            return generate_agent_specific_system_prompt(agent_type=agent_type)

        formatted_date = (
            date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
        )
        # Only {date} is a recognized placeholder. Using `.replace()` instead of
        # `.format()` so that arbitrary braces in user-supplied templates (e.g.
        # JSON examples, regex) don't blow up with KeyError.
        return template.replace("{date}", formatted_date).strip()

    raise ValueError(
        f"Unknown prompt style: {style!r}. Expected 'default' or 'custom'."
    )


# ----------------------------------------------------------------------------
# Header surgery: strip the default "# General Objective" block from the
# provider-specific header so a custom objective can fully replace it.
# ----------------------------------------------------------------------------

# All four provider header templates (OpenAI native-fc, Anthropic native-tools,
# Gemini, and the MCP-XML generate_mcp_system_prompt) end with the same
# hard-coded section:
#
#     # General Objective
#
#     You accomplish a given task iteratively, breaking it down into clear
#     steps and working through them methodically.
#
# When the user picks a `style: "custom"` prompt (e.g. Kimi), keeping that
# section in the header would produce a confusing "double objective" system
# prompt -- one default and one custom. This helper removes it cleanly.
_DEFAULT_HEADER_OBJECTIVE_MARKER = "# General Objective"


def _strip_default_objective_section(header: str) -> str:
    """Remove the trailing default '# General Objective' block (if present)
    from the given header string. Idempotent: returns the header unchanged
    if no such marker is found. Always leaves a single trailing blank line
    so concatenating an objective afterwards stays readable.
    """
    idx = header.rfind(_DEFAULT_HEADER_OBJECTIVE_MARKER)
    if idx < 0:
        # Nothing to strip; just normalize trailing whitespace.
        return header.rstrip() + "\n\n"
    return header[:idx].rstrip() + "\n\n"


# Marker that delimits "intro paragraph" from "tool schemas" in the
# mcp_xml-mode header. If found, everything BEFORE this marker is the
# generic 'In this environment ... Today is: DATE' preamble and is safe
# to drop when the user supplies their own self-contained objective.
# If NOT found (native_fc mode), the entire header is intro-only and can
# be dropped wholesale -- tools are passed via the API `tools` parameter
# instead of being described in the system prompt.
_TOOL_SCHEMA_SECTION_MARKER = "# Tool-Use Formatting Instructions"


def _strip_header_intro(header: str) -> str:
    """Remove the boilerplate intro paragraphs at the top of the header
    ('In this environment you have access ... Today is: DATE'), while
    PRESERVING downstream tool schemas / format instructions (mcp_xml mode).

    - mcp_xml mode  -> returns the portion starting from
                       '# Tool-Use Formatting Instructions' onward.
    - native_fc mode -> returns '' (header was intro-only).

    Use this only when the custom objective is fully self-contained
    (e.g. Kimi's prompt already includes its own date and role) -- otherwise
    you'd lose the model's environmental cues.
    """
    idx = header.find(_TOOL_SCHEMA_SECTION_MARKER)
    if idx >= 0:
        return header[idx:].rstrip() + "\n\n"
    return ""


def compose_full_system_prompt(
    cfg_prompt,
    llm_client,
    date,
    mcp_servers,
    agent_type,
):
    """Build the complete system prompt (header + objective) honoring
    Hydra `cfg.prompt`.

    The provider header naturally splits into three layered sections:

        [A] Intro paragraph:
            "In this environment you have access ... Today is: DATE"
        [B] Tool schemas / formatting instructions
            (mcp_xml mode only; native_fc passes tools via API param instead)
        [C] Default "# General Objective\\n\\nYou accomplish ..." block

    Behaviors:

    - `style: "default"` (or cfg_prompt is None)
        Returns exactly the legacy concatenation:
            [A] + [B?] + [C] + default_agent_specific_objective
        Fully backward compatible.

    - `style: "custom"` and this agent_type is to be replaced
        Always strips [C] (otherwise you'd get a confusing "double
        objective"). Then if `strip_header_intro: true` is also set in the
        yaml, additionally strips [A] -- useful when the custom template is
        self-contained (e.g. Kimi already has its own "today's date: DATE"
        line, so keeping [A] would just be a redundant duplicate). [B] is
        ALWAYS preserved if present, because mcp_xml mode needs it for the
        tool-call mechanism to work.

    Args:
        cfg_prompt: OmegaConf node (or None) from cfg.prompt.
        llm_client: an LLM client exposing generate_agent_system_prompt().
        date: today's date object.
        mcp_servers: tool definitions to pass to the client.
        agent_type: "main" / "agent-browsing" / etc.

    Returns:
        Fully-rendered system prompt string (stripped).
    """
    header = llm_client.generate_agent_system_prompt(
        date=date, mcp_servers=mcp_servers,
    )

    # Detect whether a custom objective will fully replace this agent's
    # default. Only in that case do we surgically rewrite the header.
    custom_owns_objective = False
    strip_intro = False
    if cfg_prompt is not None and hasattr(cfg_prompt, "get"):
        style = cfg_prompt.get("style", "default") or "default"
        if style == "custom":
            apply_to = cfg_prompt.get("apply_to", "main") or "main"
            template = cfg_prompt.get("template", "") or ""
            if template and (apply_to == "all" or agent_type == "main"):
                custom_owns_objective = True
                strip_intro = bool(cfg_prompt.get("strip_header_intro", False))

    if custom_owns_objective:
        # Order matters: strip [C] first to avoid the "double-objective"
        # problem, then optionally strip [A] to remove the redundant
        # 'Today is: DATE' preamble. Tool schemas [B] are preserved
        # by both strippers (when present).
        header = _strip_default_objective_section(header)
        if strip_intro:
            header = _strip_header_intro(header)

    objective = render_objective_prompt(cfg_prompt, date, agent_type=agent_type)
    return (header + objective).strip()


def generate_agent_summarize_prompt(task_description, agent_type="", benchmark_name=""):
    """
    Generate the final summarization prompt for an agent.

    Creates prompts that instruct agents to summarize their work and provide
    final answers. Different agent types have different summarization formats:
    - main: Must wrap answer in \\boxed{} with strict formatting rules
    - agent-browsing: Provides structured report of findings

    Supports benchmark-specific prompt variants via ``benchmark_name``.
    When ``benchmark_name`` starts with ``"deepsearchqa"``, additional
    multi-item verification rules are injected to reduce excessive answers.

    Args:
        task_description: The original task/question to reference in the summary
        agent_type: Type of agent ("main" or "agent-browsing")
        benchmark_name: Optional benchmark identifier (e.g. "deepsearchqa").
            Defaults to "" which uses the generic prompt for all benchmarks.

    Returns:
        Summarization prompt string with formatting instructions
    """
    if agent_type == "main":
        # Benchmark-specific rules can be injected here via benchmark_name.
        # Currently all benchmarks use the same default prompt.
        # To add benchmark-specific rules, check benchmark_name and build
        # a rules string, then insert it before the \boxed{} instruction.

        summarize_prompt = (
            "Summarize the above conversation, and output the FINAL ANSWER to the original question.\n\n"
            "If a clear answer has already been provided earlier in the conversation, do not rethink or recalculate it — "
            "simply extract that answer and reformat it to match the required format below.\n"
            "If a definitive answer could not be determined, make a well-informed educated guess based on the conversation.\n\n"
            "The original question is repeated here for reference:\n\n"
            f'"{task_description}"\n\n'
            "Wrap your final answer in \\boxed{}.\n"
            "Your final answer should be:\n"
            "- a number, OR\n"
            "- as few words as possible, OR\n"
            "- a comma-separated list of numbers and/or strings.\n\n"
            "ADDITIONALLY, your final answer MUST strictly follow any formatting instructions in the original question — "
            "such as alphabetization, sequencing, units, rounding, decimal places, etc.\n"
            "If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and DO NOT INCLUDE UNITS such as $ or USD or percent signs unless specified otherwise.\n"
            "If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.\n"
            "If you are asked for a comma-separated list, apply the above rules depending on whether the elements are numbers or strings.\n"
            "Do NOT include any punctuation such as '.', '!', or '?' at the end of the answer.\n"
            "Do NOT include any invisible or non-printable characters in the answer output.\n\n"
            "You must absolutely not perform any MCP tool call, tool invocation, search, scrape, code execution, or similar actions.\n"
            "You can only answer the original question based on the information already retrieved and your own internal knowledge.\n"
            "If you attempt to call any tool, it will be considered a mistake."
        )
    elif agent_type == "agent-browsing":
        summarize_prompt = (
            "This is a direct instruction to you (the assistant), not the result of a tool call.\n\n"
            "We are now ending this session, and your conversation history will be deleted. "
            "You must NOT initiate any further tool use. This is your final opportunity to report "
            "*all* of the information gathered during the session.\n\n"
            "The original task is repeated here for reference:\n\n"
            f'"{task_description}"\n\n'
            "Summarize the above search and browsing history. Output the FINAL RESPONSE and detailed supporting information of the task given to you.\n\n"
            "If you found any useful facts, data, quotes, or answers directly relevant to the original task, include them clearly and completely.\n"
            "If you reached a conclusion or answer, include it as part of the response.\n"
            "If the task could not be fully answered, do NOT make up any content. Instead, return all partially relevant findings, "
            "Search results, quotes, and observations that might help a downstream agent solve the problem.\n"
            "If partial, conflicting, or inconclusive information was found, clearly indicate this in your response.\n\n"
            "Your final response should be a clear, complete, and structured report.\n"
            "Organize the content into logical sections with appropriate headings.\n"
            "Do NOT include any tool call instructions, speculative filler, or vague summaries.\n"
            "Focus on factual, specific, and well-organized information."
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    return summarize_prompt.strip()


def generate_direct_summarize_prompt(task_description: str) -> str:
    """
    Generate a lightweight summarization prompt for direct answer mode.

    Used only when the agent terminates abnormally (max turns, context overflow,
    consecutive rollbacks) in ``answer_mode="direct"``.  Does NOT require the
    model to use ``\\boxed{}``; instead asks for a natural-language answer.

    Args:
        task_description: The original task/question to reference.

    Returns:
        Summarization prompt string (no special formatting requirements).
    """
    return (
        "The research session has ended. Based on all the information gathered above, "
        "please provide your final answer to the original question.\n\n"
        "The original question is repeated here for reference:\n\n"
        f'"{task_description}"\n\n'
        "Give your best answer directly in natural language. "
        "If a clear answer has already been found earlier in the conversation, simply restate it. "
        "If a definitive answer could not be determined, make a well-informed educated guess "
        "based on the information you have gathered.\n\n"
        "You must absolutely not perform any tool call, search, scrape, or code execution. "
        "Answer based only on the information already retrieved and your own knowledge."
    ).strip()
