# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Configuration settings and MCP server parameter management.

This module handles:
- Loading environment variables for API keys and service URLs
- Creating MCP server configurations for different tools
- Exposing sub-agents as callable tools
- Collecting environment information for logging
"""

import os
import sys

from dotenv import load_dotenv
from mcp import StdioServerParameters
from omegaconf import DictConfig

# Load environment variables from .env file
load_dotenv()

# Serper — Google SERP + page scrape. Powers ``tool-serper-search``, which
# exposes the unified ``web_search`` + ``scrape_website`` pair.
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
SERPER_BASE_URL = os.environ.get("SERPER_BASE_URL", "https://google.serper.dev")
SERPER_SCRAPE_URL = os.environ.get("SERPER_SCRAPE_URL", "https://scrape.serper.dev")
SERPER_SEARCH_LIMIT = os.environ.get("SERPER_SEARCH_LIMIT", "10")
SERPER_DEFAULT_GL = os.environ.get("SERPER_DEFAULT_GL", "us")
SERPER_DEFAULT_HL = os.environ.get("SERPER_DEFAULT_HL", "en")

# Jina Reader — scrape only. Powers ``tool-jina-scrape``; pair it with another
# backend's web_search via the agent's tool_blacklist (see
# conf/agent/serper_jina_search_agent.yaml).
JINA_API_KEY = os.environ.get("JINA_API_KEY")
JINA_BASE_URL = os.environ.get("JINA_BASE_URL", "https://r.jina.ai")
# Optional HTTP(S) proxy for Jina traffic ONLY (``*.jina.ai`` is DNS-blocked in
# some internal networks). Scoped to the ``tool-jina-scrape`` subprocess, so the
# summary LLM and every other tool/server stay on a direct connection.
JINA_PROXY = os.environ.get("JINA_PROXY", "")
# X-Engine: "direct" beats "browser" on academic pages by 200x (browser returns
# an SVG figure caption instead of the paper) and is ~4x faster; "browser" is
# kept as the escalation engine for genuine JS SPAs. See the module docstring of
# searching_jina_mcp_server for the measurements.
JINA_ENGINE = os.environ.get("JINA_ENGINE", "direct")
JINA_FALLBACK_ENGINE = os.environ.get("JINA_FALLBACK_ENGINE", "browser")
JINA_THIN_CONTENT_CHARS = os.environ.get("JINA_THIN_CONTENT_CHARS", "1200")
JINA_MIN_USABLE_CHARS = os.environ.get("JINA_MIN_USABLE_CHARS", "200")
JINA_RETAIN_IMAGES = os.environ.get("JINA_RETAIN_IMAGES", "none")
JINA_WITH_IMAGES_SUMMARY = os.environ.get("JINA_WITH_IMAGES_SUMMARY", "false")
JINA_NO_CACHE = os.environ.get("JINA_NO_CACHE", "false")
JINA_CONNECT_TIMEOUT = os.environ.get("JINA_CONNECT_TIMEOUT", "20")
JINA_READ_TIMEOUT = os.environ.get("JINA_READ_TIMEOUT", "90")
JINA_PYTHON_FALLBACK = os.environ.get("JINA_PYTHON_FALLBACK", "true")

# Zhipu AI — search + reader. Powers ``tool-zhipu-search``.
ZHIPU_AI_KEY = os.environ.get("ZHIPU_AI_KEY")
ZHIPU_BASE_URL = os.environ.get("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")

# API keys for the LLM-as-judge fallback (see benchmarks/evaluators/eval_utils.py:
# JUDGE_API_KEY / JUDGE_BASE_URL take precedence; these are the fallback).
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

# API for Summary LLM
SUMMARY_LLM_API_KEY = os.environ.get("SUMMARY_LLM_API_KEY")
SUMMARY_LLM_BASE_URL = os.environ.get("SUMMARY_LLM_BASE_URL")
SUMMARY_LLM_MODEL_NAME = os.environ.get("SUMMARY_LLM_MODEL_NAME")
SUMMARY_LLM_PROVIDER = os.environ.get("SUMMARY_LLM_PROVIDER", "")
SUMMARY_LLM_API_VERSION = os.environ.get("SUMMARY_LLM_API_VERSION", "")


# MCP server configuration generation function
def create_mcp_server_parameters(cfg: DictConfig, agent_cfg: DictConfig):
    """
    Create MCP server configurations based on agent configuration.

    Dynamically generates StdioServerParameters for each tool specified in the
    agent configuration. Each backend (Serper / Jina / Zhipu) has its own MCP
    server with appropriate environment variables.

    Args:
        cfg: Global Hydra configuration object
        agent_cfg: Agent-specific configuration containing 'tools' and 'tool_blacklist'

    Returns:
        Tuple of (configs, blacklist) where:
        - configs: List of dicts with 'name' and 'params' (StdioServerParameters)
        - blacklist: Set of (server_name, tool_name) tuples to exclude
    """
    configs = []

    if (
        agent_cfg.get("tools", None) is not None
        and "tool-serper-search" in agent_cfg["tools"]
    ):
        if not SERPER_API_KEY:
            raise ValueError(
                "SERPER_API_KEY not set, tool-serper-search will be unavailable."
            )

        configs.append(
            {
                "name": "tool-serper-search",
                "params": StdioServerParameters(
                    command=sys.executable,
                    args=[
                        "-m",
                        "miroflow_tools.mcp_servers.searching_serper_mcp_server",
                    ],
                    env={
                        "SERPER_API_KEY": SERPER_API_KEY,
                        "SERPER_BASE_URL": SERPER_BASE_URL,
                        "SERPER_SCRAPE_URL": SERPER_SCRAPE_URL,
                        "SERPER_SEARCH_LIMIT": SERPER_SEARCH_LIMIT,
                        "SERPER_DEFAULT_GL": SERPER_DEFAULT_GL,
                        "SERPER_DEFAULT_HL": SERPER_DEFAULT_HL,
                        # Summary LLM for scrape_website extraction
                        "SUMMARY_LLM_BASE_URL": SUMMARY_LLM_BASE_URL or "",
                        "SUMMARY_LLM_MODEL_NAME": SUMMARY_LLM_MODEL_NAME or "",
                        "SUMMARY_LLM_API_KEY": SUMMARY_LLM_API_KEY or "",
                        "SUMMARY_LLM_PROVIDER": SUMMARY_LLM_PROVIDER or "",
                        "SUMMARY_LLM_API_VERSION": SUMMARY_LLM_API_VERSION or "",
                    },
                ),
            }
        )

    if (
        agent_cfg.get("tools", None) is not None
        and "tool-jina-scrape" in agent_cfg["tools"]
    ):
        if not JINA_API_KEY:
            raise ValueError(
                "JINA_API_KEY not set, tool-jina-scrape will be unavailable."
            )

        # Scrape-only server: exposes a single ``scrape_website`` whose name,
        # signature and JSON contract match ``tool-serper-search``'s, so it is a
        # drop-in replacement once serper's scrape is blacklisted.
        #
        # NOTE on the proxy: StdioServerParameters.env is EXCLUSIVE — the child
        # process sees ONLY the keys listed here. Passing JINA_PROXY (and no
        # HTTP_PROXY/HTTPS_PROXY) therefore confines proxying to this one
        # subprocess; the server itself applies it only to its web client and
        # runs both clients with trust_env=False. Concurrent evals in other
        # processes are unaffected.
        _jina_env = {
            "JINA_API_KEY": JINA_API_KEY,
            "JINA_BASE_URL": JINA_BASE_URL,
            "JINA_PROXY": JINA_PROXY or "",
            "JINA_ENGINE": JINA_ENGINE,
            "JINA_FALLBACK_ENGINE": JINA_FALLBACK_ENGINE,
            "JINA_THIN_CONTENT_CHARS": JINA_THIN_CONTENT_CHARS,
            "JINA_MIN_USABLE_CHARS": JINA_MIN_USABLE_CHARS,
            "JINA_RETAIN_IMAGES": JINA_RETAIN_IMAGES,
            "JINA_WITH_IMAGES_SUMMARY": JINA_WITH_IMAGES_SUMMARY,
            "JINA_NO_CACHE": JINA_NO_CACHE,
            "JINA_CONNECT_TIMEOUT": JINA_CONNECT_TIMEOUT,
            "JINA_READ_TIMEOUT": JINA_READ_TIMEOUT,
            "JINA_PYTHON_FALLBACK": JINA_PYTHON_FALLBACK,
            # Summary LLM for scrape_website extraction (always forced)
            "SUMMARY_LLM_BASE_URL": SUMMARY_LLM_BASE_URL or "",
            "SUMMARY_LLM_MODEL_NAME": SUMMARY_LLM_MODEL_NAME or "",
            "SUMMARY_LLM_API_KEY": SUMMARY_LLM_API_KEY or "",
            "SUMMARY_LLM_PROVIDER": SUMMARY_LLM_PROVIDER or "",
            "SUMMARY_LLM_API_VERSION": SUMMARY_LLM_API_VERSION or "",
        }
        # Only forward when explicitly set; the server's own default otherwise.
        if os.environ.get("MAX_SCRAPE_CHARS"):
            _jina_env["MAX_SCRAPE_CHARS"] = os.environ["MAX_SCRAPE_CHARS"]

        configs.append(
            {
                "name": "tool-jina-scrape",
                "params": StdioServerParameters(
                    command=sys.executable,
                    args=[
                        "-m",
                        "miroflow_tools.mcp_servers.searching_jina_mcp_server",
                    ],
                    env=_jina_env,
                ),
            }
        )

    if (
        agent_cfg.get("tools", None) is not None
        and "tool-zhipu-search" in agent_cfg["tools"]
    ):
        # searching_zhipu_mcp_server: async httpx, Zhipu-reader -> Python httpx
        # fallback, 409600-char cap, repeat-output detection, forced summary.
        if not ZHIPU_AI_KEY:
            raise ValueError(
                "ZHIPU_AI_KEY not set, tool-zhipu-search will be unavailable."
            )

        configs.append(
            {
                "name": "tool-zhipu-search",
                "params": StdioServerParameters(
                    command=sys.executable,
                    args=[
                        "-m",
                        "miroflow_tools.mcp_servers.searching_zhipu_mcp_server",
                    ],
                    env={
                        "ZHIPU_AI_KEY": ZHIPU_AI_KEY,
                        "ZHIPU_BASE_URL": ZHIPU_BASE_URL,
                        # Max chars kept from a scraped page before summary
                        "MAX_SCRAPE_CHARS": os.environ.get(
                            "MAX_SCRAPE_CHARS", str(102400 * 4)
                        ),
                        # Summary LLM for scrape_website extraction
                        "SUMMARY_LLM_BASE_URL": SUMMARY_LLM_BASE_URL or "",
                        "SUMMARY_LLM_MODEL_NAME": SUMMARY_LLM_MODEL_NAME or "",
                        "SUMMARY_LLM_API_KEY": SUMMARY_LLM_API_KEY or "",
                        "SUMMARY_LLM_PROVIDER": SUMMARY_LLM_PROVIDER or "",
                        "SUMMARY_LLM_API_VERSION": SUMMARY_LLM_API_VERSION or "",
                    },
                ),
            }
        )

    blacklist = set()
    for black_list_item in agent_cfg.get("tool_blacklist", []):
        blacklist.add((black_list_item[0], black_list_item[1]))
    return configs, blacklist


def expose_sub_agents_as_tools(sub_agents_cfg: DictConfig):
    """
    Convert sub-agent configurations into tool definitions for the main agent.

    This allows the main agent to invoke sub-agents (like the browsing agent)
    as if they were regular MCP tools, enabling a hierarchical agent architecture.

    Args:
        sub_agents_cfg: Configuration containing sub-agent definitions

    Returns:
        List of server parameter dicts, each with 'name' and 'tools' keys.
        Each tool includes 'name', 'description', and 'schema' for the sub-agent.

    Note: every agent config shipped here declares an empty ``sub_agents``, so
    this returns an empty list. The plumbing is kept for anyone adding a
    hierarchical agent on top of the harness.
    """
    sub_agents_server_params = []
    for sub_agent in sub_agents_cfg.keys():
        if "agent-browsing" in sub_agent:
            sub_agents_server_params.append(
                dict(
                    name="agent-browsing",
                    tools=[
                        dict(
                            name="search_and_browse",
                            description="This tool is an agent that performs the subtask of searching and browsing the web for specific missing information and generating the desired answer. The subtask should be clearly defined, include relevant background, and focus on factual gaps. It does not perform vague or speculative subtasks. \nArgs: \n\tsubtask: the subtask to be performed. \nReturns: \n\tthe result of the subtask. ",
                            schema={
                                "type": "object",
                                "properties": {
                                    "subtask": {"title": "Subtask", "type": "string"}
                                },
                                "required": ["subtask"],
                                "title": "search_and_browseArguments",
                            },
                        )
                    ],
                )
            )
    return sub_agents_server_params


def get_env_info(cfg: DictConfig) -> dict:
    """
    Collect current configuration and environment information for logging.

    Gathers LLM settings, agent configuration, API key availability (masked),
    and base URLs. Used for debugging and task log enrichment.

    Args:
        cfg: Hydra configuration object

    Returns:
        Dictionary containing:
        - LLM configuration (provider, model, temperature, etc.)
        - Agent configuration (max turns for main/sub agents)
        - API key availability flags (boolean, not actual keys)
        - Service base URLs
    """
    return {
        # LLM Configuration
        "llm_provider": cfg.llm.provider,
        "llm_base_url": cfg.llm.base_url,
        "llm_model_name": cfg.llm.model_name,
        "llm_temperature": cfg.llm.temperature,
        "llm_top_p": cfg.llm.top_p,
        "llm_min_p": cfg.llm.min_p,
        "llm_top_k": cfg.llm.top_k,
        "llm_max_tokens": cfg.llm.max_tokens,
        "llm_repetition_penalty": cfg.llm.repetition_penalty,
        "llm_async_client": cfg.llm.async_client,
        "keep_tool_result": cfg.agent.keep_tool_result,
        # Agent Configuration
        "main_agent_max_turns": cfg.agent.main_agent.max_turns,
        **(
            {
                f"sub_{sub_agent}_max_turns": cfg.agent.sub_agents[sub_agent].max_turns
                for sub_agent in cfg.agent.sub_agents
            }
            if cfg.agent.sub_agents is not None
            else {}
        ),
        # API Keys (masked for security)
        "has_serper_api_key": bool(SERPER_API_KEY),
        "has_jina_api_key": bool(JINA_API_KEY),
        "has_zhipu_ai_key": bool(ZHIPU_AI_KEY),
        "has_openai_api_key": bool(OPENAI_API_KEY),
        "has_summary_llm_api_key": bool(SUMMARY_LLM_API_KEY),
        # Base URLs
        "openai_base_url": OPENAI_BASE_URL,
        "jina_base_url": JINA_BASE_URL,
        "jina_proxy_set": bool(JINA_PROXY),
        "serper_base_url": SERPER_BASE_URL,
        "summary_llm_base_url": SUMMARY_LLM_BASE_URL,
    }
