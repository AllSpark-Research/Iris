# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
LLM Client Factory module.

Creates the LLM client for the policy model from the Hydra config. The harness
drives OpenAI-compatible endpoints only (SGLang / vLLM / OpenAI / Azure), which
is how both Iris models and every open-weight baseline are served.
"""

from typing import Optional

from omegaconf import DictConfig, OmegaConf

from ..logging.task_logger import TaskLog
from .providers.openai_client import OpenAIClient

# Supported LLM providers
SUPPORTED_PROVIDERS = {"openai", "qwen", "azure"}


def ClientFactory(
    task_id: str, cfg: DictConfig, task_log: Optional[TaskLog] = None, **kwargs
) -> OpenAIClient:
    """
    Create an LLM client based on the provider specified in configuration.

    Supported providers:
        - "openai": OpenAI / any OpenAI-compatible server (SGLang, vLLM, ...)
        - "qwen":   Qwen models over an OpenAI-compatible API
        - "azure":  Azure OpenAI (custom header auth)

    Args:
        task_id: Unique identifier for the current task (used for tracking)
        cfg: Hydra configuration object containing LLM settings
        task_log: Optional logger for recording task execution details
        **kwargs: Additional keyword arguments to merge into configuration

    Returns:
        An OpenAIClient bound to the configured endpoint.

    Example:
        >>> client = ClientFactory(task_id="task_001", cfg=cfg, task_log=task_log)
    """
    provider = cfg.llm.provider
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported provider: '{provider}'. "
            f"Supported providers are: {', '.join(sorted(SUPPORTED_PROVIDERS))}"
        )
    return OpenAIClient(task_id=task_id, task_log=task_log, cfg=OmegaConf.merge(cfg, kwargs))
