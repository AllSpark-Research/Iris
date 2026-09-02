# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

import asyncio

import hydra
from omegaconf import DictConfig, OmegaConf

# Import from the new modular structure
from src.core.pipeline import (
    create_pipeline_components,
    execute_task_pipeline,
)
from src.logging.task_logger import bootstrap_logger

# Configure logger and get the configured instance
logger = bootstrap_logger()


async def amain(cfg: DictConfig) -> None:
    """Asynchronous main function."""

    logger.info(OmegaConf.to_yaml(cfg))

    # Create pipeline components using the factory function
    main_agent_tool_manager, sub_agent_tool_managers, output_formatter = (
        create_pipeline_components(cfg)
    )

    # Define task parameters
    task_id = "task_example"
    task_description = "What is the title of today's arxiv paper in computer science?"
    task_file_name = ""

    # Execute task using the pipeline
    final_summary, final_boxed_answer, log_file_path, _ = await execute_task_pipeline(
        cfg=cfg,
        task_id=task_id,
        task_file_name=task_file_name,
        task_description=task_description,
        main_agent_tool_manager=main_agent_tool_manager,
        sub_agent_tool_managers=sub_agent_tool_managers,
        output_formatter=output_formatter,
        log_dir=cfg.debug_dir,
    )

    print("\n" + "=" * 60)
    print(f"Question: {task_description}")
    print(f"Answer:   {final_boxed_answer or '(no answer extracted)'}")
    print(f"Trace:    {log_file_path}")
    print("=" * 60)
    if final_summary:
        print(f"\n{final_summary}")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    asyncio.run(amain(cfg))


if __name__ == "__main__":
    main()
