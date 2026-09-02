# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Task input preparation.

Turns a raw benchmark record into the user message the agent starts from. The
four benchmarks this harness reports on -- BrowseComp, BrowseComp-ZH,
DeepSearchQA and HLE (text-only) -- are pure text: none of their 4,613 tasks
carries an attachment, so this module deliberately has no document, image or
audio converters. A task that does name a file is refused loudly rather than
silently evaluated with the attachment missing.
"""

from typing import Tuple

# Appended in boxed mode so the final answer can be extracted deterministically.
# Keep this string stable: changing it changes the prompt, and therefore the
# scores.
BOXED_INSTRUCTION = (
    "\nYou should follow the format instruction in the request strictly "
    "and wrap the final answer in \\boxed{}."
)


def process_input(
    task_description: str, task_file_name: str, answer_mode: str = "boxed"
) -> Tuple[str, str]:
    """
    Build the task description handed to the agent.

    Args:
        task_description: The question as it appears in the benchmark file.
        task_file_name: Attachment path. Must be empty -- this harness is
            text-only.
        answer_mode: "boxed" appends the ``\\boxed{}`` format instruction;
            "direct" leaves the question untouched and the answer is taken from
            the model's final message.

    Returns:
        ``(task_description, task_description)`` -- the same string twice; the
        pair is kept for call-site compatibility.

    Raises:
        ValueError: if the task carries an attachment.
    """
    if task_file_name and task_file_name.strip():
        raise ValueError(
            f"Task references the attachment {task_file_name!r}, but this harness "
            "evaluates text-only benchmarks and has no file converters. Use a "
            "multimodal harness for tasks with attachments."
        )

    updated_task_description = task_description
    if answer_mode == "boxed":
        updated_task_description += BOXED_INSTRUCTION

    updated_task_description = updated_task_description.strip()
    return updated_task_description, updated_task_description
