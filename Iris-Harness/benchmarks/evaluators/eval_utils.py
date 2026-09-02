# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

import asyncio
import json
import logging
import os
import re
import string
import warnings
from typing import Any, Dict, Literal, Optional

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger("miroflow_agent")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")

# Judge model endpoint configuration.
# If JUDGE_API_KEY / JUDGE_BASE_URL are set, the judge uses a separate service;
# otherwise it falls back to OPENAI_API_KEY / OPENAI_BASE_URL.
# Set JUDGE_API_VERSION to enable Azure-style proxy (api-key header + api-version query).
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", OPENAI_API_KEY)
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", OPENAI_BASE_URL)
JUDGE_API_VERSION = os.environ.get("JUDGE_API_VERSION")
# No default: a silently-wrong judge produces a plausible-looking score, which
# is worse than refusing to start. Gateways that route by URL path legitimately
# take an empty model name -- that case is covered by JUDGE_PROVIDER=maas below,
# and _derive_judge_model_label() resolves a readable label for the banner.
JUDGE_MODEL_NAME = os.environ.get("JUDGE_MODEL_NAME", "")

# Judge provider: "gemini" | "azure" | "maas" | "openai"
# Auto-detection (when JUDGE_PROVIDER is not set):
#   1) "gemini" in model name           → gemini
#   2) JUDGE_API_VERSION is set         → azure
#   3) otherwise                        → openai
#   4) fallback                         → openai
_judge_provider_env = os.environ.get("JUDGE_PROVIDER", "").lower()
if _judge_provider_env:
    JUDGE_PROVIDER = _judge_provider_env
elif "gemini" in JUDGE_MODEL_NAME.lower():
    JUDGE_PROVIDER = "gemini"
elif JUDGE_API_VERSION:
    JUDGE_PROVIDER = "azure"
else:
    JUDGE_PROVIDER = "openai"






def _derive_judge_model_label() -> str:
    """Best-effort human-readable label for the judge model.

    Use this for logging/trace/banner display instead of ``JUDGE_MODEL_NAME``.
    Some model-as-a-service gateways route by URL path slug and accept
    ``model=""`` in the request body, so ``JUDGE_MODEL_NAME`` can legitimately be
    empty and every trace would then report an empty ``judge_model``. In that
    case we fall back to the first non-version path segment of the base URL
    (e.g. ``.../my-judge-deployment/v1`` -> ``maas:my-judge-deployment``).
    """
    if JUDGE_MODEL_NAME:
        return JUDGE_MODEL_NAME
    if JUDGE_PROVIDER == "maas" and JUDGE_BASE_URL:
        try:
            from urllib.parse import urlparse

            parts = [p for p in urlparse(JUDGE_BASE_URL).path.split("/") if p]
            # Filter out trailing "v1" / "v2" version segments
            slug = next((p for p in parts if not p.lower().startswith("v")), None) or (
                parts[0] if parts else None
            )
            if slug:
                return f"maas:{slug}"
        except Exception:
            pass
        return "maas"
    if JUDGE_PROVIDER == "azure":
        return "azure"
    if JUDGE_PROVIDER == "gemini":
        return "gemini"
    return "unknown"


JUDGE_MODEL_LABEL = _derive_judge_model_label()


def _create_judge_client(*, async_mode: bool):
    """Create a judge LLM client.

    Only used when JUDGE_PROVIDER is "azure", "maas", or "openai".
    For "gemini" provider, we use raw httpx requests instead.

    Auth/header conventions:
      - azure : api-key header + api-version query param (via http_client params)
      - maas  : api-key header (no api-version)
      - openai: standard Bearer ``Authorization`` header (OpenAI SDK default)
    """
    if JUDGE_PROVIDER == "gemini":
        # Gemini uses raw httpx; return None as placeholder — callers use
        # _judge_chat_completion() instead of the OpenAI client directly.
        return None

    if JUDGE_PROVIDER == "azure" or JUDGE_API_VERSION:
        # Azure-style proxy: authenticate via "api-key" header, pass api-version as query param.
        headers = {"api-key": JUDGE_API_KEY}
        if async_mode:
            http_client = httpx.AsyncClient(params={"api-version": JUDGE_API_VERSION})
            return AsyncOpenAI(
                api_key="azure-api-key-in-header",
                base_url=JUDGE_BASE_URL,
                default_headers=headers,
                http_client=http_client,
            )
        else:
            http_client = httpx.Client(params={"api-version": JUDGE_API_VERSION})
            return OpenAI(
                api_key="azure-api-key-in-header",
                base_url=JUDGE_BASE_URL,
                default_headers=headers,
                http_client=http_client,
            )
    elif JUDGE_PROVIDER == "maas":
        # MaaS-style gateway: OpenAI-compatible, but auth is an "api-key" header
        # (NOT Bearer). BASE_URL should point to the /v1 segment; the OpenAI
        # SDK will append /chat/completions automatically.
        headers = {"api-key": JUDGE_API_KEY}
        if async_mode:
            return AsyncOpenAI(
                api_key="maas-api-key-in-header",
                base_url=JUDGE_BASE_URL,
                default_headers=headers,
            )
        else:
            return OpenAI(
                api_key="maas-api-key-in-header",
                base_url=JUDGE_BASE_URL,
                default_headers=headers,
            )
    else:
        # Standard OpenAI-compatible endpoint (Bearer auth via SDK default).
        if async_mode:
            return AsyncOpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)
        else:
            return OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)


def _judge_config_error() -> str:
    """Explain a missing/unusable judge configuration in terms of the env vars."""
    return (
        "No usable judge configuration. Set JUDGE_BASE_URL + JUDGE_MODEL_NAME + "
        "JUDGE_API_KEY (or OPENAI_BASE_URL / OPENAI_API_KEY as the fallback) in "
        "your .env. Gateways that route by URL path and accept an empty model "
        "name should also set JUDGE_PROVIDER=maas. See .env.example.\n"
        f"  JUDGE_PROVIDER={JUDGE_PROVIDER!r}  JUDGE_BASE_URL={JUDGE_BASE_URL!r}  "
        f"JUDGE_MODEL_NAME={JUDGE_MODEL_NAME!r}  api_key_set={bool(JUDGE_API_KEY)}"
    )


try:
    evaluation_llm_client = _create_judge_client(async_mode=True)
    model_as_a_judge_client = _create_judge_client(async_mode=False)
except Exception as _judge_init_error:  # pragma: no cover - configuration error
    # Importing this module must not explode with an opaque SDK error; the
    # benchmark runner checks JUDGE_MODEL_LABEL up front and reports this instead.
    raise RuntimeError(f"{_judge_config_error()}") from _judge_init_error

# Shared async httpx client for Gemini provider
_gemini_http_client: Optional[httpx.AsyncClient] = None


def _judge_extra_body() -> Dict[str, Any]:
    """Provider-specific extra fields passed via OpenAI SDK ``extra_body``.

    - Qwen3.x is a reasoning model; without disabling thinking, all
      output tokens go into ``reasoning_content`` and ``content`` ends up
      ``None``. Judge tasks only need deterministic verdicts, so we turn
      thinking off explicitly.
    """
    if JUDGE_PROVIDER == "maas":
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {}


def _extract_text_from_message(msg) -> str:
    """Pull text out of an OpenAI SDK ``ChatCompletionMessage``, with a
    fallback to ``reasoning_content`` for reasoning models that may leave
    ``content`` empty even when thinking is supposedly off.
    """
    text = getattr(msg, "content", None)
    if text:
        return text
    # Some non-standard fields (sglang/Qwen) live on model_extra.
    extra = getattr(msg, "model_extra", None) or {}
    rc = extra.get("reasoning_content")
    if rc:
        return rc
    # Last resort: dump and probe.
    try:
        d = msg.model_dump() if hasattr(msg, "model_dump") else dict(msg)
        return d.get("content") or d.get("reasoning_content") or ""
    except Exception:
        return ""


def _get_gemini_client() -> httpx.AsyncClient:
    """Lazy-init a shared async httpx client for Gemini judge requests."""
    global _gemini_http_client
    if _gemini_http_client is None or _gemini_http_client.is_closed:
        _gemini_http_client = httpx.AsyncClient(
            headers={
                "Content-Type": "application/json",
                "api-key": JUDGE_API_KEY or "",
            },
            timeout=httpx.Timeout(300.0, connect=30.0),
        )
    return _gemini_http_client


async def _judge_chat_completion(
    messages: list,
    model: Optional[str] = None,
    max_completion_tokens: Optional[int] = None,
) -> str:
    """Unified judge LLM completion that supports Gemini / Azure / OpenAI.

    Args:
        messages: OpenAI-style message list [{"role": ..., "content": ...}].
        model: Model name override (defaults to JUDGE_MODEL_NAME).
        max_completion_tokens: Max output tokens (optional).

    Returns:
        The assistant message content string.

    Raises:
        Exception: on API errors (callers handle retries).
    """
    model = model or JUDGE_MODEL_NAME

    if JUDGE_PROVIDER == "gemini":
        return await _judge_chat_completion_gemini(
            messages, model, max_completion_tokens
        )
    else:
        # Azure / MaaS / OpenAI — all speak the OpenAI SDK protocol
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = max_completion_tokens
        extra = _judge_extra_body()
        if extra:
            kwargs["extra_body"] = extra
        response = await evaluation_llm_client.chat.completions.create(**kwargs)
        return _extract_text_from_message(response.choices[0].message)


async def _judge_chat_completion_gemini(
    messages: list,
    model: str,
    max_completion_tokens: Optional[int] = None,
) -> str:
    """Call Gemini through a gateway that exposes its native generateContent API.

    URL: {JUDGE_BASE_URL}/google/v1:generateContent
    Auth: api-key header

    Thinking is disabled (thinkingBudget=0) since judge tasks only need
    deterministic grading — no chain-of-thought required.  This also avoids
    issues where maxOutputTokens conflicts with thinking token budgets.
    """
    base = (JUDGE_BASE_URL or "").rstrip("/")
    url = f"{base}/google/v1:generateContent"

    # Convert OpenAI messages to Gemini contents format
    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        gemini_role = "model" if role == "assistant" else "user"
        text = msg.get("content", "")
        if contents and contents[-1]["role"] == gemini_role:
            # Merge consecutive same-role messages
            contents[-1]["parts"].append({"text": text})
        else:
            contents.append({"role": gemini_role, "parts": [{"text": text}]})

    payload: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.0,
            # Disable thinking for judge tasks — deterministic grading only.
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    if max_completion_tokens is not None:
        # Gemini uses maxOutputTokens. Set a safe floor of 16 tokens to
        # avoid overly-restrictive limits (a 2-token cap works
        # for OpenAI but Gemini may produce slightly more formatting).
        payload["generationConfig"]["maxOutputTokens"] = max(
            max_completion_tokens, 16
        )

    client = _get_gemini_client()
    resp = await client.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()

    candidates = data.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        # Collect text parts, skip thinking/thought parts (should be none
        # with thinkingBudget=0, but defensive filtering anyway).
        text_parts = [
            p["text"] for p in parts if "text" in p and not p.get("thought")
        ]
        result = "\n".join(text_parts)

        usage = data.get("usageMetadata", {})
        tokens = usage.get("promptTokenCount", 0) + usage.get(
            "candidatesTokenCount", 0
        )
        logger.info("Judge Gemini tokens used: %d", tokens)

        return result

    if "error" in data:
        raise RuntimeError(f"Gemini judge API error: {data['error']}")
    raise RuntimeError(f"Unexpected Gemini judge response: {data}")



# ================================================
# verify_answer_hle
# ================================================

HLE_JUDGE_PROMPT = r"""Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available."""


class HLEExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True] = True  # 100% reliability


async def verify_answer_hle(question: str, target: str, predicted_answer: str) -> str:
    """
    Use HLE-style LLM judge to verify if the predicted answer is correct.
    Returns the evaluation result as a string: "CORRECT", "INCORRECT", or "NOT_ATTEMPTED".

    Args:
        question: The question being answered
        target: The correct/target answer
        predicted_answer: The model's predicted answer

    Returns:
        String indicating the evaluation result
    """
    prompt = HLE_JUDGE_PROMPT.format(
        question=question, correct_answer=target, response=predicted_answer
    )

    try:
        # Providers that don't support OpenAI structured output (response_format
        # with Pydantic schema) — ask for JSON and parse manually.
        if JUDGE_PROVIDER in ("gemini", "maas"):
            # Note: Qwen3.5-397B accepts response_format=json_object but
            # not response_format=<pydantic-class>, so we fall back to the
            # same JSON-instruction path as Gemini.
            json_instruction = (
                "\n\nRespond ONLY with a JSON object containing these fields: "
                '"extracted_final_answer" (string), "reasoning" (string), '
                '"correct" ("yes" or "no"), "confidence" (integer 0-100).'
            )
            content_str = await _judge_chat_completion(
                messages=[{"role": "user", "content": prompt + json_instruction}],
                max_completion_tokens=4096,
            )
            # Parse JSON from response (may be wrapped in markdown code blocks)
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", content_str, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(1))
            else:
                json_match = re.search(r"\{.*\}", content_str, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                else:
                    print(f"Warning: Could not parse HLE JSON from Gemini: {content_str[:200]}")
                    return "NOT_ATTEMPTED"

            reasoning = parsed.get("reasoning", "")
            correct = parsed.get("correct", "no")
            confidence = parsed.get("confidence", 0)

            print(f"LLM as Judge Reasoning: {reasoning}")
            print(f"LLM as Judge Result: {correct}")
            print(f"LLM as Judge Confidence: {confidence}%")

            if correct == "yes":
                return "CORRECT"
            else:
                return "INCORRECT"
        else:
            # OpenAI / Azure — use structured output
            response = await evaluation_llm_client.beta.chat.completions.parse(
                model=JUDGE_MODEL_NAME,
                max_completion_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                response_format=HLEExtractedAnswer,
                extra_body=_judge_extra_body() or None,
            )

            content = response.choices[0].message.parsed

            # Print HLE reasoning
            print(f"LLM as Judge Reasoning: {content.reasoning}")
            print(f"LLM as Judge Result: {content.correct}")
            print(f"LLM as Judge Confidence: {content.confidence}%")

            # Convert HLE format to eval_utils format
            if content.correct == "yes":
                return "CORRECT"
            else:
                return "INCORRECT"

    except Exception as e:
        # Never exit() here: this runs inside a worker process, SystemExit escapes
        # every `except Exception` above it, and the whole run dies mid-flight with
        # in-progress tasks lost. Bad credentials surface as NOT_ATTEMPTED like any
        # other judge failure, and _append_error_stats reports the count.
        print(f"LLM evaluation failed: {e}")
        return "NOT_ATTEMPTED"


# ================================================
# verify_answer_general — fallback judge for any benchmark without its own
# grader. Prompt from WebAgent
# https://github.com/Alibaba-NLP/WebAgent/blob/f25dae54daf0ce2874ffd5ed5ffb20feca7c4c4e/WebSailor/src/prompt.py#L98
# ================================================

GENERAL_JUDGE_PROMPT = """You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer.

Question: {question}

Labeled Answer: {correct_answer}

Predicted Answer: {response}

Did the model give an answer **equivalent** to the labeled answer? Please respond with "Correct" if they are equivalent, or "Incorrect" if they are not equivalent. Do not include any other text.
"""


async def verify_answer_general(
    question: str, target: str, predicted_answer: str
) -> str:
    prompt = GENERAL_JUDGE_PROMPT.format(
        question=question, correct_answer=target, response=predicted_answer
    )

    max_tries = 10
    for attempt in range(max_tries):
        try:
            content = await _judge_chat_completion(
                messages=[{"role": "user", "content": prompt}],
            )
            print("LLM Judge Response: ", content)

            if content:
                break
        except Exception as e:
            if attempt == (max_tries - 1):
                raise e

    # Use case-insensitive matching and strip whitespace/punctuation
    content_normalized = content.strip().rstrip(".").lower()
    if content_normalized == "correct":
        return "CORRECT"
    elif content_normalized == "incorrect":
        return "INCORRECT"
    else:
        # If we can't parse the response, default to NOT_ATTEMPTED to trigger retry
        print(f"Warning: Could not parse judge response: {content}")
        return "NOT_ATTEMPTED"


# ================================================
# verify_answer_browsecomp

# Prompt from Tongyi DeepResearch
# https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/WebWatcher/infer/evaluation/prompt.py#L110
# ================================================

JUDGE_PROMPT_BC_zh = """
请根据给定问题、标准答案和模型预测的答案来评估模型的回答是否正确。您的任务是将结果评定为：【正确】、【错误】。

首先，我们将列出每个评定类别的示例，然后请您对新问题的预测答案进行评定。
以下是【正确】的答复示例：
```
问题：贝拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：Malia Obama and Sasha Obama
模型预测2：玛丽亚和萨沙
模型预测3：大多数人会说是玛丽亚和萨莎，但我不确定，需要再确认
模型预测4：巴拉克·奥巴马有两个女儿，她们分别是玛丽亚·安和娜塔莎·玛丽安，但通常称作玛丽亚·奥巴马和萨莎·奥巴马。
```
这些答复均为【正确】，因为：
    - 完整地包含了标准答案中的重要信息。
    - 不包含任何与标准答案矛盾的信息。
    - 只关注语义内容，中英文，大小写、标点、语法和顺序不重要。
    - 答复中出现模糊语句或猜测是可以接受的，前提是包含了标准答案且不含有不正确信息或矛盾。

以下是【错误】的答复示例：
```
问题：巴拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：玛丽亚
模型预测2：玛丽亚、萨莎和苏珊和萨莎·奥巴马或玛丽亚·奥巴马，或娜塔莎·玛丽安，或爱因斯坦
模型预测3：虽然我不知道他们的确切名字，但能说出巴拉克·奥巴马有两个孩子。
模型预测4：你可能是想说贝茜和奥利维亚。不过您应通过最新的参考资料确认详细信息。那是正确的答案吗？
模型预测5：巴拉克·奥巴马的孩子
```
这些答复均为【错误】，因为：
    - 答复中包含与标准答案矛盾的事实陈述。
    - 答案为空、重复表述问题。
    - 答案枚举了多个答案，重复表述答案。

需要格外注意的是：
- 标准答案中包含对于问题中多个方面的回答，并且在同一个方面的答案中可能会有多种不同的描述，这些描述均是正确的，并且在同一个括号中给出，通过逗号连接。例如，考虑问题"抖音自己的人工智能大模型叫什么名字？"，标准答案为"【【豆包，云雀】】"：
    - 预测答案"豆包"、"豆包、云雀"、"云雀"等均为【正确】。
- 对于标准答案中包含的不同方面的回答，模型需要同时给出所有方面的回答才可以算是正确，否则直接判断为【错误】，不存在【部分正确】这种输出方式，这些答案会在不同的括号中给出。例如，考虑问题"TFBOYS组合中的成员有哪些？"，标准答案为"【【王俊凯】【王源】【易洋千玺】】"：
    - 预测答案"王俊凯、王源、易洋千玺"等同时包含所有答案，才可以算为【正确】。
    - 预测答案为"王俊凯、易洋千玺"等没有同时包含所有答案，会被算为【错误】。

另外注意以下几点：
- 对于标准答案为数字的问题，预测答案应和标准答案一致。例如，考虑问题"金山铁路黄浦江特大桥的全长是多少米？"，标准答案为"3518.17"：
    - 预测答案"3518"、"3518.1"、"3518.17"均为【正确】。
    - 预测答案"3520"和"3600"均为【错误】。
- 如果模型预测并没有直接回答问题，模型试图绕过或未能直接给出标准答案视为【错误】答案。
    - 例如：问题"林宥嘉的老婆是谁"，标准答案为"丁文琪"。模型预测"林宥嘉的老婆"、"林宥嘉的老婆应该很优秀"、"林宥嘉的老婆可能是某个公众人物"均为【错误】。
- 如果标准答案包含比问题更多的信息，预测答案只需包含问题中提到的信息。
    - 例如，考虑问题"菱镁矿的主要化学成分是什么？"标准答案为"碳酸镁（MgCO3）"。"碳酸镁"或"MgCO3"均视为【正确】答案。
- 如果从问题中明显可以推断出预测答案省略的信息，那么算作正确。
    - 例如，问题"巴鲁米尼的努拉吉遗迹在1997年被联合国教科文组织列为世界文化遗产，那么这遗址在哪个地区？"标准答案为"意大利撒丁岛"，预测答案"撒丁岛"被视为【正确】。
- 如果能明显看出名字翻译版本不同但是是同一个人也认为正确。
    - 例如，如果标准答案是"Robinson"，那么回答鲁滨逊或者鲁滨孙均正确。
- 你应该更关注标准答案和模型预测的匹配度，而不是关心标准答案是否是正确的。

下面是一个新的问题示例。请只回复【正确】、【错误】之一，不要道歉或纠正自己的错误，只需要评估该回答。
```
问题: {question}
标准答案: {correct_answer}
预测答案: {response}
```

将此新问题的预测答案评定为以下之一：
A.【正确】
B.【错误】

只返回【正确】、【错误】所代表的选项即可，即仅返回A或B即可，无须添加任何其他的文本。
""".strip()




JUDGE_PROMPT_BC_en = """
Based on the given question, standard answer, and model-predicted answer, evaluate whether the model's response is correct. Your task is to classify the result as: [CORRECT] or [INCORRECT].

First, we'll list examples for each category, then you'll evaluate a new question's predicted answer.
Here are examples of [CORRECT] responses:
```
Question: What are the names of Barack Obama's children?
Standard Answer: Malia Obama and Sasha Obama
Model Prediction 1: Malia Obama and Sasha Obama
Model Prediction 2: Malia and Sasha
Model Prediction 3: Most would say Malia and Sasha, but I'm not sure, I should verify
Model Prediction 4: Barack Obama has two daughters, Malia Ann and Natasha Marian, commonly known as Malia Obama and Sasha Obama.
```
These responses are all [CORRECT] because they:
    - Fully include the important information from the standard answer.
    - Don't contain any information that contradicts the standard answer.
    - Focus only on semantic content; language, capitalization, punctuation, grammar, and order aren't important.
    - Vague statements or guesses are acceptable as long as they include the standard answer and don't contain incorrect information or contradictions.

Here are examples of [INCORRECT] responses:
```
Question: What are the names of Barack Obama's children?
Standard Answer: Malia Obama and Sasha Obama
Model Prediction 1: Malia
Model Prediction 2: Malia, Sasha and Susan or Sasha Obama or Malia Obama, or Natasha Marian, or Einstein
Model Prediction 3: While I don't know their exact names, I can tell you Barack Obama has two children.
Model Prediction 4: You might be thinking of Betsy and Olivia. But you should verify the details with the latest references. Is that the correct answer?
Model Prediction 5: Barack Obama's children
```
These responses are all [INCORRECT] because they:
    - Contain factual statements that contradict the standard answer.
    - Are empty or merely repeat the question.
    - Enumerate multiple answers or repeat the answer.

Pay special attention to the following:
- The standard answer may contain responses to multiple aspects of the question, and within the same aspect, there might be different descriptions, all of which are correct and are given in the same bracket, connected by commas. For example, for the question "What is the name of ByteDance's AI model?", the standard answer is "[[Doubao, Skylark]]":
    - Predicted answers "Doubao", "Doubao, Skylark", "Skylark", etc. are all [CORRECT].
- For standard answers containing responses to different aspects, the model needs to provide answers to all aspects to be considered correct; otherwise, it's directly judged as [INCORRECT]. There is no [PARTIALLY CORRECT] output option. These answers will be given in different brackets. For example, for the question "Who are the members of TFBOYS?", the standard answer is "[[Wang Junkai][Wang Yuan][Yi Yangqianxi]]":
    - Predicted answers like "Wang Junkai, Wang Yuan, Yi Yangqianxi" that include all answers are [CORRECT].
    - Predicted answers like "Wang Junkai, Yi Yangqianxi" that don't include all answers are [INCORRECT].

Also note the following points:
- For questions with numerical standard answers, the predicted answer should match the standard answer. For example, for the question "What is the total length in meters of the Huangpu River Bridge on the Jinshan Railway?", the standard answer is "3518.17":
    - Predicted answers "3518", "3518.1", "3518.17" are all [CORRECT].
    - Predicted answers "3520" and "3600" are [INCORRECT].
- If the model prediction doesn't directly answer the question, attempts to circumvent or fails to directly provide the standard answer, it's considered an [INCORRECT] answer.
    - For example, for the question "Who is JJ Lin's wife?", with the standard answer "Ding Wenqi", model predictions like "JJ Lin's wife", "JJ Lin's wife should be excellent", "JJ Lin's wife might be a public figure" are all [INCORRECT].
- If the standard answer contains more information than the question asks for, the predicted answer only needs to include the information mentioned in the question.
    - For example, for the question "What is the main chemical component of magnesite?", with the standard answer "Magnesium carbonate (MgCO3)", "Magnesium carbonate" or "MgCO3" are both considered [CORRECT] answers.
- If information omitted in the predicted answer can be clearly inferred from the question, it's considered correct.
    - For example, for the question "The Nuragic ruins of Barumini were listed as a World Cultural Heritage by UNESCO in 1997, so where is this site located?", with the standard answer "Sardinia, Italy", the predicted answer "Sardinia" is considered [CORRECT].
- If it's clear that different translations of a name refer to the same person, it's considered correct.
    - For example, if the standard answer is "Robinson", answers like "Lubinson" or "Lubinsun" are both correct.
- You should focus more on the match between the standard answer and the model prediction, rather than whether the standard answer itself is correct.

Below is a new question example. Please reply with only [CORRECT] or [INCORRECT], without apologies or corrections to your own errors, just evaluate the answer.
```
Question: {question}
Standard Answer: {correct_answer}
Predicted Answer: {response}
```

Evaluate this new question's predicted answer as one of the following:
A. [CORRECT]
B. [INCORRECT]

Return only the option representing [CORRECT] or [INCORRECT], i.e., just return A or B, without adding any other text.
""".strip()




async def verify_answer_browsecomp(
    question: str, target: str, predicted_answer: str
) -> str:
    """
    Use BrowseComp judge (English version) to verify if the predicted answer is correct.
    Expects the LLM to return A (correct) or B (incorrect).
    """

    prompt = JUDGE_PROMPT_BC_en.format(
        question=question, correct_answer=target, response=predicted_answer
    )

    try:
        content = await _judge_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=16,
        )
        print(f"BrowseComp Judge Response: {content}")

        # Extract A or B from the response
        match = re.search(r"[AB]", content)
        if match:
            choice = match.group(0)
            if choice == "A":
                return "CORRECT"
            elif choice == "B":
                return "INCORRECT"

        # If no clear A or B is found, return NOT_ATTEMPTED to trigger retry
        print(f"Warning: Could not parse BrowseComp judge response: {content}")
        return "NOT_ATTEMPTED"

    except Exception as e:
        print(f"BrowseComp evaluation failed: {e}")
        raise e


async def verify_answer_browsecomp_zh(
    question: str, target: str, predicted_answer: str
) -> str:
    """
    Use BrowseComp judge (Chinese version) to verify if the predicted answer is correct.
    Expects the LLM to return A (correct) or B (incorrect).
    """

    prompt = JUDGE_PROMPT_BC_zh.format(
        question=question, correct_answer=target, response=predicted_answer
    )

    try:
        content = await _judge_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=16,
        )
        print(f"BrowseComp-ZH Judge Response: {content}")

        # Extract A or B from the response
        match = re.search(r"[AB]", content)
        if match:
            choice = match.group(0)
            if choice == "A":
                return "CORRECT"
            elif choice == "B":
                return "INCORRECT"

        # If no clear A or B is found, return NOT_ATTEMPTED to trigger retry
        print(f"Warning: Could not parse BrowseComp-ZH judge response: {content}")
        return "NOT_ATTEMPTED"

    except Exception as e:
        print(f"BrowseComp-ZH evaluation failed: {e}")
        raise e



# ================================================
# verify_answer_deepsearchqa
#
# Official prompt from DeepSearchQA benchmark
# https://www.kaggle.com/code/andrewmingwang/deepsearchqa-starter-code
# ================================================

JUDGE_PROMPT_DEEPSEARCHQA = """Your task is to evaluate whether a given "AI Response" for a specific "User Prompt" arrived at the correct answer.

**Answer Correctness Task**

*   **Purpose:** Assess whether the AI response provides the correct answer(s) based on the provided "Correct Answer" and "Prompt Type".
*   **Process:**
    *   Identify the "Prompt Type": "<prompt_type>".
    *   Refer to the "Correct Answer": "<answer>".
    *   Based on the "Prompt Type", determine if the "AI Response" contains the expected answer(s).
        *   **'Single Answer'**: Check if the response provides the answer that addresses the user's question. It does not have to match the exact wording of the provided answer.
        *   **'Set Answer'**: Check if the response includes *each* item from the provided ground truth answers. The order might not matter unless specified otherwise. The response might include more answers than the list. Determine the correctness *only* based on the list first and then check if the response includes answers not in the list.
    *   **Explanation:** Provide a brief explanation justifying your assessment of answer correctness, referencing specific parts of the AI response and the correct answer.
    *   **Correctness Details:** Provide a dictionary, one key for each expected answer part, and value is a boolean indicating whether each expected answer part was found.
        *   For 'Set Answer', this will be a list of attributes, one for each item/part in the "Correct Answer". Each key will be a string indicating the expected answer part, and the value will be a boolean indicating whether that part was found in the response.
    *   **Excessive Answers:** Provide a list of strings, each indicating an excessive answer part. If the response provides answers that are **not** in the "Correct Answer" list, add these answers as excessive answers. Return an empty list when there's no excessive answers in the response.


**Output Format:**

Your evaluation *must* be structured as a nested JSON dictionary with the following top-level keys: `"Answer Correctness"`. Please return NULL if any of "Prompt", "AI Response" or "Correct Answer" is empty.
The value for `"Answer Correctness"` should be a dictionary containing `"Explanation"` (a string), `"Correctness Details"` (a dictionary where each key is the expected correct answer, and the value is a boolean indicating whether the response contains the correct answer), and `"Excessive Answers"` (a list of strings indicating the excessive answers).

Make sure you return a valid JSON string. Pay special attention to quotes, commas and special characters in the JSON string. Make sure to escape all special characters and quotes in the JSON string.


**Example (Partial):**

"```json
{{
  "Answer Correctness": {{
    "Explanation": "The response correctly identified Belgium and France but also includes an excessive answer, Italy.",
    "Correctness Details": {{
      "Belgium": true,
      "France": true,
    }},
    "Excessive Answers": [ "Italy" ]
  }}
}}
```"

**Now, proceed with the evaluation using the provided User Prompt, AI Response, and Correct Answer.**

User Prompt (Wrapped in <prompt> and </prompt>):
<prompt>
{prompt}
</prompt>
--------------------
**  Correct Answer (Wrapped in <answer> and </answer>):
Prompt Type: {prompt_type}
<answer>
{answer}
</answer>
--------------------
AI assistant response (Wrapped in <response> and </response>):
<response>
{response}
</response>

--------------------
Rating:"""


async def verify_answer_deepsearchqa(
    question: str,
    target: str,
    predicted_answer: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> tuple[str, str, Optional[Dict[str, Any]]]:
    """
    Use DeepSearchQA-specific judge to verify if the predicted answer is correct.
    Uses the official DeepSearchQA evaluation prompt with JSON output format.

    Args:
        question: The question being answered
        target: The correct/target answer
        predicted_answer: The model's predicted answer
        metadata: Optional metadata dict with additional context (e.g., problem_category, answer_type)

    Returns:
        Tuple of (result, judge_type, details_dict):
        - result: "CORRECT", "INCORRECT", or "NOT_ATTEMPTED"
        - judge_type: "deepsearchqa_judge"
        - details_dict: Dict with keys:
            - correctness_details: Dict[str, bool] mapping answer parts to correctness
            - excessive_answers: List[str] of extra answers not in ground truth
            - explanation: str explaining the judgment
            - num_correct: int number of correct answer parts
            - num_expected: int total number of expected answer parts
            - num_excessive: int number of excessive answers
    """

    if predicted_answer is None:
        return "INCORRECT", "deepsearchqa_judge", None

    # Determine prompt_type from metadata
    prompt_type = "Single Answer"  # Default
    if metadata and "answer_type" in metadata:
        answer_type = metadata["answer_type"]
        # Map answer_type to prompt_type
        if answer_type == "Set Answer":
            prompt_type = "Set Answer"
        # Add more mappings if needed

    judge_prompt = JUDGE_PROMPT_DEEPSEARCHQA.format(
        prompt_type=prompt_type,
        prompt=question,
        answer=target,
        response=predicted_answer,
    )

    try:
        judge_response = await _judge_chat_completion(
            messages=[{"role": "user", "content": judge_prompt}],
        )
    except Exception as e:
        print(f"DeepSearchQA judge failed: {e}")
        return "NOT_ATTEMPTED", "deepsearchqa_judge", None

    if judge_response is None:
        return "NOT_ATTEMPTED", "deepsearchqa_judge", None

    # Parse JSON response
    try:
        # Extract JSON from the response (might be wrapped in markdown code blocks)
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", judge_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON without code blocks
            json_match = re.search(r"\{.*\}", judge_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                print("Warning: Could not find JSON in DeepSearchQA judge response")
                return "NOT_ATTEMPTED", "deepsearchqa_judge", None

        result = json.loads(json_str)
        answer_correctness = result.get("Answer Correctness", {})

        explanation = answer_correctness.get("Explanation", "")
        correctness_details = answer_correctness.get("Correctness Details", {})
        excessive_answers = answer_correctness.get("Excessive Answers", [])

        # Calculate statistics
        num_expected = len(correctness_details)
        num_correct = sum(1 for v in correctness_details.values() if v)
        num_excessive = len(excessive_answers)

        # Build details dict
        details = {
            "correctness_details": correctness_details,
            "excessive_answers": excessive_answers,
            "explanation": explanation,
            "num_correct": num_correct,
            "num_expected": num_expected,
            "num_excessive": num_excessive,
        }

        # Print debug info
        print(
            f"DeepSearchQA Judge - Correct: {num_correct}/{num_expected}, Excessive: {num_excessive}"
        )
        print(f"DeepSearchQA Judge - Explanation: {explanation}")

        # Determine if answer is correct
        # Following official logic: all expected parts must be found, and no excessive answers
        if correctness_details:
            all_correct = all(correctness_details.values())
            if all_correct and not excessive_answers:
                return "CORRECT", "deepsearchqa_judge", details
            else:
                # Either missing some expected answers or has excessive answers
                return "INCORRECT", "deepsearchqa_judge", details
        else:
            # No correctness details, can't determine
            return "NOT_ATTEMPTED", "deepsearchqa_judge", None

    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse JSON from DeepSearchQA judge: {e}")
        print(f"Response: {judge_response[:200]}...")
        return "NOT_ATTEMPTED", "deepsearchqa_judge", None
    except Exception as e:
        print(f"Warning: Error processing DeepSearchQA judge response: {e}")
        return "NOT_ATTEMPTED", "deepsearchqa_judge", None


# ================================================
# verify_answer_for_datasets
# ================================================


async def _verify_answer_for_datasets_core(
    benchmark_name: str,
    question: str,
    target: str,
    predicted_answer: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> tuple[str, str, Optional[Dict[str, Any]]]:
    """
    Verify the answer for a given dataset.

    Args:
        benchmark_name: Name of the benchmark dataset
        question: The question being answered
        target: The correct/target answer
        predicted_answer: The model's predicted answer
        metadata: Optional metadata dict with additional context

    Returns:
        A tuple of (result, judge_type, details_dict).
        details_dict is None for most benchmarks, but contains evaluation details for DeepSearchQA.
    """

    # Exact string match short-circuits the judge, except for DeepSearchQA whose
    # answers are sets and are always scored with F1.
    if not benchmark_name.startswith("deepsearchqa"):
        if predicted_answer == target:
            return "CORRECT", "exact_match", None

    # BrowseComp (English) and BrowseComp-ZH (Chinese) use different judge prompts.
    if benchmark_name.startswith("browsecomp") and benchmark_name != "browsecomp_zh":
        result = await verify_answer_browsecomp(question, target, predicted_answer)
        return result, "browsecomp_judge", None

    elif benchmark_name == "browsecomp_zh":
        result = await verify_answer_browsecomp_zh(question, target, predicted_answer)
        return result, "browsecomp_zh_judge", None

    # hle, hle-text-2158, and any other HLE subset
    elif "hle" in benchmark_name:
        result = await verify_answer_hle(question, target, predicted_answer)
        return result, "hle_judge", None

    # DeepSearchQA answers are sets: the judge marks each expected item present or
    # absent and flags excessive ones, and the caller turns that into F1.
    elif benchmark_name.startswith("deepsearchqa"):
        result, judge_type, details = await verify_answer_deepsearchqa(
            question, target, predicted_answer, metadata
        )
        return result, judge_type, details

    # Any benchmark you add: a general-purpose LLM-as-judge over short answers.
    else:
        result = await verify_answer_general(question, target, predicted_answer)
        return result, "general_judge", None


async def verify_answer_for_datasets(
    benchmark_name: str,
    question: str,
    target: str,
    predicted_answer: str,
    metadata: Optional[Dict[str, Any]] = None,
    max_retries: int = 10,
    retry_interval: int = 5,
) -> tuple[str, str, Optional[Dict[str, Any]]]:
    """
    Wrapper with retry logic for NOT_ATTEMPTED results.

    Args:
        benchmark_name: Name of the benchmark dataset
        question: The question being answered
        target: The correct/target answer
        predicted_answer: The model's predicted answer
        metadata: Optional metadata dict with additional context
        max_retries: Maximum number of retry attempts
        retry_interval: Seconds to wait between retries

    Returns:
        A tuple of (result, judge_type, details_dict).
        details_dict contains evaluation details (for DeepSearchQA) or None (for other benchmarks).
    """
    for attempt in range(1, max_retries + 1):
        result, judge_type, details = await _verify_answer_for_datasets_core(
            benchmark_name, question, target, predicted_answer, metadata
        )
        if result != "NOT_ATTEMPTED":
            return result, judge_type, details
        if attempt < max_retries:
            print(
                f"[Retry {attempt}/{max_retries}] Got NOT_ATTEMPTED, retrying in {retry_interval}s..."
            )
            await asyncio.sleep(retry_interval)

    # still NOT_ATTEMPTED after retries
    print(f"All {max_retries} attempts resulted in NOT_ATTEMPTED.")
    return "NOT_ATTEMPTED", "retry_wrapper", None
