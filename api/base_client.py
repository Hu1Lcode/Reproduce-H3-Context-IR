"""OpenAI 兼容接口封装：重试 / 超时 / 限流。

所有 provider（DashScope / DeepSeek / 本地 vLLM）统一走 OpenAI 协议，
仅 base_url / api_key / model 不同。
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from openai import AsyncOpenAI, OpenAI

from config.settings import Settings, settings as _settings

logger = logging.getLogger("h3c.api")

# 可重试的状态码（限流/服务端错误）
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class APIClientError(RuntimeError):
    """API 调用最终失败（重试耗尽）。"""


class EmptyContentError(RuntimeError):
    """模型返回空 content（思考型模型 max_tokens 被思考占满时可能出现）。"""


class BaseClient:
    """OpenAI 兼容客户端基类。

    用法（同步）：
        client = BaseClient("dashscope", "qwen3.6-plus")
        resp = client.chat(messages=[...], temperature=0.1)

    用法（异步）：
        await client.achat(messages=[...])
    """

    def __init__(self, provider: str, model: str | None = None, settings: Settings | None = None):
        from api.providers import resolve_provider

        self.settings = settings or _settings
        cfg = resolve_provider(provider, self.settings)
        if cfg.requires_key and not cfg.api_key:
            raise RuntimeError(
                f"provider '{provider}' 未配置 API key，请在 config/config.yaml 中"
                f"配置 api_key，或设置环境变量 {cfg.api_key_env}"
            )
        self.provider = provider
        self.model = model or cfg.default_model
        self.base_url = cfg.base_url
        self.api_key = cfg.api_key or "EMPTY"

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.settings.request_timeout,
            max_retries=0,  # 重试由本类实现（可观察）
        )
        self._aclient = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.settings.request_timeout,
            max_retries=0,
        )
        self._sem = asyncio.Semaphore(self.settings.max_concurrent)

    # ------------------------------------------------------------------
    def _retry(self, fn, *args, **kwargs):
        """同步调用带指数退避重试。"""
        last_exc: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                status = getattr(exc, "status_code", None)
                if not _should_retry(exc, status) or attempt >= self.settings.max_retries:
                    break
                backoff = self.settings.retry_backoff_base**attempt
                logger.warning(
                    "API 调用失败（第 %d 次重试，%.1fs 后）: %s",
                    attempt + 1, backoff, exc,
                )
                time.sleep(backoff)
        raise APIClientError(f"{self.provider}/{self.model} 调用失败: {last_exc}") from last_exc

    async def _aretry(self, fn, *args, **kwargs):
        """异步调用带指数退避重试 + 限流信号量。"""
        last_exc: Exception | None = None
        async with self._sem:
            for attempt in range(self.settings.max_retries + 1):
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    status = getattr(exc, "status_code", None)
                    if not _should_retry(exc, status) or attempt >= self.settings.max_retries:
                        break
                    backoff = self.settings.retry_backoff_base**attempt
                    logger.warning(
                        "API 调用失败（第 %d 次重试，%.1fs 后）: %s",
                        attempt + 1, backoff, exc,
                    )
                    await asyncio.sleep(backoff)
        raise APIClientError(f"{self.provider}/{self.model} 调用失败: {last_exc}") from last_exc

    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        """同步对话补全，返回文本。"""

        def _call():
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            if response_format:
                kwargs["response_format"] = response_format
            resp = self._client.chat.completions.create(**kwargs)
            return _extract_content(resp)

        return self._retry(_call)

    async def achat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        """异步对话补全，返回文本。"""

        async def _call():
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            if response_format:
                kwargs["response_format"] = response_format
            resp = await self._aclient.chat.completions.create(**kwargs)
            return _extract_content(resp)

        return await self._aretry(_call)


def _extract_content(resp: Any) -> str:
    """从响应提取文本；content 为空（思考截断等）时抛可重试异常。"""
    message = resp.choices[0].message
    content = (message.content or "").strip()
    if not content:
        reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
        raise EmptyContentError(
            "模型返回空 content（思考长度 "
            f"{len(reasoning or '')}，可能 max_tokens 被思考过程占满）"
        )
    return content


def _should_retry(exc: Exception, status: int | None) -> bool:
    """判断异常是否值得重试。"""
    if isinstance(exc, EmptyContentError):
        return True
    if status is not None:
        return status in RETRYABLE_STATUS
    # 网络层错误（超时 / 连接中断）
    msg = str(exc).lower()
    return any(k in msg for k in ("timeout", "timed out", "connection", "reset", "network"))
