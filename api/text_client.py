"""文本模型调用（qwen3.6 / deepseek-v4-flash 等纯文本模型）。"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from api.base_client import BaseClient, APIClientError

logger = logging.getLogger("h3c.api.text")


class TextClient(BaseClient):
    """纯文本对话客户端，附带 JSON 输出解析辅助。"""

    def chat_text(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        return self.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def chat_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """对话并要求输出 JSON，自动剥离 markdown 围栏并解析。

        Raises:
            APIClientError: 解析失败。
        """
        text = self.chat_text(
            system=system,
            user=user + "\n\nOutput strictly as JSON without any markdown fences.",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return parse_json_response(text)


def parse_json_response(text: str) -> dict[str, Any]:
    """解析模型输出的 JSON（容忍 markdown 围栏与前后噪声）。"""
    if not text:
        raise APIClientError("模型返回空内容")
    stripped = text.strip()
    # 去掉 ```json ... ``` 围栏
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        # 兜底：提取第一个 {...} 块
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise APIClientError(f"模型输出不是合法 JSON: {text[:300]!r}")
