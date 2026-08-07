"""多模态调用（图片 / 视频 / 音频 URL 或 Base64）。

qwen3.6-plus 等视觉模型支持图片/视频 URL 直传；音频需先经 Captioner
转为语义描述（见 audio_captioner.py），或以文本形式注入。
"""
from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from api.base_client import BaseClient


def _guess_media_type(path: str | Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def make_image_part(url_or_path: str | Path) -> dict[str, Any]:
    """构造 image_url 消息片段（支持本地文件自动转 data URL）。"""
    return {"type": "image_url", "image_url": {"url": to_url_or_data(url_or_path)}}


def make_video_part(url_or_path: str | Path) -> dict[str, Any]:
    """构造 video 消息片段。

    注意：视频直接以 URL 传入（OpenAI 兼容协议下部分模型支持
    video_url 或视频路径）；本地文件请先上传（media_uploader）。
    """
    return {"type": "video_url", "video_url": {"url": to_url_or_data(url_or_path)}}


def to_url_or_data(url_or_path: str | Path) -> str:
    """URL 直通；本地文件转 data URL（仅适合小文件，如单帧图片）。"""
    s = str(url_or_path)
    if s.startswith(("http://", "https://", "data:", "file://")):
        return s
    path = Path(s)
    if not path.exists():
        raise FileNotFoundError(f"媒体文件不存在: {path}")
    mime = _guess_media_type(path)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


class VLClient(BaseClient):
    """多模态视觉语言模型客户端。"""

    def chat_multimodal(
        self,
        system: str,
        user_text: str,
        media: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        """多模态对话。

        Args:
            system: 系统提示。
            user_text: 用户文本。
            media: 消息片段列表，如 [{"type": "image_url", ...}]。
        """
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        if media:
            content.extend(media)
        return self.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
