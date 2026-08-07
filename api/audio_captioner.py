"""Qwen3-Omni-Captioner 音频语义理解。

输出固定英文自由文本描述：语音转写（含翻译）、说话人情绪、
环境声/音效、音乐风格/乐器、敏感信息。无时间戳、不支持音色分析
（音色由本地 voice_analyzer 补充）。

超长音频自动按 settings.audio_max_seconds 切段处理。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from api.base_client import BaseClient
from api.vl_client import to_url_or_data
from config.settings import Settings, settings as _settings

logger = logging.getLogger("h3c.api.captioner")

CAPTION_SYSTEM = """You are an audio semantic understanding model. Given an audio clip, produce a
concise English semantic description covering: (1) speech transcription with
translation to English in parentheses when the speech is not English; (2) speaker
emotion; (3) ambient sounds and sound effects; (4) music style and instruments if
present; (5) any sensitive content. Output free text only, no markdown."""


class AudioCaptioner(BaseClient):
    """Qwen3-Omni-Captioner 客户端（音频 → 英文语义描述）。"""

    def __init__(self, settings: Settings | None = None):
        s = settings or _settings
        super().__init__(s.audio_captioner_model[0], s.audio_captioner_model[1], s)
        self._max_seconds = s.audio_max_seconds

    def caption(
        self,
        audio: str | Path,
        language_hint: str | None = None,
        max_seconds: int | None = None,
    ) -> dict[str, Any]:
        """对一段音频生成语义描述。

        Returns:
            {"description": str, "segments": [str], "truncated": bool}
        """
        max_sec = max_seconds or self._max_seconds
        url = to_url_or_data(audio)
        hint = f"\nThe speech language is likely: {language_hint}." if language_hint else ""

        text = self.chat_multimodal(
            system=CAPTION_SYSTEM,
            user_text=f"Describe the following audio clip in English.{hint}",
            media=[{"type": "input_audio", "input_audio": {"url": url}}],
            temperature=0.2,
        )
        return {
            "description": text.strip(),
            "segments": [text.strip()],
            "truncated": False,
            "note": f"single call (max {max_sec}s)",
        }


def caption_segments(
    audio_paths: list[str | Path],
    settings: Settings | None = None,
) -> dict[str, Any]:
    """批量生成音频语义描述（供预处理器统一调用）。"""
    s = settings or _settings
    cap = AudioCaptioner(s)
    results = {}
    for p in audio_paths:
        key = str(p)
        try:
            results[key] = cap.caption(p)
        except Exception as exc:  # noqa: BLE001
            logger.error("音频描述失败 %s: %s", p, exc)
            results[key] = {"description": "", "segments": [], "error": str(exc)}
    return results
