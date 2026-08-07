"""官方输入规格校验。

Ref2VA 模式输入规格（来自 README / HF 模型卡）：
    - 图像 ≤ 9 张
    - 视频 ≤ 3 段；每段 2-15 秒；总时长 ≤ 15 秒
    - 音频 ≤ 3 段；音频必须搭配图像或视频输入（不能作为唯一输入）；
      每段 2-15 秒；总时长 ≤ 15 秒
    - 混合输入：全部文件总数 ≤ 12

Base 模式（FL2VA 变体）：
    - 图像 0-2 张（0 = T2VA，1 = I2VA/L2VA，2 = FL2VA）

注意：官方规格中的时长限制针对 Ref2VA 参考素材；T2VA/I2VA/L2VA/FL2VA
对参考素材另有宽松处理，但本项目统一按官方 Ref2VA 规格校验参考素材。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 官方限制常量
MAX_IMAGES = 9
MAX_VIDEOS = 3
MAX_AUDIOS = 3
MAX_FILES = 12
MIN_CLIP_SECONDS = 2.0
MAX_CLIP_SECONDS = 15.0
MAX_TOTAL_VIDEO_SECONDS = 15.0
MAX_TOTAL_AUDIO_SECONDS = 15.0


@dataclass
class ValidationResult:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings}


def _get_duration(item: dict[str, Any]) -> float | None:
    """读取条目时长（秒）：支持 duration / duration_ms / duration_seconds。"""
    for key in ("duration", "duration_seconds"):
        if key in item:
            return float(item[key])
    if "duration_ms" in item:
        return float(item["duration_ms"]) / 1000.0
    return None


def validate_input(content: list[dict[str, Any]], task_type: str | None = None) -> ValidationResult:
    """校验输入 content 是否满足官方规格。

    Args:
        content: 官方格式 content 数组。
        task_type: 可选的已知任务类型（由 classify_task 得到，避免重复分类）。
    """
    result = ValidationResult()

    images = [c for c in content if str(c.get("type", "")).lower().startswith("image")]
    videos = [c for c in content if str(c.get("type", "")).lower().startswith("video")]
    audios = [c for c in content if str(c.get("type", "")).lower().startswith("audio")]

    # --- 数量限制 ---
    if len(images) > MAX_IMAGES:
        result.add_error(f"图像数量 {len(images)} 超过上限 {MAX_IMAGES}")
    if len(videos) > MAX_VIDEOS:
        result.add_error(f"视频数量 {len(videos)} 超过上限 {MAX_VIDEOS}")
    if len(audios) > MAX_AUDIOS:
        result.add_error(f"音频数量 {len(audios)} 超过上限 {MAX_AUDIOS}")
    if len(images) + len(videos) + len(audios) > MAX_FILES:
        result.add_error(
            f"媒体文件总数 {len(images) + len(videos) + len(audios)} 超过上限 {MAX_FILES}"
        )

    # --- 音频必须搭配图像或视频 ---
    if audios and not (images or videos):
        result.add_error("音频输入必须搭配图像或视频输入，不能作为唯一输入")

    # --- 每段时长 2-15s ---
    for kind, items in (("视频", videos), ("音频", audios)):
        for i, item in enumerate(items):
            d = _get_duration(item)
            if d is None:
                result.add_warning(f"{kind}[{i}] 未提供时长，跳过时长校验（建议补全）")
                continue
            if d < MIN_CLIP_SECONDS or d > MAX_CLIP_SECONDS:
                result.add_error(
                    f"{kind}[{i}] 时长 {d:.2f}s 超出官方范围 "
                    f"[{MIN_CLIP_SECONDS:.0f}s, {MAX_CLIP_SECONDS:.0f}s]"
                )

    # --- 总时长限制 ---
    video_total = sum(_get_duration(v) or 0.0 for v in videos)
    if video_total > MAX_TOTAL_VIDEO_SECONDS:
        result.add_error(
            f"视频总时长 {video_total:.2f}s 超过上限 {MAX_TOTAL_VIDEO_SECONDS:.0f}s"
        )
    audio_total = sum(_get_duration(a) or 0.0 for a in audios)
    if audio_total > MAX_TOTAL_AUDIO_SECONDS:
        result.add_error(
            f"音频总时长 {audio_total:.2f}s 超过上限 {MAX_TOTAL_AUDIO_SECONDS:.0f}s"
        )

    return result
