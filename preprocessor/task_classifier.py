"""Step 0: 任务区分（规则引擎，无模型）。

按官方 content 数组中各素材的 role 字段映射 5 种任务类型：
    [text]                                   -> t2va   (文生视频)
    [text, img(role=first_frame)]            -> i2va   (首帧生视频)
    [text, img(role=last_frame)]             -> l2va   (末帧生视频)
    [text, first_frame + last_frame]         -> fl2va  (首末帧生视频)
    [text, 任一 reference_*]                  -> ref2va (全参考生视频)

官方 content 条目类型（type 字段）：
    text / image_url / video_url / audio_url
官方 role 字段（content role，与视频生成任务相关）：
    first_frame / last_frame / reference_video / reference_audio
    （另见官方 API 文档中的 other role 值，这里只关心任务区分所需子集）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 官方 content 条目允许的 type -> 简写
TYPE_SHORT = {
    "text": "text",
    "image": "image",
    "image_url": "image",
    "video": "video",
    "video_url": "video",
    "audio": "audio",
    "audio_url": "audio",
}

TASK_TYPES = ("t2va", "i2va", "l2va", "fl2va", "ref2va")


@dataclass
class TaskClassification:
    task_type: str
    media_roles: dict[str, str] = field(default_factory=dict)  # asset_id -> role
    has_first_frame: bool = False
    has_last_frame: bool = False
    has_reference: bool = False
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "media_roles": self.media_roles,
            "has_first_frame": self.has_first_frame,
            "has_last_frame": self.has_last_frame,
            "has_reference": self.has_reference,
            "problems": self.problems,
        }


class TaskClassifierError(ValueError):
    """输入 content 无法映射到任何任务类型。"""


def classify_task(content: list[dict[str, Any]]) -> TaskClassification:
    """根据 content 数组的 type + role 判定任务类型。

    Args:
        content: 官方格式的 content 数组。每条目为 dict，至少含 "type"；
            role 字段可选（image/video/audio 条目上出现）。

    Returns:
        TaskClassification

    Raises:
        TaskClassifierError: 无 text 条目，或类型未知。
    """
    media_roles: dict[str, str] = {}
    has_text = False
    has_first = False
    has_last = False
    has_reference = False
    problems: list[str] = []
    used_ids: set[str] = set()

    for idx, item in enumerate(content):
        ctype = TYPE_SHORT.get(str(item.get("type", "")).strip().lower(), None)
        if ctype is None:
            problems.append(f"content[{idx}]: 未知 type={item.get('type')!r}")
            continue
        if ctype == "text":
            has_text = True
            continue

        # 媒体条目：id 用于后续步骤引用
        asset_id = str(item.get("asset_id") or item.get("id") or f"Media {idx}")
        if asset_id in used_ids:
            problems.append(f"asset_id 重复: {asset_id}")
        used_ids.add(asset_id)

        role = str(item.get("role", "")).strip().lower()
        media_roles[asset_id] = role
        if role == "first_frame":
            has_first = True
        elif role == "last_frame":
            has_last = True
        elif role.startswith("reference_"):
            has_reference = True
        elif role:
            problems.append(f"{asset_id}: 未知 role={role!r}（保留原样）")

    if not has_text:
        raise TaskClassifierError("content 必须包含至少一个 text 条目")

    if has_first and has_last:
        task = "fl2va"
    elif has_first:
        task = "i2va"
    elif has_last:
        task = "l2va"
    elif has_reference:
        task = "ref2va"
    else:
        task = "t2va"

    return TaskClassification(
        task_type=task,
        media_roles=media_roles,
        has_first_frame=has_first,
        has_last_frame=has_last,
        has_reference=has_reference,
        problems=problems,
    )
