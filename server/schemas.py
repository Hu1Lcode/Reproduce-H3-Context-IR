"""请求/响应 Schema（兼容官方 /v2/h3_context_ir）。

官方请求：
    POST /v2/h3_context_ir
    {
      "model": "MiniMax-H3",
      "content": [
        {"type": "text", "text": "..."},
        {"type": "video_url", "url": "...", "role": "reference_video"},
        {"type": "audio_url", "url": "...", "role": "reference_audio"}
      ],
      "duration": 5,
      "ratio": "adaptive"
    }

官方响应（异步任务语义）：
    {"task_id": "..."}  →  GET /v2/query/video_generation/{task_id}
    → {"task": {"status": "...", "content": {"prompt": "..."}}}
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MediaRole(str, Enum):
    first_frame = "first_frame"
    last_frame = "last_frame"
    reference_video = "reference_video"
    reference_audio = "reference_audio"
    reference = "reference"


class ContentItem(BaseModel):
    type: str = Field(..., description="text / image_url / video_url / audio_url")
    text: str | None = None
    url: str | None = None
    role: MediaRole | str | None = None
    asset_id: str | None = None
    duration: float | None = None  # 素材时长（秒），用于校验


class H3ContextIRRequest(BaseModel):
    model: str = "MiniMax-H3"
    content: list[ContentItem]
    duration: float | None = Field(None, ge=1, le=15, description="目标视频时长（秒）")
    ratio: str | None = "adaptive"
    callback_url: str | None = None  # 可选回调（本实现暂不支持，仅透传）


class CreateResponse(BaseModel):
    task_id: str


class TaskStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    success = "success"
    failed = "failed"


class TaskContent(BaseModel):
    prompt: str | None = None
    task_type: str | None = None
    validation: dict[str, Any] | None = None


class QueryResponse(BaseModel):
    task_id: str
    status: TaskStatus
    content: TaskContent | None = None
    error: str | None = None
