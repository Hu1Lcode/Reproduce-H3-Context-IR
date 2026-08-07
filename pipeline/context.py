"""PipelineContext：阶段间数据传递对象（见复现方案第 10 节）。"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class PipelineContext:
    # --- 原始输入（官方 content 数组 + 任务参数）---
    raw_input: dict[str, Any] = field(default_factory=dict)
    # 官方格式：{"content": [...], "duration": 10, "ratio": "16:9", "model": "MiniMax-H3"}

    # --- Step 0（规则引擎）---
    task_type: str = ""                # t2va / i2va / l2va / fl2va / ref2va
    media_roles: dict[str, str] = field(default_factory=dict)

    # --- 预处理结果（本地 CPU + Captioner API）---
    preprocessed: dict[str, Any] = field(default_factory=dict)
    # {media_descriptions: {...}, scene_changes: [...], audio_features: {...}}

    # --- Step 1（qwen3.6-plus API）---
    semantic_descriptions: dict[str, Any] = field(default_factory=dict)
    # {prompt_analysis: {...}, media_analysis: {...}}

    # --- Step 2（qwen3.6 API）---
    cross_modal_mapping: dict[str, Any] = field(default_factory=dict)
    subject_definitions: list[str] = field(default_factory=list)

    # --- Step 3（deepseek-v4-flash API）---
    shot_timeline: dict[str, Any] = field(default_factory=dict)

    # --- Step 4（deepseek-v4-flash API）---
    validation: dict[str, Any] = field(default_factory=dict)
    # {corrections, retention_analysis, inferred_details, consistency_notes, is_consistent}

    # --- Step 5（deepseek-v4-flash API / 确定性模板渲染）---
    final_prompt: str = ""
    validation_report: dict[str, Any] = field(default_factory=dict)

    # --- 元信息 ---
    log: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PipelineContext":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def log_step(self, step: str, status: str, detail: str = "") -> None:
        self.log.append({"step": step, "status": status, "detail": detail})

    @property
    def duration(self) -> float | None:
        d = self.raw_input.get("duration")
        return float(d) if d is not None else None

    @property
    def ratio(self) -> str | None:
        return self.raw_input.get("ratio")

    @property
    def content(self) -> list[dict[str, Any]]:
        return self.raw_input.get("content", [])
