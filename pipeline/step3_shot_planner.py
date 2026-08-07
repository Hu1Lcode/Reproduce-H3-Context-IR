"""Step 3: Shot 分割与描述（deepseek-v4-flash API）。

输入：Step 1/2 输出 + 场景切分结果
输出：镜头时间线 JSON → ctx.shot_timeline
"""
from __future__ import annotations

import json
from pathlib import Path

from api.text_client import TextClient, parse_json_response
from config.settings import Settings
from pipeline.base_pipeline import StepBase
from pipeline.context import PipelineContext


def _load_prompt(name: str) -> str:
    path = Path(__file__).resolve().parent.parent / "config" / "prompts" / name
    return path.read_text(encoding="utf-8")


class Step3ShotPlanner(StepBase):
    name = "step3"

    def __init__(self, settings: Settings | None = None):
        super().__init__(settings)
        provider, model = self.settings.step3_model
        self._client = TextClient(provider, model, self.settings)
        self._system = _load_prompt("step3_shot_plan.txt")

    def run(self, ctx: PipelineContext) -> PipelineContext:
        payload = {
            "task_type": ctx.task_type,
            "duration": ctx.duration,
            "ratio": ctx.ratio,
            "prompt_analysis": ctx.semantic_descriptions.get("prompt_analysis", {}),
            "media_analysis": ctx.semantic_descriptions.get("media_analysis", {}),
            "subject_definitions": ctx.subject_definitions,
            "cross_modal_mapping": ctx.cross_modal_mapping,
            "scene_changes": ctx.preprocessed.get("scene_changes", {}),
        }
        raw = self._client.chat_json(
            self._system,
            json.dumps(payload, ensure_ascii=False, indent=1),
            temperature=self.settings.temperature_step3,
            max_tokens=self.settings.max_tokens_step,
        )
        ctx.shot_timeline = raw
        return ctx
