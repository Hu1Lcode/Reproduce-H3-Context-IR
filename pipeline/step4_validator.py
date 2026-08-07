"""Step 4: 逻辑校验（deepseek-v4-flash API）。

输入：Step 1-3 输出
输出：corrections / retention_analysis / inferred_details →
      ctx.validation
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


class Step4Validator(StepBase):
    name = "step4"

    def __init__(self, settings: Settings | None = None):
        super().__init__(settings)
        provider, model = self.settings.step4_model
        self._client = TextClient(provider, model, self.settings)
        self._system = _load_prompt("step4_validate.txt")

    def run(self, ctx: PipelineContext) -> PipelineContext:
        payload = {
            "task_type": ctx.task_type,
            "duration": ctx.duration,
            "subject_definitions": ctx.subject_definitions,
            "shot_timeline": ctx.shot_timeline,
        }
        raw = self._client.chat_json(
            self._system,
            json.dumps(payload, ensure_ascii=False, indent=1),
            temperature=self.settings.temperature_step4,
            max_tokens=self.settings.max_tokens_step,
        )
        ctx.validation = raw
        return ctx
