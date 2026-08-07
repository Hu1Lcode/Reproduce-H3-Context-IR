"""Step 2: 跨模态关联（qwen3.6 API）。

输入：Step 1 语义描述 + Step 0 媒体角色
输出：subject_definitions / cross_modal_mapping / audio_roles /
      reference_relationships → ctx.cross_modal_mapping + ctx.subject_definitions
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


class Step2Associator(StepBase):
    name = "step2"

    def __init__(self, settings: Settings | None = None):
        super().__init__(settings)
        provider, model = self.settings.step2_model
        self._client = TextClient(provider, model, self.settings)
        self._system = _load_prompt("step2_associate.txt")

    def run(self, ctx: PipelineContext) -> PipelineContext:
        payload = {
            "task_type": ctx.task_type,
            "media_roles": ctx.media_roles,
            "prompt_analysis": ctx.semantic_descriptions.get("prompt_analysis", {}),
            "media_analysis": ctx.semantic_descriptions.get("media_analysis", {}),
        }
        raw = self._client.chat_json(
            self._system,
            json.dumps(payload, ensure_ascii=False, indent=1),
            temperature=0.2,
            max_tokens=self.settings.max_tokens_step,
        )
        ctx.cross_modal_mapping = raw
        ctx.subject_definitions = raw.get("subject_definitions", [])
        return ctx
