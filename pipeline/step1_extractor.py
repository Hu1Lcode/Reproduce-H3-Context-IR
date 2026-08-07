"""Step 1: 语义提取（qwen3.6-plus API）。

输入：prompt + 预处理结果（媒体描述、帧 URL、音频语义描述）
输出：结构化 JSON（prompt_analysis + media_analysis），写入
      ctx.semantic_descriptions。
"""
from __future__ import annotations

import json
from pathlib import Path

from api.text_client import TextClient, parse_json_response
from api.vl_client import VLClient, make_image_part
from config.settings import Settings
from pipeline.base_pipeline import StepBase
from pipeline.context import PipelineContext


def _load_prompt(name: str) -> str:
    path = Path(__file__).resolve().parent.parent / "config" / "prompts" / name
    return path.read_text(encoding="utf-8")


class Step1Extractor(StepBase):
    name = "step1"

    def __init__(self, settings: Settings | None = None):
        super().__init__(settings)
        provider, model = self.settings.step1_model
        self._client = VLClient(provider, model, self.settings)
        self._system = _load_prompt("step1_extract.txt")

    def run(self, ctx: PipelineContext) -> PipelineContext:
        # --- 组装 user 输入 ---
        content = ctx.content
        prompt_text = next(
            (c.get("text", "") for c in content if str(c.get("type", "")).lower() == "text"),
            "",
        )
        media = [c for c in content if str(c.get("type", "")).lower() != "text"]

        # 媒体描述（预处理器产出）+ 参考帧图片
        pre = ctx.preprocessed
        media_descs = pre.get("media_descriptions", {})
        user_media: list[dict] = []
        media_infos = []
        for item in media:
            asset_id = str(item.get("asset_id") or item.get("id") or "")
            desc = media_descs.get(asset_id, {})
            info = {
                "asset_id": asset_id,
                "type": str(item.get("type", "")).lower(),
                "description": desc,
            }
            media_infos.append(info)
            # 视频参考帧：作为图片注入视觉模型
            frames = desc.get("frame_paths", []) if isinstance(desc, dict) else []
            for fp in frames[:5]:
                if Path(fp).exists():
                    user_media.append(make_image_part(fp))

        user_payload = {
            "prompt": prompt_text,
            "media": media_infos,
            "duration": ctx.duration,
            "ratio": ctx.ratio,
        }
        user_text = (
            "Here is the input. Respond with the structured JSON as specified.\n\n"
            + json.dumps(user_payload, ensure_ascii=False, indent=1)
        )

        raw = self._client.chat_multimodal(
            system=self._system,
            user_text=user_text,
            media=user_media or None,
            temperature=0.2,
            max_tokens=self.settings.max_tokens_step,
        )
        ctx.semantic_descriptions = parse_json_response(raw)
        return ctx


# 便捷函数：文本-only 场景（无图片/视频帧时）
def run_step1_text(
    prompt_text: str,
    preprocessed: dict,
    duration: float | None,
    ratio: str | None,
    settings: Settings,
) -> dict:
    """文本版 Step 1（调试/无多模态端点时使用）。"""
    client = TextClient(*settings.step1_model, settings)
    system = _load_prompt("step1_extract.txt")
    payload = {"prompt": prompt_text, "media": [], "duration": duration, "ratio": ratio}
    raw = client.chat_text(system, json.dumps(payload, ensure_ascii=False), temperature=0.2)
    return parse_json_response(raw)
