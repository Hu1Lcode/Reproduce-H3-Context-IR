"""Step 5: 格式化输出（确定性模板渲染 + 可选 LLM 润色）。

设计（复现方案 9.10）：
    1. 结构化 JSON（shot_timeline 等）先按官方模板【确定性拼接】——
       字段名/顺序/时间戳/标签由代码保证，格式 100% 可控；
    2. 可选：LLM 语言润色（温度 ≤ 0.2，settings.llm_polish）；
    3. output_validator 正则兜底校验（修复 + 报告）。

模板矩阵（5 套）：
    t2va / i2va / fl2va / l2va → 三字段（指令行不同）
    ref2va                      → 六字段
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from api.output_validator import auto_fix, validate_prompt
from api.text_client import TextClient
from config.settings import Settings
from pipeline.base_pipeline import StepBase
from pipeline.context import PipelineContext

logger = logging.getLogger("h3c.pipeline.step5")


def _load_prompt(name: str) -> str:
    path = Path(__file__).resolve().parent.parent / "config" / "prompts" / name
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def format_ts(seconds: float | None) -> str:
    """秒 → MM:SS.mmm（如 3.5 → 00:03.500）。"""
    if seconds is None:
        return ""
    s = max(0.0, float(seconds))
    m = int(s // 60)
    rest = s - m * 60
    return f"{m:02d}:{rest:06.3f}"


def format_ts_secs(seconds: float | None) -> str:
    """秒 → S.SS（如 3.5 → 3.50；0.0 → 0.00），恰好两位小数。"""
    if seconds is None:
        return ""
    return f"{max(0.0, float(seconds)):.2f}"


def _trim(s: str | None) -> str:
    return (s or "").strip()


CAMERA_FIXES = [
    (r"\s*with zero amplitude and zero speed\b", ""),
    (r"\s*with zero amplitude\b", ""),
    (r"\s*with zero speed\b", ""),
    (r"\s*with moderate amplitude\b", ""),
    (r"\s*with medium amplitude\b", ""),
    (r"\s*at (a )?(steady|normal) speed\b", ""),
]

LABEL_RE = re.compile(r"<?(?:Subject|Audio|Picture|Video) \d+>?")


def _filter_labels(text: str, task_type: str) -> str:
    """Base 模式删除 <Subject N> 等标签（仅 Ref2VA 保留），避免泄漏到最终输出。"""
    if task_type == "ref2va" or not text:
        return text
    return re.sub(r"\s*" + LABEL_RE.pattern + r"\s*", " ", text).strip()


def _clean_style(style: str) -> str:
    """删除风格字段中的非英文片段（中文残留），空则回退默认。"""
    s = re.sub(r"[^\x00-\x7F]+", "", style or "").strip().strip(",").strip()
    return s


def _normalize_camera(camera: str) -> str:
    """运镜文本规范化：清理非官方振幅/速度词汇（Guide 仅 small/large、slow/fast）。"""
    c = camera.strip()
    for pattern, repl in CAMERA_FIXES:
        c = re.sub(pattern, repl, c)
    c = re.sub(r"\s{2,}", " ", c).strip()
    # 静态镜头收尾
    if re.search(r"static shot\s*\.?$", c, re.IGNORECASE) and not c.rstrip().endswith("."):
        c += "."
    return c


# ---------------------------------------------------------------------------
# 确定性渲染：Shot 级
# ---------------------------------------------------------------------------
def render_shot(shot: dict[str, Any], task_type: str) -> str:
    """把单个 shot 结构渲染为官方格式的自然语言描述。"""
    parts: list[str] = []

    # 镜头标记
    shot_id = int(shot.get("shot_id", 1))
    start = shot.get("start_time")
    if shot_id == 1:
        parts.append(f"[Shot 1]")
    else:
        parts.append(f"[Shot {shot_id}] At {format_ts(start)},")

    # 视觉主体（非 ref2va 任务不使用 <Subject N> 等标签，仅保留描述性主体）
    visual: list[str] = []
    comp = _filter_labels(_trim(shot.get("composition")), task_type)
    if comp:
        visual.append(comp)
    subjects = shot.get("subjects") or []
    if task_type != "ref2va":
        subjects = [
            str(s)
            for s in subjects
            if not re.fullmatch(r"<?(?:Subject|Audio|Picture|Video) \d+>?", str(s).strip())
        ]
    if subjects:
        subj_str = ", ".join(str(s) for s in subjects)
        visual.append(f"featuring {subj_str}" if visual else f"featuring {subj_str}")
    actions = _filter_labels(_trim(shot.get("actions")), task_type)
    if actions:
        visual.append(actions)
    camera = _trim(shot.get("camera"))
    if camera:
        camera = _normalize_camera(camera)
        if camera:
            visual.append(camera)

    body = " ".join(visual).strip()
    if body:
        parts.append(" " + body + ".")

    # 对话
    for d in shot.get("dialogue") or []:
        speaker = str(d.get("speaker", "S1"))
        text = str(d.get("text", ""))
        lang = str(d.get("language", "English"))
        delivery = _trim(d.get("delivery"))
        prefix = f"The {delivery} speaker {speaker}" if delivery else f"{speaker}"
        parts.append(
            f" {prefix} says: <d>[{lang}] {text}</d>."
        )

    # 转场：Guide 只在多镜头切换处写 "the camera cuts to"（下一镜头开头）；
    # 单镜头自然结束，无任何后缀。模板一律不渲染转场，避免自创后缀。
    return "".join(parts).strip()


def render_style_opening(style: str | None) -> str:
    return f"The target video is in a {style} style." if style else ""


# ---------------------------------------------------------------------------
# 确定性渲染：三字段（Base 模式）
# ---------------------------------------------------------------------------
def render_multimodal_description(
    ctx: PipelineContext, style: str | None = None
) -> str:
    """integrated_multimodal_description 主体。"""
    shots = ctx.shot_timeline.get("shots", [])
    descs = [render_shot(s, ctx.task_type) for s in shots]
    body = " ".join(descs)

    # Ref2VA：风格 1-2 句写在 [Shot 1] 之前；Base：风格写在 [Shot 1] 之后
    if ctx.task_type == "ref2va":
        if style:
            return f"{style} " + body
        return body
    if style:
        body = re.sub(r"^\[Shot 1\]", f"[Shot 1] {style}", body, count=1)
    return body


def render_overall_soundscape(ctx: PipelineContext) -> str:
    """overall_soundscape：1-4 句连续段落（环境声 + 物理动作声 + 非语言人声）。

    注意：Step 4 的 inferred_details.missing_sound 是给 Step 3 的补充建议
    （措辞如 "Add ... / Specify ..."），不能直接拼入最终输出；
    仅在 shot 规划完全没有声音事件时，才取其信息性内容兜底。
    """
    events: list[str] = []
    for s in ctx.shot_timeline.get("shots", []):
        for ev in s.get("sound_events") or []:
            d = _trim(ev.get("description")) or _trim(ev.get("type"))
            if d and d not in events:
                events.append(d)
    if events:
        # 自然段落式拼接（避免 "Ambient and physical sounds include ..." 列表式）
        head = events[0][0].upper() + events[0][1:]
        if len(events) > 1:
            mid = "; ".join(events[1:4])
            return (
                f"{head} sets the ambient base. In the foreground, {mid} "
                f"follow as the action unfolds."
            )
        return f"{head} fills the scene throughout."
    inferred = _trim(ctx.validation.get("inferred_details", {}).get("missing_sound"))
    if inferred:
        return inferred
    return "N/A"


def render_non_diegetic_music(ctx: PipelineContext) -> str:
    """non_diegetic_music：观众才能听到的背景音乐（1-3 句，无语调情绪词）。"""
    styles: list[str] = []
    for s in ctx.shot_timeline.get("shots", []):
        bgm = s.get("bgm") or {}
        # 只要存在风格描述即收集（active 缺失/为 false 时也保留，避免大量 N/A）
        style = _trim(bgm.get("style"))
        if style and style not in styles:
            styles.append(style)
    if styles:
        return "; ".join(styles) + "."
    return "N/A"


def render_base_instruction(task_type: str, ctx: PipelineContext) -> str:
    """Base 模式对齐指令行（I2VA / FL2VA / L2VA），T2VA 返回空。"""
    duration = ctx.duration or 0.0
    last_shot = max((s.get("shot_id", 1) for s in ctx.shot_timeline.get("shots", [])), default=1)
    if task_type == "i2va":
        return (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced."
        )
    if task_type == "fl2va":
        return (
            "How the reference pictures align with the target video — "
            "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; "
            f"Picture 2 (from Shot {last_shot}) aligns with the "
            f"{format_ts_secs(duration)}-second mark of the target video."
        )
    if task_type == "l2va":
        return (
            "How the reference pictures align with the target video — "
            f"<Picture 1> (from [Shot {last_shot}]) aligns with the "
            f"{format_ts_secs(duration)}-second mark of the target video."
        )
    return ""


def render_base_prompt(task_type: str, ctx: PipelineContext) -> str:
    """Base 模式三字段最终 prompt。"""
    style = (
        _clean_style(ctx.semantic_descriptions.get("prompt_analysis", {}).get("style"))
        or "cinematic live-action"
    )
    instruction = render_base_instruction(task_type, ctx)
    fields = [
        f"integrated_multimodal_description: {render_multimodal_description(ctx, style)}",
        f"overall_soundscape: {render_overall_soundscape(ctx)}",
        f"non_diegetic_music: {render_non_diegetic_music(ctx)}",
    ]
    if instruction:
        return instruction + "\n\n" + "\n\n".join(fields)
    return "\n\n".join(fields)


# ---------------------------------------------------------------------------
# 确定性渲染：六字段（Ref2VA）
# ---------------------------------------------------------------------------
def render_ref2va_prompt(ctx: PipelineContext) -> str:
    """Ref2VA 六字段最终 prompt。"""
    style = (
        _clean_style(ctx.semantic_descriptions.get("prompt_analysis", {}).get("style"))
        or "cinematic live-action"
    )
    detailed = render_multimodal_description(ctx, style)

    # summary 任务类型前缀（优先使用 Step 2/4 给出的任务类型提示）
    summary_parts: list[str] = []
    hint = (
        ctx.cross_modal_mapping.get("_task_type_hint")
        or ctx.validation.get("task_type_hint")
    )
    if hint:
        summary_parts.append(hint)
    if ctx.media_roles and any(v == "reference_audio" for v in ctx.media_roles.values()):
        summary_parts.append("audio reference")
    if ctx.media_roles and any(v in ("first_frame", "last_frame") for v in ctx.media_roles.values()):
        summary_parts.append("keyframe completion")
    if not summary_parts:
        summary_parts.append("reference generation")
    prefix = f"[{' + '.join(dict.fromkeys(summary_parts))}]"

    summary_body = _trim(ctx.validation.get("summary_hint")) or (
        "The target video synthesizes the referenced subjects into a new scene "
        "following the user's instruction."
    )

    sections = [
        "subject_definitions:\n" + "\n".join(ctx.subject_definitions),
        f"summary:\n{prefix} {summary_body}",
        "retention_analysis:\n" + _render_retention(ctx),
        f"detailed_description:\n{detailed}",
        f"overall_soundscape:\n{render_overall_soundscape(ctx)}",
        f"non_diegetic_music:\n{render_non_diegetic_music(ctx)}",
    ]
    return "\n\n".join(sections)


def _render_retention(ctx: PipelineContext) -> str:
    """retention_analysis：优先用 Step 4 输出，缺失时按词表兜底。"""
    ra = ctx.validation.get("retention_analysis") or []
    if ra:
        return "\n".join(str(x) for x in ra)
    lines = []
    for sd in ctx.subject_definitions:
        m = re.search(r"<(Subject|Audio) (\d+)>", sd)
        if not m:
            continue
        label = f"<{m.group(1)} {m.group(2)}>"
        marker = "reference" if m.group(1) == "Audio" else "fully_preserved"
        lines.append(f"{label}: {marker} - see subject_definitions for the referenced characteristics.")
    return "\n".join(lines) if lines else "N/A"


# ---------------------------------------------------------------------------
# Step 5 类
# ---------------------------------------------------------------------------
class Step5Formatter(StepBase):
    name = "step5"

    def __init__(self, settings: Settings | None = None):
        super().__init__(settings)
        provider, model = self.settings.step5_model
        self._client = TextClient(provider, model, self.settings)

    def run(self, ctx: PipelineContext) -> PipelineContext:
        # 1. 确定性模板渲染（最终约束 / 兜底，100% 符合官方结构）
        if ctx.task_type == "ref2va":
            fallback = render_ref2va_prompt(ctx)
        else:
            fallback = render_base_prompt(ctx.task_type, ctx)
        prompt = fallback

        # 2. LLM 按官方指南直接生成最终 prompt（llm_polish=true 时）
        if self.settings.llm_polish:
            template = _load_prompt(f"step5_format_{ctx.task_type}.txt")
            plan = {
                "task_type": ctx.task_type,
                "duration": ctx.duration,
                "ratio": ctx.ratio,
                "media_roles": ctx.media_roles,
                "prompt_analysis": ctx.semantic_descriptions.get("prompt_analysis", {}),
                "media_analysis": ctx.semantic_descriptions.get("media_analysis", {}),
                "subject_definitions": ctx.subject_definitions,
                "cross_modal_mapping": ctx.cross_modal_mapping,
                "shot_timeline": ctx.shot_timeline,
                "validation": ctx.validation,
            }
            user = (
                "Below is the structured plan for the target video (JSON). "
                "Write the FINAL video-generation prompt from this plan, "
                "following the official guide rules above. "
                "Output only the final prompt text, with no preamble or explanation.\n\n"
                "STRUCTURED PLAN (JSON):\n"
                f"{json.dumps(plan, ensure_ascii=False, indent=1)}"
            )
            try:
                generated = self._client.chat_text(
                    template,
                    user,
                    temperature=self.settings.temperature_step5,
                    max_tokens=self.settings.max_tokens_step,
                )
                if generated.strip():
                    prompt = generated.strip()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Step 5 LLM 生成失败，使用确定性模板兜底: %s", exc)

            # 3. 模板最终约束：LLM 输出不合规（缺字段/顺序错/时间戳非法/列表音效等）→ 回退模板
            report = validate_prompt(prompt, ctx.task_type, ctx.duration)
            if not report.ok:
                logger.warning(
                    "Step 5 LLM 输出不合规（%s），回退确定性模板",
                    report.summary()["issues"],
                )
                prompt = fallback

        # 4. 校验 + 自动修复兜底
        prompt = auto_fix(prompt, ctx.task_type, ctx.duration)
        report = validate_prompt(prompt, ctx.task_type, ctx.duration)
        ctx.final_prompt = prompt
        ctx.validation_report = report.summary()
        if not report.ok:
            logger.warning("Step 5 输出存在格式问题: %s", report.summary()["issues"])
        return ctx
