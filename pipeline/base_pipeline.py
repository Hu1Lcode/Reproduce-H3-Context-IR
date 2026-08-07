"""流水线基类：阶段编排、重试、缓存。

每个 Step 实现类继承 StepBase，重写 run(ctx)。H3ContextPipeline 按顺序
执行 Step 0（本地规则）→ 预处理 → Step 1 → ... → Step 5，并支持：
    - 磁盘缓存（按阶段输入 hash 缓存中间结果，调试/省钱）
    - 阶段失败隔离（--skip-errors 下记录错误继续）
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from config.settings import Settings, settings as _settings
from pipeline.context import PipelineContext

logger = logging.getLogger("h3c.pipeline")


class StepError(RuntimeError):
    """单步执行失败。"""


class StepBase:
    """单步基类。"""

    name: str = "step"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or _settings

    def run(self, ctx: PipelineContext) -> PipelineContext:
        """执行本步骤，返回（可能被修改的）ctx。"""
        raise NotImplementedError


class H3ContextPipeline:
    """6 步流水线编排器。"""

    def __init__(
        self,
        steps: list[StepBase] | None = None,
        settings: Settings | None = None,
        skip_errors: bool = False,
    ):
        self.settings = settings or _settings
        self.skip_errors = skip_errors
        self._steps: list[StepBase] = steps if steps is not None else _default_steps(self.settings)

    def add_step(self, step: StepBase) -> None:
        self._steps.append(step)

    def run(self, ctx: PipelineContext, stages: list[str] | None = None) -> PipelineContext:
        """按顺序执行全部步骤。

        Args:
            ctx: 初始上下文（需含 raw_input）。
            stages: 可选子集，如 ["step1", "step2"]；None 表示全部。
        """
        for step in self._steps:
            if stages is not None and step.name not in stages:
                continue
            try:
                if self.settings.enable_cache and step.name != "step5":
                    self._run_cached(step, ctx)
                else:
                    step.run(ctx)
                ctx.log_step(step.name, "ok")
                logger.info("阶段 %s 完成", step.name)
            except Exception as exc:  # noqa: BLE001
                ctx.log_step(step.name, "error", str(exc))
                if not self.skip_errors:
                    raise StepError(f"阶段 {step.name} 失败: {exc}") from exc
                logger.error("阶段 %s 失败（继续）: %s", step.name, exc)
        return ctx

    # ------------------------------------------------------------------
    def _run_cached(self, step: StepBase, ctx: PipelineContext) -> None:
        """带磁盘缓存的步骤执行。"""
        cache_dir = self.settings.cache_dir / step.name
        cache_dir.mkdir(parents=True, exist_ok=True)

        key = self._cache_key(step.name, ctx)
        cache_file = cache_dir / f"{key}.json"
        if cache_file.exists():
            cached = json.loads(cache_file.read_text())
            _restore_step_output(step, ctx, cached)
            logger.info("阶段 %s 命中缓存", step.name)
            return

        step.run(ctx)
        payload = _capture_step_output(step, ctx)
        cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=1))

    @staticmethod
    def _cache_key(step: str, ctx: PipelineContext) -> str:
        """阶段缓存键：输入哈希。"""
        if step == "step1":
            blob = json.dumps(
                {"raw": ctx.raw_input, "pre": ctx.preprocessed}, ensure_ascii=False
            )
        elif step == "step2":
            blob = json.dumps({"task": ctx.task_type, "sem": ctx.semantic_descriptions}, ensure_ascii=False)
        elif step == "step3":
            blob = json.dumps(
                {"task": ctx.task_type, "sem": ctx.semantic_descriptions, "cm": ctx.cross_modal_mapping},
                ensure_ascii=False,
            )
        elif step == "step4":
            blob = json.dumps(
                {"task": ctx.task_type, "shot": ctx.shot_timeline, "subj": ctx.subject_definitions},
                ensure_ascii=False,
            )
        else:
            blob = json.dumps(ctx.to_dict(), ensure_ascii=False)
        return hashlib.md5(blob.encode()).hexdigest()[:16]


def _capture_step_output(step: StepBase, ctx: PipelineContext) -> dict[str, Any]:
    """捕获步骤写入 ctx 的字段（按步骤名）。"""
    field_map = {
        "preprocess": "preprocessed",
        "step1": "semantic_descriptions",
        "step2": "cross_modal_mapping",
        "step3": "shot_timeline",
        "step4": "validation",
    }
    key = field_map.get(step.name)
    return {key: getattr(ctx, key, {})} if key else {}


def _restore_step_output(step: StepBase, ctx: PipelineContext, payload: dict[str, Any]) -> None:
    for k, v in payload.items():
        setattr(ctx, k, v)


def _default_steps(settings: Settings) -> list[StepBase]:
    """默认 6 步实例化。"""
    from pipeline.step1_extractor import Step1Extractor
    from pipeline.step2_associator import Step2Associator
    from pipeline.step3_shot_planner import Step3ShotPlanner
    from pipeline.step4_validator import Step4Validator
    from pipeline.step5_formatter import Step5Formatter

    return [
        Step1Extractor(settings),
        Step2Associator(settings),
        Step3ShotPlanner(settings),
        Step4Validator(settings),
        Step5Formatter(settings),
    ]
