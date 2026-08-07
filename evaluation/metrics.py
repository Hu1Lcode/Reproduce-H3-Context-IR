"""硬性评估指标（复现方案第 7 节）。

指标：
    1. 格式合规率     —— validate_prompt 无 error
    2. retention 词表合法率 —— Ref2VA 中出现的 marker 全部属于两套词表
    3. (Sx) 一致性    —— 说话人 ID 集合稳定（跨镜头不出现孤立新 ID 无定义）
    4. 时间戳越界率   —— 切点超出视频时长的比例
    5. 字段完整性     —— 期望字段全部存在
    6. 与官方输出相似度 —— ROUGE-L（可选依赖）或简单 n-gram 重叠
"""
from __future__ import annotations

import re
from typing import Any

from api.output_validator import (
    AUDIO_MARKERS,
    BASE_FIELDS,
    DIALOG_RE,
    REF2VA_FIELDS,
    SHOT_TS_ANY_RE,
    VISUAL_MARKERS,
    validate_prompt,
)

RETENTION_RE = re.compile(
    r"<(Subject|Picture|Video|Audio) \d+>[^:]*:"
    r"(fully_preserved|partially_preserved|attribute_transfer|fully_copy|partially_copy|reference|weak_reference)"
)


def evaluate_prompt(prompt: str, task_type: str, duration: float | None = None) -> dict[str, Any]:
    """对单个最终 prompt 计算全部硬性指标。"""
    report = validate_prompt(prompt, task_type, duration)
    errors = [i for i in report.issues if i.severity == "error"]

    # retention 词表
    markers = RETENTION_RE.findall(prompt)
    valid = 0
    for label, marker in markers:
        allowed = AUDIO_MARKERS if label == "Audio" else VISUAL_MARKERS
        if marker in allowed:
            valid += 1
    retention_rate = valid / len(markers) if markers else 1.0

    # 时间戳越界
    shots = SHOT_TS_ANY_RE.findall(prompt)
    out_of_bounds = 0
    for _, m, s, ms in shots:
        t = int(m) * 60 + int(s) + int(ms) / 1000.0
        if duration is not None and t > duration:
            out_of_bounds += 1
    oob_rate = out_of_bounds / len(shots) if shots else 0.0

    # 字段完整性
    fields = REF2VA_FIELDS if task_type == "ref2va" else BASE_FIELDS
    present = [f for f in report.field_order if f in fields]
    field_completeness = len(present) / len(fields)

    # 对话标签平衡（<d> 与 </d> 数量一致）
    dialog_blocks = DIALOG_RE.findall(prompt)
    dialog_ok = len(dialog_blocks) == prompt.count("<d>") == prompt.count("</d>") or (
        prompt.count("<d>") == 0 and prompt.count("</d>") == 0
    )

    return {
        "format_compliant": report.ok,
        "format_error_count": len(errors),
        "retention_vocab_rate": retention_rate,
        "timestamp_oob_rate": oob_rate,
        "field_completeness": field_completeness,
        "dialog_balanced": dialog_ok,
        "shot_count": len(report.shot_times),
        "speaker_ids": sorted(report.speaker_ids),
        "issues": [i.message for i in report.issues],
    }


def ngram_overlap(a: str, b: str, n: int = 3) -> float:
    """n-gram 重叠相似度（0-1），无需额外依赖。"""
    a = re.sub(r"\s+", " ", a.lower()).split()
    b = re.sub(r"\s+", " ", b.lower()).split()
    if len(a) < n or len(b) < n:
        return 0.0
    sa = {" ".join(a[i : i + n]) for i in range(len(a) - n + 1)}
    sb = {" ".join(b[i : i + n]) for i in range(len(b) - n + 1)}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa), len(sb))


def rouge_l_f1(a: str, b: str) -> float:
    """ROUGE-L F1（最长公共子序列），可选依赖不满足时退化到 ngram。"""
    try:
        from rouge_score.rouge_scorer import RougeScorer  # type: ignore

        scorer = RougeScorer(["rougeL"], use_stemmer=True)
        return scorer.score(a, b)["rougeL"].fmeasure
    except Exception:
        return ngram_overlap(a, b, 3)


def evaluate_batch(
    results: list[dict[str, Any]],
    task_type: str,
    duration: float | None = None,
) -> dict[str, Any]:
    """批量评估（results 为 [{prompt, ...}] 列表），输出聚合指标。"""
    per = [evaluate_prompt(r["prompt"], task_type, duration) for r in results]
    n = max(len(per), 1)
    return {
        "sample_count": len(per),
        "avg_format_compliant_rate": sum(1 for p in per if p["format_compliant"]) / n,
        "avg_retention_vocab_rate": sum(p["retention_vocab_rate"] for p in per) / n,
        "avg_timestamp_oob_rate": sum(p["timestamp_oob_rate"] for p in per) / n,
        "avg_field_completeness": sum(p["field_completeness"] for p in per) / n,
        "dialog_balanced_rate": sum(1 for p in per if p["dialog_balanced"]) / n,
    }
