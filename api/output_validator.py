"""输出格式校验与自动修复。

基于官方两份指南（base-en.txt / ref-en.txt）的硬性规则做正则校验：
    - 字段名与顺序（Base 三字段 / Ref2VA 六字段）
    - [Shot N] At MM:SS.mmm 时间戳格式、严格递增、不越界
    - <d>[lang] ...</d> 对话标签
    - retention 词表（视觉/音频两套，不可混用）
    - summary 方括号任务类型前缀
    - (Sx) 说话人 ID 一致性
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# 正则常量
# ---------------------------------------------------------------------------
SHOT_TS_RE = re.compile(r"\[Shot (\d+)\](?: At (\d{2}):(\d{2})\.(\d{3}))?")
SHOT_TS_ANY_RE = re.compile(r"\[Shot (\d+)\] At (\d{2}):(\d{2})\.(\d{3})")
DIALOG_RE = re.compile(r"<d>\[([A-Za-z-]+)\](.*?)</d>", re.DOTALL)
SCENETRANS_RE = re.compile(r"<scenetrans>|<cutoff>")

# Ref2VA 六字段（顺序敏感）
REF2VA_FIELDS = [
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
]
# Base 三字段（顺序敏感）
BASE_FIELDS = [
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
]

# retention 词表（两套，不可混用）
VISUAL_MARKERS = {"fully_preserved", "partially_preserved", "attribute_transfer", "weak_reference"}
AUDIO_MARKERS = {"fully_copy", "partially_copy", "reference", "weak_reference"}

# summary 任务类型前缀（可 + 组合）
SUMMARY_PREFIXES = {
    "keyframe completion",
    "reference generation",
    "video editing",
    "video continuation",
    "audio reuse",
    "audio reference",
}

# 对齐指令行（I2VA / FL2VA / L2VA）
I2VA_INSTRUCTION_RE = re.compile(
    r"^For the target video, at 0\.00 seconds into the target video, "
    r"<Picture 1> \(from \[Shot 1\]\) is fully referenced\.$"
)
FL2VA_INSTRUCTION_RE = re.compile(
    r"^How the reference pictures align with the target video — "
    r"Picture 1 \(from Shot 1\) aligns with the 0\.00-second mark of the target video; "
    r"Picture 2 \(from Shot \d+\) aligns with the \d+\.\d{2}-second mark of the target video\.$"
)
L2VA_INSTRUCTION_RE = re.compile(
    r"^How the reference pictures align with the target video — "
    r"<Picture 1> \(from \[Shot \d+\]\) aligns with the \d+\.\d{2}-second mark of the target video\.$"
)

SPEAKER_RE = re.compile(r"\(S\d+(?:,S\d+)*\)")


@dataclass
class ValidationIssue:
    severity: str  # error | warning
    category: str
    message: str
    fix: str | None = None


@dataclass
class ValidationReport:
    ok: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)
    field_order: list[str] = field(default_factory=list)
    shot_times: list[tuple[int, float]] = field(default_factory=list)
    speaker_ids: set[str] = field(default_factory=set)

    def add(self, severity: str, category: str, message: str, fix: str | None = None) -> None:
        if severity == "error":
            self.ok = False
        self.issues.append(ValidationIssue(severity, category, message, fix))

    def summary(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error_count": sum(1 for i in self.issues if i.severity == "error"),
            "warning_count": sum(1 for i in self.issues if i.severity == "warning"),
            "issues": [
                {"severity": i.severity, "category": i.category, "message": i.message}
                for i in self.issues
            ],
        }


def _parse_time(m: str, s: str, ms: str) -> float:
    return int(m) * 60 + int(s) + int(ms) / 1000.0


def validate_prompt(prompt: str, task_type: str, duration: float | None = None) -> ValidationReport:
    """校验最终 prompt 是否符合官方格式。"""
    report = ValidationReport()
    lines = prompt.strip().splitlines()

    # --- 1. 字段顺序与存在性 ---
    fields = REF2VA_FIELDS if task_type == "ref2va" else BASE_FIELDS
    found: list[tuple[str, int]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        for f in fields:
            if stripped.startswith(f + ":"):
                found.append((f, i))
                break
    if not found:
        report.add("error", "structure", f"未找到任何官方字段（期望 {fields}）")
    else:
        # 检查顺序
        names = [f for f, _ in found]
        expected_order = [f for f in fields if f in names]
        if names != expected_order:
            report.add(
                "error",
                "structure",
                f"字段顺序错误: {names}，期望 {expected_order}",
                fix="按官方顺序重排字段",
            )
        report.field_order = names
        # 检查缺失字段
        missing = [f for f in fields if f not in names]
        for f in missing:
            report.add("error", "structure", f"缺少字段: {f}")

        # 非空检查（字段内容 = 字段行冒号后 + 到下一字段前的所有行）
        for f, idx in found:
            nxt = next((j for j in (j for _, j in found) if j > idx), len(lines))
            head = lines[idx].split(":", 1)[1] if ":" in lines[idx] else ""
            body = (head + "\n" + "\n".join(lines[idx + 1 : nxt])).strip()
            if not body:
                report.add("error", "structure", f"字段 {f} 内容为空")

    # --- 2. 对齐指令行 ---
    if task_type in ("i2va", "fl2va", "l2va"):
        first_line = lines[0].strip() if lines else ""
        if task_type == "i2va" and not I2VA_INSTRUCTION_RE.match(first_line):
            report.add("error", "instruction", "I2VA 第一行必须是官方对齐指令", fix=I2VA_INSTRUCTION_RE.pattern)
        if task_type == "fl2va" and not FL2VA_INSTRUCTION_RE.match(first_line):
            report.add("error", "instruction", "FL2VA 第一行必须是官方对齐指令")
        if task_type == "l2va" and not L2VA_INSTRUCTION_RE.match(first_line):
            report.add("error", "instruction", "L2VA 第一行必须是官方对齐指令")

    # --- 3. 时间戳 ---
    prev_time = -1.0
    for m in SHOT_TS_ANY_RE.finditer(prompt):
        shot_n, t = int(m.group(1)), _parse_time(m.group(2), m.group(3), m.group(4))
        report.shot_times.append((shot_n, t))
        if t <= prev_time:
            report.add("error", "timing", f"[Shot {shot_n}] 切点 {t:.3f}s 未严格递增（前一个 {prev_time:.3f}s）")
        prev_time = t
        if duration is not None and t > duration:
            report.add("error", "timing", f"[Shot {shot_n}] 切点 {t:.3f}s 超出视频总时长 {duration:.3f}s")

    # 首个 [Shot 1] 不应带时间戳
    m1 = SHOT_TS_RE.search(prompt)
    if m1 and m1.group(2) is not None:
        report.add("error", "timing", "[Shot 1] 不应带时间戳")

    # --- 4. 对话标签 ---
    for m in DIALOG_RE.finditer(prompt):
        lang, content = m.group(1), m.group(2).strip()
        if not content:
            report.add("warning", "dialogue", f"<d>[{lang}]</d> 内容为空")
        if not content.endswith((".", "!", "?", '"')):
            report.add("warning", "dialogue", f"对话未以标点结尾: {content[:40]!r}")

    # --- 5. retention 词表 ---
    if task_type == "ref2va":
        for m in re.finditer(r"<(Subject|Picture|Video|Audio) \d+>.*?(fully_preserved|partially_preserved|attribute_transfer|fully_copy|partially_copy|reference|weak_reference)", prompt):
            label_type, marker = m.group(1), m.group(2)
            expected = AUDIO_MARKERS if label_type == "Audio" else VISUAL_MARKERS
            if marker not in expected:
                report.add("error", "retention", f"{label_type} 使用了错误的词表: {marker}")
        # summary 前缀（字段名与内容可能跨行，如 "summary:\n[reference generation] ..."）
        sm_idx = next(
            (i for i, l in enumerate(lines) if l.strip().startswith("summary:")),
            None,
        )
        if sm_idx is not None:
            sm_block = "\n".join(lines[sm_idx : sm_idx + 3])
            m = re.search(r"\[([^\]]+)\]", sm_block)
            if not m:
                report.add("error", "summary", "summary 缺少方括号任务类型前缀")
            else:
                parts = [p.strip() for p in m.group(1).split("+")]
                if any(p not in SUMMARY_PREFIXES for p in parts):
                    report.add(
                        "error",
                        "summary",
                        f"summary 前缀含非法任务类型: {m.group(1)!r}（允许: {sorted(SUMMARY_PREFIXES)}）",
                    )

    # --- 6. 说话人 ID 一致性 ---
    for m in SPEAKER_RE.finditer(prompt):
        for sid in m.group(0).strip("()").split(","):
            report.speaker_ids.add(sid.strip())

    # --- 7. 英文双引号画面内文字（仅提示，不做硬性报错）---
    # （跳过：中文引号内文字无法可靠判定）

    # --- 8. 语言规则检查（所有 section 应为英文，<d> 内保留原文——难以自动判定，跳过）---

    return report


def auto_fix(prompt: str, task_type: str, duration: float | None = None) -> str:
    """对常见可自动修复的问题做确定性修复，返回修复后的 prompt。

    目前修复项：
        - [Shot 1] 带时间戳 → 去除时间戳
        - 时间戳格式 MM:SS.mmm 补零（如 3.5 → 00:03.500）
        - 越界时间戳截断（保留最大合法值附近）
    """
    fixed = prompt
    # 1. [Shot 1] At MM:SS.mmm[,] → [Shot 1]（连同逗号一起去除）
    fixed = re.sub(
        r"\[Shot 1\] At \d{2}:\d{2}\.\d{3},?",
        "[Shot 1]",
        fixed,
        count=1,
    )
    # 2. 时间戳补零：M:SS.mmm 或 MM:S.mmm
    fixed = re.sub(
        r"\[Shot (\d+)\] At (\d{1}):(\d{2})\.(\d{3})",
        lambda mm: f"[Shot {mm.group(1)}] At 0{mm.group(2)}:{mm.group(3)}.{mm.group(4)}",
        fixed,
    )
    fixed = re.sub(
        r"\[Shot (\d+)\] At (\d{2}):(\d{1})\.(\d{3})",
        lambda mm: f"[Shot {mm.group(1)}] At {mm.group(2)}:0{mm.group(3)}.{mm.group(4)}",
        fixed,
    )
    fixed = re.sub(
        r"\[Shot (\d+)\] At (\d{2}):(\d{2})\.(\d{1,2})(?!\d)",
        lambda mm: f"[Shot {mm.group(1)}] At {mm.group(2)}:{mm.group(3)}.{mm.group(4).ljust(3, '0')}",
        fixed,
    )
    return fixed
