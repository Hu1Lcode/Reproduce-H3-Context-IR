"""核心模块冒烟测试：task_classifier / input_validator / step5 渲染 / output_validator。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessor.task_classifier import classify_task, TaskClassifierError
from preprocessor.input_validator import validate_input
from pipeline.context import PipelineContext
from pipeline.step5_formatter import (
    render_base_prompt,
    render_ref2va_prompt,
    format_ts,
    format_ts_secs,
)
from api.output_validator import validate_prompt, auto_fix

import re as _re


def _field_order(prompt: str):
    return _re.findall(
        r"^(integrated_multimodal_description|subject_definitions|summary|retention_analysis|detailed_description|overall_soundscape|non_diegetic_music):",
        prompt,
        _re.M,
    )


PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


# ---------------------------------------------------------------------------
# 1. Step 0 任务区分
# ---------------------------------------------------------------------------
print("== task_classifier ==")
check(
    "T2VA", classify_task([{"type": "text", "text": "hi"}]).task_type == "t2va"
)
check(
    "I2VA",
    classify_task(
        [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "url": "x", "role": "first_frame", "asset_id": "P1"},
        ]
    ).task_type == "i2va",
)
check(
    "L2VA",
    classify_task(
        [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "url": "x", "role": "last_frame", "asset_id": "P1"},
        ]
    ).task_type == "l2va",
)
check(
    "FL2VA",
    classify_task(
        [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "url": "x", "role": "first_frame"},
            {"type": "image_url", "url": "y", "role": "last_frame"},
        ]
    ).task_type == "fl2va",
)
check(
    "Ref2VA",
    classify_task(
        [
            {"type": "text", "text": "hi"},
            {"type": "video_url", "url": "v", "role": "reference_video"},
            {"type": "audio_url", "url": "a", "role": "reference_audio"},
        ]
    ).task_type == "ref2va",
)
try:
    classify_task([{"type": "image_url", "url": "x"}])
    check("无 text 报错", False)
except TaskClassifierError:
    check("无 text 报错", True)

# ---------------------------------------------------------------------------
# 2. 输入规格校验
# ---------------------------------------------------------------------------
print("== input_validator ==")
ok = validate_input(
    [
        {"type": "text", "text": "hi"},
        {"type": "video_url", "role": "reference_video", "duration": 5},
        {"type": "audio_url", "role": "reference_audio", "duration": 3},
    ]
)
check("合法 Ref2VA 输入通过", ok.ok, str(ok.errors))

bad = validate_input(
    [
        {"type": "text", "text": "hi"},
        {"type": "audio_url", "role": "reference_audio", "duration": 3},
    ]
)
check("音频无图像/视频被拒", "音频" in "".join(bad.errors), str(bad.errors))

bad2 = validate_input(
    [{"type": "text", "text": "hi"}]
    + [{"type": "image_url", "url": f"i{i}"} for i in range(10)]
)
check("图像超 9 张被拒", any("图像" in e for e in bad2.errors), str(bad2.errors))

# ---------------------------------------------------------------------------
# 3. Step 5 确定性渲染
# ---------------------------------------------------------------------------
print("== step5 formatter ==")


def make_ctx(task_type: str, duration: float = 8.0) -> PipelineContext:
    ctx = PipelineContext(
        raw_input={"content": [], "duration": duration, "ratio": "16:9"},
        task_type=task_type,
        media_roles={},
        semantic_descriptions={
            "prompt_analysis": {"style": "live-action, cinematic", "task_params": {"duration": duration}}
        },
        shot_timeline={
            "total_duration": duration,
            "shots": [
                {
                    "shot_id": 1,
                    "start_time": 0.0,
                    "end_time": 4.0,
                    "camera": "the camera pushes in with small amplitude at slow speed",
                    "composition": "a medium-wide shot frames a woman beside a window",
                    "subjects": ["Subject 1"],
                    "actions": "she lifts her gaze from a folded letter",
                    "dialogue": [
                        {"speaker": "S1", "text": "I get off at the next station.", "language": "English", "time": 2.0}
                    ],
                    "bgm": {"active": True, "style": "soft piano at a slow tempo", "fade": "in"},
                    "sound_events": [{"type": "train_rattle", "time": 1.0, "description": "steady metallic train rhythm"}],
                    "transition": "cut",
                },
                {
                    "shot_id": 2,
                    "start_time": 4.0,
                    "end_time": 8.0,
                    "camera": "the camera trucks right with small amplitude",
                    "composition": "a close-up of her reflection in the glass",
                    "subjects": ["Subject 1"],
                    "actions": "she folds the letter along its crease",
                    "dialogue": [],
                    "bgm": {"active": True, "style": "soft piano at a slow tempo", "fade": "out"},
                    "sound_events": [],
                    "transition": "cut",
                },
            ],
        },
        subject_definitions=["<Subject 1> is the young woman in <Picture 1>, with long dark hair."],
        validation={
            "retention_analysis": ["<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - retained throughout."]
        },
    )
    return ctx


# T2VA
p_t2va = render_base_prompt("t2va", make_ctx("t2va"))
check("T2VA 无指令行", not p_t2va.startswith("For the target video"), p_t2va[:60])
check("T2VA 三字段顺序", _field_order(p_t2va) == ["integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"], p_t2va[:80])

# I2VA
p_i2va = render_base_prompt("i2va", make_ctx("i2va"))
check("I2VA 指令行", p_i2va.startswith("For the target video, at 0.00 seconds"), p_i2va[:80])

# FL2VA
p_fl2va = render_base_prompt("fl2va", make_ctx("fl2va", 8.0))
check("FL2VA 指令行含 Picture 2 与时长", "Picture 2 (from Shot 2) aligns with the 8.00-second mark" in p_fl2va, p_fl2va[:120])

# L2VA
p_l2va = render_base_prompt("l2va", make_ctx("l2va", 8.0))
check("L2VA 指令行", "<Picture 1> (from [Shot 2]) aligns with the 8.00-second mark" in p_l2va, p_l2va[:120])

# Ref2VA 六字段
p_ref = render_ref2va_prompt(make_ctx("ref2va"))
order = _field_order(p_ref)
check("Ref2VA 六字段顺序", order == ["subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape", "non_diegetic_music"], str(order))
check("Ref2VA summary 前缀", "[reference generation]" in p_ref or "[reference generation + audio reference]" in p_ref, p_ref[:200])

# 时间戳工具
check("format_ts", format_ts(3.5) == "00:03.500", format_ts(3.5))
check("format_ts_secs", format_ts_secs(8.0) == "8.00", format_ts_secs(8.0))

# ---------------------------------------------------------------------------
# 4. output_validator（用渲染结果自校验 + 官方示例校验）
# ---------------------------------------------------------------------------
print("== output_validator ==")
for name, p, task in [
    ("T2VA", p_t2va, "t2va"),
    ("I2VA", p_i2va, "i2va"),
    ("FL2VA", p_fl2va, "fl2va"),
    ("L2VA", p_l2va, "l2va"),
    ("Ref2VA", p_ref, "ref2va"),
]:
    r = validate_prompt(p, task, 8.0)
    check(f"{name} 渲染结果合规", r.ok, str([i.message for i in r.issues]))

# 官方示例（来自 base-en.txt Case 2）应通过校验
official_i2va = """For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, the young woman shown in <Picture 1> remains beside the rain-covered train window. The quiet, breathy young woman (S1) says: <d>[English] I get off at the next station.</d> [Shot 2] At 00:05.000, the camera cuts to a close-up of her hands folding the letter.

overall_soundscape: The train wheels produce a steady metallic rhythm beneath a low ventilation hum. Rain ticks against the window while paper rustles softly in her hands.

non_diegetic_music: Sustained cello notes at a slow tempo with widely spaced piano tones, gradually decreasing in volume."""
r = validate_prompt(official_i2va, "i2va", 8.0)
check("官方 I2VA 示例通过校验", r.ok, str([i.message for i in r.issues]))

# 构造错误 prompt 应报错
bad_prompt = "integrated_multimodal_description: [Shot 1] hello.\n[Shot 2] At 00:03.500, cut.\n"
bad_prompt += "non_diegetic_music: N/A\noverall_soundscape: N/A\n"  # 顺序错误
r = validate_prompt(bad_prompt, "t2va", 10.0)
check("字段顺序错误被检测", any("顺序" in i.message for i in r.issues), str([i.message for i in r.issues]))

# auto_fix：[Shot 1] 时间戳去除
fixed = auto_fix("[Shot 1] At 00:00.000, start.\n[Shot 2] At 00:03.5, next.", "t2va", 10.0)
check("auto_fix 去除 [Shot 1] 时间戳", fixed.startswith("[Shot 1] start."), fixed)
check("auto_fix 时间戳补零", "At 00:03.500" in fixed, fixed)

# ---------------------------------------------------------------------------
print(f"\n===== 结果: {PASS} 通过 / {FAIL} 失败 =====")
sys.exit(1 if FAIL else 0)
