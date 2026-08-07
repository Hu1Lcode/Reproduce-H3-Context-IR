"""端到端运行 H3-Context-IR 复现流水线。

用法示例：
    # 纯文本 T2VA
    python scripts/run_pipeline.py --prompt "一只猫在窗台上看雨" --duration 6

    # I2VA（首帧生视频）
    python scripts/run_pipeline.py --prompt "..." --image ./first.jpg --role first_frame

    # Ref2VA（全参考模式）
    python scripts/run_pipeline.py --prompt "..." \
        --video ./ref.mp4 --video-role reference_video \
        --audio ./voice.wav --audio-role reference_audio \
        --duration 5 --out ./work/result.json

    # 官方 content JSON 输入（与官方 API 请求格式一致）
    python scripts/run_pipeline.py --content ./work/input.json --out ./work/result.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# 允许直接从项目根运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.output_validator import validate_prompt  # noqa: E402
from pipeline.base_pipeline import H3ContextPipeline  # noqa: E402
from pipeline.context import PipelineContext  # noqa: E402
from preprocessor.task_classifier import classify_task  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("h3c.cli")


def build_content(args: argparse.Namespace) -> list[dict]:
    """从 CLI 参数构造官方 content 数组。"""
    if args.content:
        data = json.loads(Path(args.content).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data["content"]
        return data

    content: list[dict] = [{"type": "text", "text": args.prompt}]

    def add_media(typ: str, paths: list[str] | None, role: str | None) -> None:
        for i, p in enumerate(paths or [], start=1):
            item: dict = {"type": typ, "url": p, "asset_id": f"{typ.title()} {i}"}
            if role:
                item["role"] = role
            content.append(item)

    add_media("image_url", args.image, args.image_role)
    add_media("video_url", args.video, args.video_role)
    add_media("audio_url", args.audio, args.audio_role)
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description="H3-Context-IR 复现流水线")
    parser.add_argument("--prompt", type=str, default="", help="用户文本指令")
    parser.add_argument("--image", nargs="*", help="参考图像（URL 或本地路径）")
    parser.add_argument("--image-role", default="reference", help="图像 role（first_frame/last_frame/reference）")
    parser.add_argument("--video", nargs="*", help="参考视频")
    parser.add_argument("--video-role", default="reference_video", help="视频 role")
    parser.add_argument("--audio", nargs="*", help="参考音频")
    parser.add_argument("--audio-role", default="reference_audio", help="音频 role")
    parser.add_argument("--content", type=str, help="官方 content JSON 文件（优先于 --prompt 参数）")
    parser.add_argument("--duration", type=float, default=10.0, help="目标视频时长（秒）")
    parser.add_argument("--ratio", type=str, default="16:9", help="目标宽高比")
    parser.add_argument("--out", type=str, default="./work/output.json", help="结果输出路径")
    parser.add_argument("--stages", nargs="*", default=None, help="仅执行指定阶段（如 step1 step2）")
    parser.add_argument("--no-cache", action="store_true", help="禁用缓存")
    parser.add_argument("--skip-errors", action="store_true", help="阶段失败继续")
    args = parser.parse_args()

    from config.settings import settings

    if args.no_cache:
        settings.enable_cache = False

    content = build_content(args)
    raw_input = {
        "content": content,
        "duration": args.duration,
        "ratio": args.ratio,
        "model": "MiniMax-H3",
    }

    ctx = PipelineContext(raw_input=raw_input)
    classification = classify_task(content)
    ctx.task_type = classification.task_type
    ctx.media_roles = classification.media_roles
    logger.info("任务类型: %s", ctx.task_type)

    pipeline = H3ContextPipeline(settings=settings, skip_errors=args.skip_errors)
    ctx = pipeline.run(ctx, stages=args.stages)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ctx.to_dict(), ensure_ascii=False, indent=2))
    print(f"\n===== 结果已保存: {out_path} =====\n")

    if ctx.final_prompt:
        print("----- 最终 Prompt（Step 5 输出）-----")
        print(ctx.final_prompt)
        report = validate_prompt(ctx.final_prompt, ctx.task_type, ctx.duration)
        print(f"\n----- 格式校验: {'OK' if report.ok else '存在问题'} -----")
        for i in report.issues:
            print(f"  [{i.severity}] {i.message}")


if __name__ == "__main__":
    main()
