"""官方 vs 自建输出对比。

用法：
    python -m evaluation.compare --ours ./work/output.json --official ./official_result.json

两个文件均为 dict，含 "final_prompt"（自建）与 "prompt"（官方，位于
.task.content.prompt）。输出结构化对比报告。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import ngram_overlap, rouge_l_f1  # noqa: E402


def extract_prompt(data: dict, ours: bool = True) -> str | None:
    """从结果 dict 中提取 prompt 文本。"""
    if "final_prompt" in data:
        return data["final_prompt"]
    if "prompt" in data:
        return data["prompt"]
    task = data.get("task") or {}
    content = task.get("content") or {}
    if isinstance(content, dict) and content.get("prompt"):
        return content["prompt"]
    return None


def compare(ours_prompt: str, official_prompt: str) -> dict:
    return {
        "ours_length": len(ours_prompt.split()),
        "official_length": len(official_prompt.split()),
        "ngram3_overlap": round(ngram_overlap(ours_prompt, official_prompt, 3), 4),
        "rouge_l_f1": round(rouge_l_f1(ours_prompt, official_prompt), 4),
        "field_names_ours": sorted(set(official_prompt.split(":")[0] for _ in [0]) or []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="官方 vs 自建 Context-IR 输出对比")
    parser.add_argument("--ours", required=True, help="自建流水线输出 JSON")
    parser.add_argument("--official", required=True, help="官方 API 输出 JSON")
    args = parser.parse_args()

    ours = json.loads(Path(args.ours).read_text(encoding="utf-8"))
    official = json.loads(Path(args.official).read_text(encoding="utf-8"))

    ours_p = extract_prompt(ours, ours=True)
    off_p = extract_prompt(official, ours=False)
    if not ours_p or not off_p:
        print("错误：无法从输入文件中提取 prompt")
        sys.exit(1)

    report = compare(ours_p, off_p)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
