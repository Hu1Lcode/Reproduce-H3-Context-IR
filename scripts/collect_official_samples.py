"""收集官方示例输入样本。

从 MiniMax-H3 GitHub 仓库的 scripts/readme/*.sh 提取公开测试素材 URL
（cdn.hailuoai.com 等），建立官方输入样本库（零成本 Ground Truth 素材）。

用法：
    python scripts/collect_official_samples.py \
        --repo-url https://ghfast.top/https://github.com/MiniMax-AI/MiniMax-H3 \
        --out ./examples
    # 离线模式：直接解析本地脚本目录
    python scripts/collect_official_samples.py --dir ./downloaded-scripts --out ./examples
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

# 允许直接从项目根运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

URL_RE = re.compile(r"https?://[^\s\"'\\]+")
SCRIPT_RE = re.compile(r"full-2k-(t2va|i2va|fl2va|l2va|ref2va)-h3-context-ir\.sh")


def _fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "h3c-collector"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def collect_from_scripts(scripts: list[Path], out_dir: Path) -> dict:
    """从脚本文件提取 content 数组与素材 URL。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    samples: dict = {}

    for script in scripts:
        m = SCRIPT_RE.search(script.name)
        task_type = m.group(1) if m else "unknown"
        text = script.read_text(encoding="utf-8", errors="ignore")

        # 提取 JSON 请求体（第一个 {...} 大括号块）
        match = re.search(r"\{.*content.*\}", text, re.DOTALL)
        request: dict = {}
        if match:
            try:
                request = json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # 提取所有素材 URL
        urls = list(dict.fromkeys(URL_RE.findall(text)))
        urls = [u for u in urls if any(k in u for k in ("hailuo", "cdn", ".mp4", ".wav", ".mp3", ".jpg", ".png"))]

        sample = {
            "task_type": task_type,
            "source_script": script.name,
            "request": request,
            "media_urls": urls,
        }
        key = script.stem
        samples[key] = sample
        (out_dir / task_type / f"{key}.json").write_text(
            json.dumps(sample, ensure_ascii=False, indent=2)
        )

    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps(samples, ensure_ascii=False, indent=2))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="收集官方 Context-IR 示例输入")
    parser.add_argument("--repo-url", type=str, default="https://ghfast.top/https://github.com/MiniMax-AI/MiniMax-H3", help="GitHub 仓库（可走镜像）")
    parser.add_argument("--dir", type=str, help="本地脚本目录（离线模式，跳过网络）")
    parser.add_argument("--out", type=str, default="./examples", help="输出目录")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dir:
        scripts = sorted(Path(args.dir).glob("*.sh"))
        if not scripts:
            print(f"目录 {args.dir} 中没有 .sh 文件")
            sys.exit(1)
        samples = collect_from_scripts(scripts, out_dir)
        print(f"离线模式：从 {len(scripts)} 个脚本提取 {len(samples)} 个样本 -> {out_dir}")
        return

    # 网络模式：下载仓库 zip（走镜像），仅取 scripts/readme/
    tmp = Path(tempfile.mkdtemp(prefix="h3c-official-"))
    try:
        zip_url = args.repo_url.rstrip("/") + "/archive/refs/heads/main.zip"
        print(f"下载仓库: {zip_url}")
        data = _fetch(zip_url)
        zip_path = tmp / "repo.zip"
        zip_path.write_bytes(data)
        shutil.unpack_archive(str(zip_path), tmp)
        scripts = sorted(tmp.glob("*/scripts/readme/*.sh"))
        if not scripts:
            print("未找到 scripts/readme/*.sh（可能镜像不完整），尝试直接抓取 raw 文件")
            base = args.repo_url.rstrip("/") + "/raw/main/scripts/readme/"
            scripts = []
            for name in ("full-2k-t2va-h3-context-ir.sh", "full-2k-i2va-h3-context-ir.sh", "full-2k-ref2va-h3-context-ir.sh"):
                try:
                    data = _fetch(base + name)
                    p = tmp / name
                    p.write_bytes(data)
                    scripts.append(p)
                except Exception as exc:
                    print(f"  抓取 {name} 失败: {exc}")
        samples = collect_from_scripts(scripts, out_dir)
        print(f"网络模式：收集 {len(samples)} 个样本 -> {out_dir}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
