"""批量调用 H3-Context-IR 服务，对 prompt.txt 中所有 prompt 做增强并保存结果。

输入: /home/wjh/Reproduce-H3-Context-IR/prompt.txt
      每行格式: 序号,Prompt内容
输出: /home/wjh/Reproduce-H3-Context-IR/enhanced_prompts.txt
      每行格式: 序号,增强后的Prompt
支持断点续跑：已成功处理的序号自动跳过。
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8888")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = Path(os.environ.get("INPUT_FILE", str(PROJECT_ROOT / "prompt.txt")))
OUTPUT_FILE = Path(os.environ.get("OUTPUT_FILE", str(PROJECT_ROOT / "enhanced_prompts.txt")))
ENHANCE_ONLY = os.environ.get("ENHANCE_ONLY", "0") == "1"  # =1 时只增强不生成视频
POLL_INTERVAL = 10
MAX_WAIT = 1800  # 单条最长等待 30 分钟
MAX_CONCURRENT_VIDEO = 3  # 视频生成最大并发数

# 已启动的视频生成进程列表
_video_procs: list[subprocess.Popen] = []

# 并发增强配置
CONCURRENCY = int(os.environ.get("CONCURRENCY", "10"))
_write_lock = threading.Lock()  # 文件写入互斥锁
_video_lock = threading.Lock()  # 视频触发互斥锁

def read_prompts(path: Path) -> list[tuple[int, str]]:
    """读取 prompt.txt，返回 [(序号, prompt), ...]。"""
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        idx_str, _, prompt = line.partition(",")
        try:
            idx = int(idx_str)
        except ValueError:
            print(f"  [WARN] 无法解析行，跳过: {line[:50]}...")
            continue
        prompt = prompt.strip()
        if prompt:
            prompts.append((idx, prompt))
    return prompts


def load_done(path: Path) -> dict[int, str]:
    """读取已完成的输出，返回 {序号: 增强后prompt}。"""
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            idx_str, _, prompt = line.partition(",")
            try:
                done[int(idx_str)] = prompt
            except ValueError:
                continue
    return done


def call_service(prompt: str) -> str:
    """调用服务，返回增强后的 prompt；失败抛异常。"""
    # 1. 创建任务
    resp = requests.post(
        f"{API_BASE}/v2/h3_context_ir",
        json={
            "model": "MiniMax-H3",
            "content": [{"type": "text", "text": prompt}],
            "duration": 5,
            "ratio": "16:9",
        },
        timeout=60,
    )
    resp.raise_for_status()
    task_id = resp.json()["task_id"]
    print(f"    创建任务: {task_id}")

    # 2. 轮询
    start = time.time()
    while time.time() - start < MAX_WAIT:
        time.sleep(POLL_INTERVAL)
        query = requests.get(f"{API_BASE}/v2/query/video_generation/{task_id}", timeout=60)
        query.raise_for_status()
        data = query.json()
        status = data["status"]
        print(f"    status={status} ({int(time.time() - start)}s)")
        if status == "success":
            content = data.get("content") or {}
            prompt_out = content.get("prompt", "")
            if not prompt_out:
                raise RuntimeError(f"任务 {task_id} 成功但无 prompt 内容")
            return prompt_out
        if status == "failed":
            raise RuntimeError(f"任务 {task_id} 失败: {data.get('error')}")
    raise RuntimeError(f"任务 {task_id} 超时")


def _file_has_idx(idx: int) -> bool:
    """检查输出文件中是否已存在指定序号（只匹配以 "序号," 开头的条目行）。"""
    if not OUTPUT_FILE.exists():
        return False
    for line in OUTPUT_FILE.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(\d+),", line)
        if m and int(m.group(1)) == idx:
            return True
    return False


def _trigger_video_gen(idx: int) -> None:
    """异步启动视频生成（不阻塞主循环），限制最大并发数。"""
    global _video_procs
    # 清理已结束的进程
    _video_procs = [p for p in _video_procs if p.poll() is None]

    # 等待并发数降至上限以下
    while len(_video_procs) >= MAX_CONCURRENT_VIDEO:
        time.sleep(3)
        _video_procs = [p for p in _video_procs if p.poll() is None]

    script = str(PROJECT_ROOT / "scripts/minimax-h3-t2va.sh")
    out_dir = str(PROJECT_ROOT / "outputs" / "t2va_v2")
    trigger_env = os.environ.copy()
    trigger_env["OUTPUT_DIR"] = out_dir
    trigger_env["SKIP_EXISTING"] = "0"
    proc = subprocess.Popen(
        ["bash", script, str(idx)],
        env=trigger_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _video_procs.append(proc)
    print(f"    [触发视频生成] 序号 {idx}（后台进程 vid={proc.pid}，当前并发 {len(_video_procs)}）")


def process_one(idx: int, prompt: str) -> tuple[int, bool, str]:
    """处理单条 prompt（可被多线程并发调用）。
    返回 (idx, 是否成功, 错误消息)。
    """
    print(f"[{idx}] 处理中: {prompt[:60]}...")
    try:
        enhanced = call_service(prompt)
    except Exception as exc:  # noqa: BLE001
        print(f"[{idx}] 失败: {exc}")
        return idx, False, str(exc)

    # 线程安全写入文件
    with _write_lock:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            if not _file_has_idx(idx):
                f.write(f"{idx},{enhanced}\n")
            else:
                print(f"  [WARN] idx={idx} 已存在于文件，跳过写入（重复条目防护）")
    print(f"[{idx}] 完成，已保存")

    # 线程安全触发视频
    if not ENHANCE_ONLY:
        with _video_lock:
            _trigger_video_gen(idx)

    return idx, True, ""


def main() -> int:
    prompts = read_prompts(INPUT_FILE)
    print(f"共读取 {len(prompts)} 条 prompt，并发数={CONCURRENCY}")

    done = load_done(OUTPUT_FILE)
    print(f"已完成 {len(done)} 条，将跳过")

    # 筛选待处理的
    todo = [(idx, p) for idx, p in prompts if idx not in done]
    print(f"待处理 {len(todo)} 条")

    failed: list[tuple[int, str]] = []

    if CONCURRENCY <= 1:
        # 串行模式
        for idx, prompt in todo:
            _, ok, err = process_one(idx, prompt)
            if not ok:
                failed.append((idx, err))
    else:
        # 并发模式
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = {executor.submit(process_one, idx, p): idx for idx, p in todo}
            for future in as_completed(futures):
                idx, ok, err = future.result()
                if not ok:
                    failed.append((idx, err))
                    print(f"  [{idx}] 加入失败列表")

    print("\n================ 汇总 ================")
    print(f"成功: {len(done) + len(todo) - len(failed)} 条 → {OUTPUT_FILE}")

    # 等待所有视频生成进程完成（ENHANCE_ONLY=1 时无视频进程）
    if not ENHANCE_ONLY:
        active = [p for p in _video_procs if p.poll() is None]
        if active:
            print(f"等待 {len(active)} 个视频生成进程完成...")
            for p in active:
                p.wait()
            print("所有视频生成完成")

    if failed:
        print(f"失败: {len(failed)} 条")
        for idx, err in failed:
            print(f"  [{idx}] {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
