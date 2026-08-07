"""视频帧采样（~1fps 关键帧序列）。

使用 OpenCV（容器内已装 opencv-python-headless）。
输出：关键帧 PNG 文件列表 + 每帧对应时间戳，供 Step 1 视觉理解使用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass
class SampledFrame:
    index: int
    time: float  # 秒
    path: Path  # 帧图像文件路径

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "time": self.time, "path": str(self.path)}


@dataclass
class SamplingResult:
    video_path: Path
    duration: float
    fps: float
    frames: list[SampledFrame] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_path": str(self.video_path),
            "duration": self.duration,
            "fps": self.fps,
            "frames": [f.to_dict() for f in self.frames],
        }


def sample_video_frames(
    video_path: str | Path,
    out_dir: str | Path,
    fps: float = 1.0,
    max_frames: int = 60,
) -> SamplingResult:
    """按 ~fps 帧/秒采样视频关键帧。

    Args:
        video_path: 输入视频文件。
        out_dir: 输出目录（自动创建）。
        fps: 采样帧率。
        max_frames: 单视频最大采样帧数（防御性上限）。

    Returns:
        SamplingResult（含每帧路径与时间戳）。
    """
    if cv2 is None:
        raise ImportError("需要 opencv-python-headless，请先安装 requirements.txt")

    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / video_fps if frame_count else 0.0

        # 计算采样步长（帧数）
        step = max(1, int(round(video_fps / max(fps, 1e-6))))
        total = min(max_frames, int(duration * fps) + 1)

        result = SamplingResult(video_path=video_path, duration=duration, fps=video_fps)
        frame_idx = 0
        while frame_idx < total:
            pos = frame_idx * step
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = cap.read()
            if not ok:
                break
            t = pos / video_fps
            out_path = out_dir / f"frame_{frame_idx:04d}_{t:07.3f}s.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            result.frames.append(SampledFrame(index=frame_idx, time=t, path=out_path))
            frame_idx += 1
        return result
    finally:
        cap.release()
