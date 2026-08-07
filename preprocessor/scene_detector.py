"""场景切分检测。

优先使用 PySceneDetect（若已安装）；否则 fallback 到内置帧差法
（连续帧直方图/像素差异 > 阈值即判定为镜头边界）。两者均可在 CPU 运行。
输出：镜头边界时间戳列表（供 Step 3 Shot 分割参考）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SceneCut:
    index: int      # 切点所在帧号（新镜头第一帧）
    time: float     # 秒

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "time": self.time}


@dataclass
class SceneDetectionResult:
    cuts: list[SceneCut] = field(default_factory=list)
    method: str = "frame_diff"

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "cuts": [c.to_dict() for c in self.cuts]}


def detect_scenes(
    video_path: str | Path,
    threshold: float = 30.0,
    min_scene_len: float = 0.5,
) -> SceneDetectionResult:
    """检测视频镜头边界。

    Args:
        video_path: 输入视频。
        threshold: 帧差阈值（越大越不敏感，0-255 量纲）。
        min_scene_len: 最小镜头长度（秒），过滤过短的误报切点。

    Returns:
        SceneDetectionResult（按时间升序的切点）。
    """
    try:
        return _detect_pyscenedetect(video_path)
    except Exception:
        return _detect_frame_diff(video_path, threshold=threshold, min_scene_len=min_scene_len)


def _detect_pyscenedetect(video_path: str | Path) -> SceneDetectionResult:
    """PySceneDetect 实现（DetectContent 内容检测法）。"""
    from scenedetect import ContentDetector, SceneManager, open_video  # type: ignore

    video = open_video(str(video_path))
    sm = SceneManager()
    sm.add_detector(ContentDetector())
    sm.detect_scenes(video)
    fps = video.frame_rate or 25.0

    result = SceneDetectionResult(method="pyscenedetect")
    for i, scene in enumerate(sm.get_scene_list()):
        if i == 0:
            continue  # 第一段开头不是切点
        start_frame = scene[0].get_frames()
        result.cuts.append(SceneCut(index=start_frame, time=start_frame / fps))
    return result


def _detect_frame_diff(video_path: str | Path, threshold: float, min_scene_len: float) -> SceneDetectionResult:
    """内置帧差法 fallback：逐帧灰度直方图差异。"""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.is_opened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    result = SceneDetectionResult(method="frame_diff")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        prev_hist = None
        prev_cut_frame = 0
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            if prev_hist is not None:
                diff = cv2.norm(prev_hist, hist, cv2.NORM_L1)
                if diff > threshold * 64 / 255.0:  # 归一化阈值
                    if (frame_idx - prev_cut_frame) / fps >= min_scene_len:
                        result.cuts.append(SceneCut(index=frame_idx, time=frame_idx / fps))
                        prev_cut_frame = frame_idx
            prev_hist = hist
            frame_idx += 1
    finally:
        cap.release()
    return result
