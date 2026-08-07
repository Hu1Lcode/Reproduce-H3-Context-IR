"""立体声分析：左右声道差异、声场宽度。

numpy 实现，无需额外依赖。输出供音频立体声特性描述。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class StereoAnalysis:
    channels: int = 1
    correlation: float = 1.0        # 左右声道相关系数（1=完全相关/单声道）
    level_difference_db: float = 0.0  # |L-R| 平均电平差
    width_description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "channels": self.channels,
            "correlation": self.correlation,
            "level_difference_db": self.level_difference_db,
            "width_description": self.width_description,
        }


def analyze_stereo(audio_path: str | Path) -> StereoAnalysis:
    """分析立体声特性。

    规则：
        - 相关系数 ≈ 1 且电平差 ≈ 0 → mono-like / narrow
        - 相关系数 < 0.5 → wide / decorrelated
    """
    try:
        import soundfile as sf

        data, _ = sf.read(str(audio_path), dtype="float32", always_2d=True)
    except Exception:
        try:
            import torchaudio

            wav, _ = torchaudio.load(str(audio_path))
            data = wav.numpy().astype("float32").T  # (n, c)
        except Exception as exc:  # pragma: no cover
            raise ValueError(f"无法解码音频 {audio_path}: {exc}") from exc

    if data.ndim == 1 or data.shape[1] == 1:
        return StereoAnalysis(channels=1, width_description="mono audio")

    left, right = data[:, 0], data[:, 1]
    left = left - left.mean()
    right = right - right.mean()

    denom = np.sqrt(np.sum(left**2) * np.sum(right**2))
    corr = float(np.sum(left * right) / denom) if denom > 0 else 1.0

    lrms = np.sqrt(np.mean(left**2))
    rrms = np.sqrt(np.mean(right**2))
    level_diff = float(abs(20.0 * np.log10(max(lrms, 1e-8)) - 20.0 * np.log10(max(rrms, 1e-8))))

    if corr > 0.95 and level_diff < 1.0:
        width = "mono-compatible, very narrow sound field"
    elif corr > 0.7:
        width = "narrow-to-moderate stereo width"
    elif corr > 0.4:
        width = "moderate stereo width with some decorrelation"
    else:
        width = "wide stereo field, channels substantially decorrelated"

    return StereoAnalysis(
        channels=2,
        correlation=corr,
        level_difference_db=round(level_diff, 2),
        width_description=width,
    )
