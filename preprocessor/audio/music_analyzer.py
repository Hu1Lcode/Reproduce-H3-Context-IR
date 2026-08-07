"""音乐数值特征：BPM / 调性 / 响度曲线。

librosa 优先；未安装时 fallback 到 torchaudio/numpy 轻量实现。
输出供 non_diegetic_music 与 bgm 描述的数值补充。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MusicAnalysis:
    method: str = "librosa"  # librosa | builtin
    bpm: float | None = None
    key: str | None = None       # 调性（仅 librosa 支持）
    loudness_curve: list[float] = field(default_factory=list)  # 每秒 RMS(dB)
    spectral_centroid: float | None = None
    note: str = ""  # 说明（如工具缺失时的提示）

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "bpm": self.bpm,
            "key": self.key,
            "loudness_curve": self.loudness_curve,
            "spectral_centroid": self.spectral_centroid,
            "note": self.note,
        }


def analyze_music(audio_path: str | Path) -> MusicAnalysis:
    """分析音频音乐数值特征。"""
    try:
        return _analyze_librosa(audio_path)
    except Exception as exc:
        try:
            result = _analyze_builtin(audio_path)
            result.note = f"librosa unavailable, used builtin fallback ({exc})"
            return result
        except Exception as exc2:  # pragma: no cover
            raise ValueError(f"音乐分析失败: {exc2}") from exc2


def _analyze_librosa(audio_path: str | Path) -> MusicAnalysis:
    import librosa  # type: ignore

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    result = MusicAnalysis(method="librosa")

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    result.bpm = float(tempo)

    try:
        key = librosa.feature.tonnetz(y=y, sr=sr)
        result.key = f"tonnetz mean {key.mean():.3f}"
    except Exception:
        result.key = None

    rms = librosa.feature.rms(y=y)[0]
    hop = 512
    result.loudness_curve = [
        round(20.0 * float(__import__("numpy").log10(max(v, 1e-8))), 2) for v in rms
    ]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    result.spectral_centroid = float(cent.mean())
    return result


def _analyze_builtin(audio_path: str | Path) -> MusicAnalysis:
    """torchaudio/numpy 轻量实现：BPM（自相关）+ 响度曲线 + 频谱质心。"""
    import numpy as np

    from .voice_analyzer import _load_mono

    signal, sr = _load_mono(audio_path, sr=22050)
    result = MusicAnalysis(method="builtin")

    # 响度曲线（每秒 RMS）
    hop = sr
    curve = []
    for start in range(0, len(signal), hop):
        seg = signal[start : start + hop]
        rms = float(np.sqrt(np.mean(seg**2))) if len(seg) else 0.0
        curve.append(round(20.0 * np.log10(max(rms, 1e-8)), 2))
    result.loudness_curve = curve

    # 包络自相关 → BPM
    env = np.abs(signal)
    frame = int(0.1 * sr)
    env_frames = np.array(
        [env[i : i + frame].mean() for i in range(0, len(env) - frame, frame)]
    )
    env_frames -= env_frames.mean()
    if len(env_frames) > sr:  # 需要至少 ~2s
        corr = np.correlate(env_frames, env_frames, "full")[len(env_frames) - 1 :]
        min_lag, max_lag = int(sr / 200.0), int(sr / 40.0)  # 120-300 BPM 搜索
        if max_lag < len(corr):
            lag = int(np.argmax(corr[min_lag:max_lag])) + min_lag
            bpm = 60.0 * sr / (lag * frame)
            if 40.0 <= bpm <= 300.0:
                result.bpm = round(float(bpm), 2)

    spec = np.abs(np.fft.rfft(signal[: sr * 5] * np.hanning(sr * 5)))
    freqs = np.fft.rfftfreq(sr * 5, 1.0 / sr)
    if spec.sum() > 0:
        result.spectral_centroid = float(np.sum(freqs * spec) / np.sum(spec))
    return result
