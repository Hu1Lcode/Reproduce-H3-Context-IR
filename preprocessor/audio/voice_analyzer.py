"""音色特征提取。

优先使用 SpeechBrain ECAPA-TDNN（说话人 embedding + 音色描述）；
未安装时 fallback 到内置声学统计特征（基频/频谱质心/过零率等），
保证在无 GPU、无 SpeechBrain 的环境中仍可产出可用的音色描述。

输出用途：Ref2VA 模式下将说话人音色映射到目标 <Subject N> 的 <Audio N>，
Captioner 不支持细粒度音色分析，故由本地层补充。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VoiceAnalysis:
    method: str = "builtin"  # speechbrain | builtin
    embedding: list[float] | None = None
    features: dict[str, float] = field(default_factory=dict)
    description: str = ""  # 自然语言音色描述

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "embedding": self.embedding,
            "features": self.features,
            "description": self.description,
        }


def analyze_voice(audio_path: str | Path) -> VoiceAnalysis:
    """分析音频中说话人的音色特征。"""
    try:
        return _analyze_speechbrain(audio_path)
    except Exception:
        return _analyze_builtin(audio_path)


def _load_mono(audio_path: str | Path, sr: int = 16000) -> tuple[Any, int]:
    """加载单声道音频，返回 (signal, sample_rate)。signal 为 float32 1-D numpy 数组。"""
    import numpy as np

    # 优先 soundfile（librosa 依赖它）
    try:
        import soundfile as sf

        data, fs = sf.read(str(audio_path), dtype="float32", always_2d=True)
        if data.shape[1] > 1:
            data = data.mean(axis=1)
        return data, fs
    except Exception:
        try:
            import torchaudio

            wav, fs = torchaudio.load(str(audio_path))
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            return wav[0].numpy().astype("float32"), int(fs)
        except Exception as exc:  # pragma: no cover
            raise ValueError(f"无法解码音频 {audio_path}: {exc}") from exc


def _analyze_speechbrain(audio_path: str | Path) -> VoiceAnalysis:
    """SpeechBrain ECAPA-TDNN 说话人 embedding。"""
    from speechbrain.inference.speaker import EncoderClassifier  # type: ignore

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir="/tmp/spkrec-ecapa"
    )
    signal, fs = _load_mono(audio_path)
    emb = classifier.encode_batch(signal[None, :], fs).squeeze().tolist()
    return VoiceAnalysis(
        method="speechbrain",
        embedding=emb,
        description="speaker embedding extracted with ECAPA-TDNN (dim=192)",
    )


def _analyze_builtin(audio_path: str | Path) -> VoiceAnalysis:
    """内置轻量音色分析：基频统计 + 频谱质心 + 过零率 + 能量动态。"""
    import numpy as np

    signal, fs = _load_mono(audio_path, sr=16000)
    if len(signal) == 0:
        raise ValueError(f"音频为空: {audio_path}")

    # 简单基频估计（自相关法，取中位数作为说话人基频）
    frame_len = int(0.03 * fs)
    hop = frame_len // 2
    pitches: list[float] = []
    for start in range(0, len(signal) - frame_len, hop):
        frame = signal[start : start + frame_len]
        frame = frame - frame.mean()
        energy = np.sum(frame**2)
        if energy < 1e-6:
            continue
        corr = np.correlate(frame, frame, "full")[frame_len - 1 :]
        min_lag, max_lag = int(fs / 400.0), int(fs / 60.0)
        if max_lag >= len(corr):
            continue
        segment = corr[min_lag:max_lag]
        lag = int(np.argmax(segment)) + min_lag
        if segment[int(np.argmax(segment))] > 0.3 * corr[0]:
            pitches.append(fs / lag)

    # 频谱质心（短时 FFT 均值）
    frame = signal[: min(len(signal), fs * 2)]
    spec = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
    freqs = np.fft.rfftfreq(len(frame), 1.0 / fs)
    if spec.sum() > 0:
        centroid = float(np.sum(freqs * spec) / np.sum(spec))
    else:
        centroid = 0.0

    zcr = float(np.mean(np.abs(np.diff(np.sign(signal))) / 2))
    rms = float(np.sqrt(np.mean(signal**2)))
    rms_db = 20.0 * np.log10(max(rms, 1e-8))

    if pitches:
        f0_med = float(np.median(pitches))
        f0_std = float(np.std(pitches))
        if f0_med < 165:
            pitch_label = "low-pitched male-like voice"
        elif f0_med < 255:
            pitch_label = "medium-pitched voice"
        else:
            pitch_label = "high-pitched female-like voice"
        pitch_desc = f"{pitch_label}, fundamental frequency around {f0_med:.0f} Hz"
    else:
        f0_med, f0_std = 0.0, 0.0
        pitch_desc = "no stable pitch detected (possibly non-speech or whispered)"

    features = {
        "pitch_median_hz": f0_med,
        "pitch_std_hz": f0_std,
        "spectral_centroid_hz": centroid,
        "zero_crossing_rate": zcr,
        "rms_db": rms_db,
    }

    if centroid < 800:
        brightness = "dark, warm timbre"
    elif centroid < 1800:
        brightness = "neutral timbre"
    else:
        brightness = "bright, forward timbre"

    description = f"{pitch_desc}; {brightness} with {zcr:.3f} zero-crossing rate"
    return VoiceAnalysis(method="builtin", features=features, description=description)
