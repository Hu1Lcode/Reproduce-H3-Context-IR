"""音源分离（可选）。

Demucs 未安装时返回 None（跳过分离），不影响主流程。
用途：区分 diegetic（人声/环境音/音效）与非 diegetic（BGM）层。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SeparationResult:
    method: str = "demucs"
    stems: dict[str, Path] = field(default_factory=dict)  # stem 名 -> 音频文件
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "stems": {k: str(v) for k, v in self.stems.items()},
            "note": self.note,
        }


def separate_sources(
    audio_path: str | Path,
    out_dir: str | Path,
    model: str = "htdemucs",
) -> SeparationResult | None:
    """分离人声 / BGM / 环境音 / 音效。

    Args:
        audio_path: 输入音频。
        out_dir: 输出目录。
        model: Demucs 模型名。

    Returns:
        SeparationResult；Demucs 不可用时返回 None。
    """
    try:
        import demucs.separate  # type: ignore
    except ImportError:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # demucs CLI 风格调用：demucs.separate.main(["-o", out, "-n", model, audio])
    import sys

    sys.argv = [
        "demucs",
        "-o",
        str(out_dir),
        "-n",
        model,
        str(audio_path),
    ]
    try:
        demucs.separate.main()
    except SystemExit:
        pass

    stems = {}
    track_dir = out_dir / model / Path(audio_path).stem
    if track_dir.exists():
        for wav in track_dir.glob("*.wav"):
            stems[wav.stem] = wav
    return SeparationResult(stems=stems)
