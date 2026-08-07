"""audio 子模块：音频数值/音色特征提取（CPU）。

分层设计：
    - voice_analyzer:    音色特征（SpeechBrain ECAPA-TDNN 优先，内置统计 fallback）
    - music_analyzer:    BPM / 调性 / 响度曲线（librosa 优先，torchaudio/numpy fallback）
    - stereo_analyzer:   左右声道差异、声场宽度（numpy）
    - source_separator:  音源分离（Demucs，可选）
"""
