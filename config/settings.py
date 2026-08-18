"""全局配置：API keys、模型端点、provider 选择。

配置优先级（低 → 高）：
    1. 代码默认值
    2. config/config.yaml（存在则自动加载，也可用 H3C_CONFIG 指定路径）
    3. 环境变量（临时覆盖，如临时切换到其他端点）

YAML 配置示例见 config/config.yaml。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Provider 配置
# ---------------------------------------------------------------------------
# 支持的 provider：dashscope（阿里云百炼）/ deepseek / openai-compatible（本地 vLLM 等）
# 所有 provider 均使用 OpenAI 兼容协议封装，仅在 base_url / api_key / 模型名上区分。


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    api_key_env: str
    api_key: str = ""
    default_model: str = ""
    requires_key: bool = True

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get(self.api_key_env, "").strip()
        # 本机地址（127.0.0.1 / localhost / 0.0.0.0）自动豁免 API key
        if self.requires_key and self.base_url.startswith(
            ("http://127.0.0.1", "http://localhost", "http://0.0.0.0")
        ):
            self.requires_key = False

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> "ProviderConfig":
        """从 YAML dict 构建。支持 api_key 直接写值，或 api_key_env 引用环境变量。"""
        api_key = str(d.get("api_key", "") or "").strip()
        return cls(
            name=name,
            base_url=str(d.get("base_url", "")).strip(),
            api_key_env=str(d.get("api_key_env", f"{name.upper()}_API_KEY")).strip(),
            api_key=api_key,
            default_model=str(d.get("default_model", "") or "").strip(),
            requires_key=bool(d.get("requires_key", True)),
        )


def _env_or(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


def _default_providers() -> dict[str, ProviderConfig]:
    """默认 provider 配置（可被 YAML / 环境变量覆盖）。"""
    return {
        "dashscope": ProviderConfig(
            name="dashscope",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key_env="DASHSCOPE_API_KEY",
            default_model="qwen3.6-plus",
        ),
        "deepseek": ProviderConfig(
            name="deepseek",
            base_url="https://api.deepseek.com/v1",
            api_key_env="DEEPSEEK_API_KEY",
            default_model="deepseek-v4-flash",
        ),
        "local": ProviderConfig(
            name="local",
            base_url="http://127.0.0.1:8000/v1",
            api_key_env="LOCAL_OPENAI_API_KEY",
            default_model="",
        ),
    }


@dataclass
class Settings:
    # --- 模型映射：步骤 → (provider, model) ---
    # Step 1 语义提取：多模态（图片/视频 URL 直传）→ qwen3.6-plus
    step1_model: tuple[str, str] = ("dashscope", "qwen3.6-plus")
    # Step 2 跨模态关联：文本为主（必要时回看媒体）→ qwen3.6
    step2_model: tuple[str, str] = ("dashscope", "qwen3.6")
    # Step 3 / 4 / 5：文本规划 / 推理校验 / 格式化输出 → deepseek-v4-flash
    step3_model: tuple[str, str] = ("deepseek", "deepseek-v4-flash")
    step4_model: tuple[str, str] = ("deepseek", "deepseek-v4-flash")
    step5_model: tuple[str, str] = ("deepseek", "deepseek-v4-flash")
    # 音频语义理解（预处理阶段）：Qwen3-Omni-Captioner
    audio_captioner_model: tuple[str, str] = ("dashscope", "qwen3-omni-30b-a3b-captioner")

    # --- Provider 配置表（YAML / 环境变量可覆盖）---
    providers: dict[str, ProviderConfig] = field(default_factory=_default_providers)

    # --- 请求参数 ---
    temperature_step3: float = 0.3      # Shot 规划：需要一定创造性
    temperature_step4: float = 0.1      # 逻辑校验：低温度保证严谨
    temperature_step5: float = 0.2      # 格式化润色：低温度保证格式稳定
    max_tokens_step: int = 8192

    # --- 客户端行为 ---
    request_timeout: float = 300.0
    max_retries: int = 3
    retry_backoff_base: float = 2.0
    # 并发限制（Semaphore 大小，避免触发平台限流）
    max_concurrent: int = 4

    # --- Step 5 输出策略 ---
    # True：确定性模板拼接后，再用 LLM 做语言润色（温度 ≤ 0.2）
    # False：纯确定性拼接（格式 100% 可控，推荐用于验证）
    llm_polish: bool = False

    # --- 预处理器 ---
    frame_sample_fps: float = 1.0       # 视频帧采样 ~1fps
    scene_detector: str = "auto"        # auto / pyscenedetect / frame_diff
    audio_max_seconds: int = 900        # Captioner 单段音频上限（~43 分钟 ≈ 2580s，保守取 900s）
    work_dir: Path = PROJECT_ROOT / "work"

    # --- 服务 ---
    server_host: str = "0.0.0.0"
    server_port: int = 8888
    task_store_dir: Path = PROJECT_ROOT / "work" / "tasks"

    # --- 缓存 ---
    enable_cache: bool = True
    cache_dir: Path = PROJECT_ROOT / "work" / "cache"

    # ------------------------------------------------------------------
    def provider(self, name: str) -> ProviderConfig:
        if name not in self.providers:
            raise ValueError(f"未知 provider: {name!r}，可用 {list(self.providers)}")
        return self.providers[name]


# ---------------------------------------------------------------------------
# YAML 加载
# ---------------------------------------------------------------------------
def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件 {path} 顶层必须是映射")
    return data


def _apply_yaml(s: Settings, data: dict[str, Any]) -> None:
    """把 YAML 配置覆盖到 Settings（不覆盖未在 YAML 中出现的项）。"""
    # providers
    prov_cfg = data.get("providers")
    if isinstance(prov_cfg, dict):
        for name, pd in prov_cfg.items():
            if not isinstance(pd, dict):
                continue
            if name in s.providers:
                # 部分覆盖：保留未指定的字段
                merged = {
                    "base_url": s.providers[name].base_url,
                    "api_key_env": s.providers[name].api_key_env,
                    "default_model": s.providers[name].default_model,
                }
                merged.update(pd)
                s.providers[name] = ProviderConfig.from_dict(name, merged)
            else:
                s.providers[name] = ProviderConfig.from_dict(name, pd)

    # models：step1..step5 / captioner → (provider, model)
    models = data.get("models")
    if isinstance(models, dict):
        mapping = {
            "step1": "step1_model",
            "step2": "step2_model",
            "step3": "step3_model",
            "step4": "step4_model",
            "step5": "step5_model",
            "captioner": "audio_captioner_model",
        }
        for key, attr in mapping.items():
            m = models.get(key)
            if isinstance(m, dict) and m.get("provider"):
                setattr(s, attr, (str(m["provider"]), str(m.get("model", ""))))

    # runtime 参数
    rt = data.get("runtime")
    if isinstance(rt, dict):
        for key, attr in {
            "temperature_step3": "temperature_step3",
            "temperature_step4": "temperature_step4",
            "temperature_step5": "temperature_step5",
            "max_tokens_step": "max_tokens_step",
            "request_timeout": "request_timeout",
            "max_retries": "max_retries",
            "max_concurrent": "max_concurrent",
            "llm_polish": "llm_polish",
            "frame_sample_fps": "frame_sample_fps",
            "scene_detector": "scene_detector",
            "audio_max_seconds": "audio_max_seconds",
            "enable_cache": "enable_cache",
            "server_host": "server_host",
            "server_port": "server_port",
        }.items():
            if key in rt and rt[key] is not None:
                setattr(s, attr, rt[key])


def _apply_env(s: Settings) -> None:
    """环境变量覆盖（最高优先级，用于临时切换端点/模型）。"""

    def _pick(env_name: str, default: tuple[str, str]) -> tuple[str, str]:
        raw = os.environ.get(env_name, "")
        if raw:
            parts = raw.split(":", 1)
            return (parts[0], parts[1]) if len(parts) == 2 else (parts[0], default[1])
        return default

    s.step1_model = _pick("H3C_STEP1_MODEL", s.step1_model)
    s.step2_model = _pick("H3C_STEP2_MODEL", s.step2_model)
    s.step3_model = _pick("H3C_STEP3_MODEL", s.step3_model)
    s.step4_model = _pick("H3C_STEP4_MODEL", s.step4_model)
    s.step5_model = _pick("H3C_STEP5_MODEL", s.step5_model)
    s.audio_captioner_model = _pick("H3C_CAPTIONER_MODEL", s.audio_captioner_model)
    _polish = os.environ.get("H3C_LLM_POLISH")
    if _polish is not None:
        s.llm_polish = _polish in ("1", "true", "True")
    s.enable_cache = os.environ.get("H3C_CACHE", "1") not in ("0", "false", "False")

    base = os.environ.get("LOCAL_OPENAI_BASE_URL", "")
    if base:
        s.providers["local"].base_url = base


def load_settings(yaml_path: str | Path | None = None) -> Settings:
    """加载配置：默认值 → YAML → 环境变量。

    Args:
        yaml_path: YAML 配置文件路径；None 时依次尝试
            H3C_CONFIG 环境变量、config/config.yaml。
    """
    s = Settings()
    if yaml_path is None:
        env_cfg = os.environ.get("H3C_CONFIG", "")
        yaml_path = env_cfg or (PROJECT_ROOT / "config" / "config.yaml")
    _apply_yaml(s, _load_yaml(Path(yaml_path)))
    _apply_env(s)
    return s


# 全局实例（模块加载时初始化一次；调用 load_settings() 可重新加载）
settings = load_settings()
