"""Provider 配置解析（从 Settings 动态读取，支持 YAML / 环境变量覆盖）。

providers 的实际配置保存在 config.settings.Settings.providers 中，
可在 config/config.yaml 中覆盖（见 config/config.yaml 示例）。
"""
from __future__ import annotations


def resolve_provider(name: str, settings=None):
    """按名字获取 provider 配置。

    Args:
        name: provider 名（dashscope / deepseek / local，或 YAML 中自定义的 provider）
        settings: Settings 实例；None 时使用全局 settings。

    Returns:
        ProviderConfig
    """
    if settings is None:
        from config.settings import settings as _global

        settings = _global
    return settings.provider(name)


def check_api_key(name: str, settings=None) -> None:
    """校验 provider 的 API key 是否已配置（本地地址端点自动豁免）。"""
    cfg = resolve_provider(name, settings)
    if cfg.requires_key and not cfg.api_key:
        raise RuntimeError(
            f"provider '{name}' 未配置 API key，请在 config/config.yaml 中配置 "
            f"api_key，或设置环境变量 {cfg.api_key_env}"
        )
