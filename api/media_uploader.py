"""媒体文件上传 / URL 化。

本地文件需转为 API 可访问的 URL（对象存储 / 临时公网链接）。
本模块提供可插拔接口：
    - 内置：开发用 HTTP 静态文件服务（uvicorn/python http.server），
      把工作目录暴露为 http://<host>:<port>/<file>。
    - 自定义：实现 uploader() 函数并注册（例如 OSS/七牛/临时公网）。
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

logger = logging.getLogger("h3c.api.media")

# 注册表：名称 -> 上传函数 (local_path: Path) -> str(URL)
_uploaders: dict[str, Callable[[Path], str]] = {}

# 内置开发服务器状态
_dev_server: dict = {"thread": None, "port": 0, "base_url": "", "root": None}


def register_uploader(name: str, fn: Callable[[Path], str]) -> None:
    """注册自定义上传器。"""
    _uploaders[name] = fn


def upload_media(
    local_path: str | Path,
    uploader: str | None = None,
    host: str = "0.0.0.0",
    port: int = 9080,
) -> str:
    """把本地媒体转为可访问 URL。

    Args:
        local_path: 本地文件。
        uploader: 上传器名；None 表示使用内置开发静态服务
            （把文件复制到静态根目录后返回 URL）。
        host/port: 内置开发服务参数（首次调用时自动启动）。
    """
    path = Path(local_path)
    if not path.exists():
        raise FileNotFoundError(f"媒体文件不存在: {path}")

    if uploader is not None:
        if uploader not in _uploaders:
            raise KeyError(f"未注册的上传器: {uploader}，已注册 {list(_uploaders)}")
        return _uploaders[uploader](path)

    # 内置开发服务：将文件软链/复制到静态根目录
    root, base_url = _ensure_dev_server(host, port)
    rel = path.name
    dest = root / rel
    if not dest.exists():
        try:
            dest.symlink_to(path.resolve())
        except OSError:
            import shutil

            shutil.copy2(path, dest)
    return f"{base_url}/{rel}"


def _ensure_dev_server(host: str, port: int) -> tuple[Path, str]:
    """启动（若未启动）内置静态文件服务，返回 (root, base_url)。"""
    from config.settings import settings

    if _dev_server["thread"] is not None and _dev_server["thread"].is_alive():
        return _dev_server["root"], _dev_server["base_url"]

    root = settings.work_dir / "static"
    root.mkdir(parents=True, exist_ok=True)

    import functools
    import http.server

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    httpd = http.server.ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    _dev_server.update(thread=thread, port=port, base_url=f"http://{host}:{port}", root=root)
    logger.info("内置媒体静态服务已启动: http://%s:%s", host, port)
    return root, _dev_server["base_url"]
