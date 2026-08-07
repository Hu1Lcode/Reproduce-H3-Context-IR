"""FastAPI 服务：兼容官方 /v2/h3_context_ir 接口（异步任务语义）。

    POST /v2/h3_context_ir              → {"task_id": "..."}
    GET  /v2/query/video_generation/{task_id}  → 任务状态 + .task.content.prompt

任务在后台线程池执行完整流水线；任务状态保存在内存 + 磁盘（重启可恢复）。

启动：
    uvicorn server.server:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from config.settings import settings  # noqa: E402
from pipeline.base_pipeline import H3ContextPipeline  # noqa: E402
from pipeline.context import PipelineContext  # noqa: E402
from preprocessor.task_classifier import TaskClassifierError, classify_task  # noqa: E402
from server.schemas import (  # noqa: E402
    CreateResponse,
    H3ContextIRRequest,
    QueryResponse,
    TaskContent,
    TaskStatus,
)

logger = logging.getLogger("h3c.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="H3-Context-IR 复现服务", version="0.1.0")

# 任务存储：task_id -> dict
_tasks: dict[str, dict] = {}
_tasks_lock = threading.Lock()
_store_dir = Path(settings.task_store_dir)
_store_dir.mkdir(parents=True, exist_ok=True)


def _load_tasks_from_disk() -> None:
    for f in _store_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            _tasks[data["task_id"]] = data
        except Exception:  # noqa: BLE001
            continue


def _save_task(task: dict) -> None:
    (_store_dir / f"{task['task_id']}.json").write_text(
        json.dumps(task, ensure_ascii=False, indent=2)
    )


def _execute_task(task_id: str, req: H3ContextIRRequest) -> None:
    """后台执行完整流水线。"""
    with _tasks_lock:
        _tasks[task_id]["status"] = TaskStatus.processing
        _save_task(_tasks[task_id])
    try:
        content = [c.model_dump(exclude_none=True) for c in req.content]
        raw_input = {
            "content": content,
            "duration": req.duration,
            "ratio": req.ratio,
            "model": req.model,
        }
        ctx = PipelineContext(raw_input=raw_input)
        classification = classify_task(content)
        ctx.task_type = classification.task_type
        ctx.media_roles = classification.media_roles

        pipeline = H3ContextPipeline(settings=settings, skip_errors=False)
        ctx = pipeline.run(ctx)

        result = {
            "task_id": task_id,
            "status": TaskStatus.success,
            "content": TaskContent(
                prompt=ctx.final_prompt,
                task_type=ctx.task_type,
                validation=ctx.validation_report,
            ).model_dump(),
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("任务 %s 失败", task_id)
        result = {"task_id": task_id, "status": TaskStatus.failed, "error": str(exc)}
    with _tasks_lock:
        _tasks[task_id] = result
        _save_task(result)


@app.post("/v2/h3_context_ir", response_model=CreateResponse)
async def create_task(req: H3ContextIRRequest) -> CreateResponse:
    """创建 Context-IR 任务（异步）。"""
    # 快速失败：content 无 text
    try:
        classify_task([c.model_dump(exclude_none=True) for c in req.content])
    except TaskClassifierError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    task_id = str(uuid.uuid4())
    with _tasks_lock:
        _tasks[task_id] = {
            "task_id": task_id,
            "status": TaskStatus.pending,
            "content": None,
        }
        _save_task(_tasks[task_id])

    threading.Thread(target=_execute_task, args=(task_id, req), daemon=True).start()
    return CreateResponse(task_id=task_id)


@app.get("/v2/query/video_generation/{task_id}", response_model=QueryResponse)
async def query_task(task_id: str) -> QueryResponse:
    """查询任务状态与结果。"""
    with _tasks_lock:
        task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"task {task_id} 不存在")
    return QueryResponse(
        task_id=task["task_id"],
        status=task["status"],
        content=TaskContent(**task["content"]) if task.get("content") else None,
        error=task.get("error"),
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


_load_tasks_from_disk()
