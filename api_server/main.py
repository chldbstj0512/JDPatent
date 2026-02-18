from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException

from api_server.celery_app import celery_app
from api_server.schemas import JobStatusResponse, SubmitJobRequest, SubmitJobResponse
from api_server.tasks import run_job


app = FastAPI(
    title="JDPatent Internal API",
    description="Internal async wrapper around legacy JDPatent analysis",
    version="0.1.0",
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/jobs", response_model=SubmitJobResponse)
def submit_job(payload: SubmitJobRequest) -> SubmitJobResponse:
    if not payload.raw_text.strip():
        raise HTTPException(status_code=400, detail="raw_text must not be empty")

    run_job.apply_async(
        kwargs={
            "task_id": payload.task_id,
            "raw_text": payload.raw_text,
            "user_id": payload.user_id,
            "user_prefer": payload.user_prefer,
            "user_prefer_nation": payload.user_prefer_nation,
            "user_prefer_area": payload.user_prefer_area,
        },
        task_id=payload.task_id,
    )
    return SubmitJobResponse(task_id=payload.task_id, status="queued")


@app.get("/api/v1/jobs/{task_id}", response_model=JobStatusResponse)
def get_job(task_id: str) -> JobStatusResponse:
    task = AsyncResult(task_id, app=celery_app)

    if task.state == "PENDING":
        # PENDING may mean either queued or unknown; check backend meta.
        backend = celery_app.backend
        task_meta = backend.get_task_meta(task_id)
        if not task_meta or task_meta.get("status") == "PENDING":
            return JobStatusResponse(task_id=task_id, status="PENDING")

    if task.state == "SUCCESS":
        return JobStatusResponse(task_id=task_id, status="SUCCESS", result=task.result)
    if task.state == "FAILURE":
        return JobStatusResponse(task_id=task_id, status="FAILURE", error=str(task.info))
    return JobStatusResponse(task_id=task_id, status=task.state)

