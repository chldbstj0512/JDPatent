from api_server.celery_app import celery_app
from api_server.service_wrapper import run_legacy_analysis


@celery_app.task(bind=True, name="api_server.tasks.run_job", max_retries=1, default_retry_delay=5)
def run_job(
    self,
    *,
    task_id: str,
    raw_text: str,
    user_id: str | None = None,
    user_prefer: str = "nation",
    user_prefer_nation: str | None = "South Korea",
    user_prefer_area: str | None = None,
):
    final_user_id = user_id or task_id
    return run_legacy_analysis(
        user_id=final_user_id,
        raw_text=raw_text,
        user_prefer=user_prefer,
        user_prefer_nation=user_prefer_nation,
        user_prefer_area=user_prefer_area,
    )

