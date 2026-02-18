import os

from celery import Celery


REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/1")

celery_app = Celery(
    "jdpatent_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["api_server.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_time_limit=1800,
    task_soft_time_limit=1500,
    result_expires=3600,
)

