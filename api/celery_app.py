import os
from celery import Celery

broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# Initialize Celery app
celery_app = Celery(
    "verity_tasks",
    broker=broker_url,
    backend=result_backend
)

# Eager fallback mode: executes tasks synchronously inline during development/testing
# if Redis isn't running or if CELERY_ALWAYS_EAGER env is True (which is the default).
eager_mode = os.environ.get("CELERY_ALWAYS_EAGER", "True").lower() == "true"
celery_app.conf.task_always_eager = eager_mode
celery_app.conf.task_eager_propagates = True

# Standard configs
celery_app.conf.timezone = 'UTC'
celery_app.conf.task_track_started = True

# Celery auto-discovery of tasks in the tasks module
celery_app.autodiscover_tasks(['api'])
