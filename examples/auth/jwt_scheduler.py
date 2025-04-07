from __future__ import annotations

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI

from examples.auth.security import update_secret_key


def start_jwt_scheduler(app: FastAPI) -> BackgroundScheduler:
    """
    Initialise a background scheduler to refresh the JWT secret key
    every 30 days.

    Args:
        app (FastAPI): The main FastAPI application instance
            to which the secret key update will be applied.

    Returns:
        BackgroundScheduler: The started background scheduler that periodically
            executes the secret key update job.

    Note:
        The job is scheduled to run every 30 days based on an interval trigger.
    """
    scheduler: BackgroundScheduler = BackgroundScheduler()
    scheduler.add_job(
        func=lambda: update_secret_key(app),
        trigger='interval',
        days=30,
    )
    scheduler.start()
    return scheduler
