# Build this image from the repository root so shared packages can be copied.
FROM base:latest

WORKDIR /app

USER root
RUN uv export \
    --quiet \
    --format=requirements-txt \
    --no-dev \
    --no-emit-project \
    --frozen \
    --extra notification \
    -o /tmp/requirements.txt \
    && uv pip install --system -r /tmp/requirements.txt \
    && rm -rf /root/.cache/pip /root/.cache/uv /tmp/requirements.txt
USER appuser

# Copy only runtime packages required by the notification application.
COPY --chown=appuser:appuser examples/auth /app/examples/auth
COPY --chown=appuser:appuser examples/db_management /app/examples/db_management
COPY --chown=appuser:appuser \
    examples/local_notification_server \
    /app/examples/local_notification_server
COPY --chown=appuser:appuser src /app/src

# Service-account credentials are mounted at runtime and are never baked in.
ENV FIREBASE_CRED_PATH=/run/secrets/firebase-service-account.json

EXPOSE 8003

USER appuser

# The process manager may override the command to configure worker count.
CMD ["uvicorn", "examples.local_notification_server.app:app", "--host", "0.0.0.0", "--port", "8003"]
