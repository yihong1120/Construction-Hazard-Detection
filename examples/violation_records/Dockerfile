# Use the shared base image
ARG BASE_IMAGE=base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

USER root
RUN uv export \
    --quiet \
    --format=requirements-txt \
    --no-dev \
    --no-emit-project \
    --frozen \
    --extra violation \
    -o /tmp/requirements.txt \
    && uv pip install --system -r /tmp/requirements.txt \
    && rm -rf /root/.cache/pip /root/.cache/uv /tmp/requirements.txt
USER appuser

# Dependencies are installed once by scripts/base.Dockerfile from uv.lock.
COPY --chown=appuser:appuser examples/auth /app/examples/auth
COPY --chown=appuser:appuser examples/deployment_registry /app/examples/deployment_registry
COPY --chown=appuser:appuser examples/shared /app/examples/shared
COPY --chown=appuser:appuser examples/db_management /app/examples/db_management
COPY --chown=appuser:appuser examples/YOLO_server_api/__init__.py examples/YOLO_server_api/model_registry.py /app/examples/YOLO_server_api/
COPY --chown=appuser:appuser examples/violation_records /app/examples/violation_records
COPY --chown=appuser:appuser src /app/src

EXPOSE 8081

CMD ["uvicorn", "examples.violation_records.app:app", "--host", "0.0.0.0", "--port", "8081"]
