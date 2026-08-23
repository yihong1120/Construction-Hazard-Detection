# Use the shared base image
FROM base:latest

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
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
USER appuser

# Dependencies are installed once by scripts/base.Dockerfile from uv.lock.
COPY --chown=appuser:appuser examples/auth /app/examples/auth
COPY --chown=appuser:appuser examples/shared /app/examples/shared
COPY --chown=appuser:appuser examples/violation_records /app/examples/violation_records
COPY --chown=appuser:appuser src /app/src

EXPOSE 8081

CMD ["uvicorn", "examples.violation_records.app:app", "--host", "0.0.0.0", "--port", "8081"]
