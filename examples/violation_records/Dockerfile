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

# Shared lifecycles and policies span backend packages. Copy the source tree
# together; optional ML imports remain lazy and artifacts stay build-ignored.
COPY --chown=appuser:appuser examples /app/examples
COPY --chown=appuser:appuser src /app/src

EXPOSE 8081

CMD ["uvicorn", "examples.violation_records.app:app", "--host", "0.0.0.0", "--port", "8081"]
