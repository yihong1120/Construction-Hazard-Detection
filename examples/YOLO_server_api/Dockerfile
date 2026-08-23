# YOLO inference is the only API that requires the GPU extension.
FROM base-gpu:latest

# Set working directory
WORKDIR /app

USER root
RUN uv export \
    --quiet \
    --format=requirements-txt \
    --no-dev \
    --no-emit-project \
    --frozen \
    --extra yolo \
    --extra yolo-gpu \
    -o /tmp/requirements.txt \
    && uv pip install --system -r /tmp/requirements.txt \
    && rm -rf /root/.cache/pip /root/.cache/uv /tmp/requirements.txt
USER appuser

# Copy only the service and its shared runtime packages. Models are mounted at
# runtime so image builds stay deterministic and do not duplicate multi-GB data.
COPY --chown=appuser:appuser examples/auth /app/examples/auth
COPY --chown=appuser:appuser examples/shared /app/examples/shared
COPY --chown=appuser:appuser examples/YOLO_server_api /app/examples/YOLO_server_api
COPY --chown=appuser:appuser src /app/src

# Expose the port for the YOLO server API
EXPOSE 6000

# Start the YOLO server API
CMD ["uvicorn", "examples.YOLO_server_api.app:app", "--host", "0.0.0.0", "--port", "6000"]
