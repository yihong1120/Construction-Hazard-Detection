# Use the shared base image
FROM base:latest

# Set the working directory within the container
WORKDIR /app

USER root
RUN uv export \
    --quiet \
    --format=requirements-txt \
    --no-dev \
    --no-emit-project \
    --frozen \
    --extra streaming \
    -o /tmp/requirements.txt \
    && uv pip install --system -r /tmp/requirements.txt \
    && rm -rf /root/.cache/pip /root/.cache/uv /tmp/requirements.txt

# MediaMTX metadata endpoints and diagnostics may use ffmpeg tooling.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg fonts-noto-core fonts-noto-cjk libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*
USER appuser

# Copy only runtime packages. The Docker Compose development bind mount may
# still replace these paths, while standalone images remain runnable.
COPY --chown=appuser:appuser examples/auth /app/examples/auth
COPY --chown=appuser:appuser examples/local_notification_server /app/examples/local_notification_server
COPY --chown=appuser:appuser examples/shared /app/examples/shared
COPY --chown=appuser:appuser examples/streaming_web /app/examples/streaming_web
COPY --chown=appuser:appuser src /app/src

# Expose the application port
EXPOSE 8000

# Define environment variables for Redis configuration (can be overridden at runtime)
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

# Define the default command to run the application
CMD ["uvicorn", "examples.streaming_web.app:app", "--host", "0.0.0.0", "--port", "8000"]
