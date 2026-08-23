# The stream processor uses TensorRT/YOLO, so only this image inherits CUDA.
FROM base-gpu:latest

# The shared base image ends as appuser; package installation needs root.
USER root

# Set the working directory in the container
WORKDIR /app

RUN uv export \
    --quiet \
    --format=requirements-txt \
    --no-dev \
    --no-emit-project \
    --frozen \
    --extra streaming \
    --extra yolo \
    --extra yolo-gpu \
    -o /tmp/requirements.txt \
    && uv pip install --system -r /tmp/requirements.txt \
    && rm -rf /root/.cache/pip /root/.cache/uv /tmp/requirements.txt

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg fonts-noto-core fonts-noto-cjk libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary files into the container
COPY --chown=appuser:appuser config /app/config
COPY --chown=appuser:appuser examples /app/examples
COPY --chown=appuser:appuser src /app/src
COPY --chown=appuser:appuser main.py /app/main.py

# Ensure 'appuser' exists (if not already present in base image)
RUN groupadd -g 1001 appuser && useradd -u 1001 -g appuser -m appuser || true

# Create the 'logs' directory and set ownership
RUN mkdir -p /app/logs && chown appuser:appuser /app/logs

# Switch to the non-root user
USER appuser

# Set ENTRYPOINT to allow dynamic arguments for the configuration file
ENTRYPOINT ["python3", "main.py"]

# Default CMD provides a placeholder configuration file
CMD ["--config", "/app/config/configuration.json"]
