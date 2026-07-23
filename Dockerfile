# Use the previously built base image
FROM base:latest

# The shared base image ends as appuser; package installation needs root.
USER root

# Set the working directory in the container
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
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
