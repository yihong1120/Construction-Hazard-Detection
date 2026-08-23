# Use a lightweight Python image as the base
FROM python:3.14-slim-bookworm

# Set the working directory
WORKDIR /app

# Copy project metadata and the lockfile for deterministic dependency installs
COPY pyproject.toml uv.lock /app/

# Install a pinned uv release and use the lockfile to install dependencies
RUN python -m pip install --no-cache-dir "uv==0.9.9" && \
    uv --version && \
    uv export \
        --quiet \
        --format=requirements-txt \
        --no-dev \
        --no-emit-project \
        --frozen \
        -o /tmp/requirements.txt && \
    uv pip install --system -r /tmp/requirements.txt && \
    rm -rf /root/.cache/pip /root/.cache/uv /tmp/requirements.txt

# Create a non-root user for security
RUN useradd -ms /bin/bash appuser && \
    chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Set default working directory for the user
WORKDIR /app

# Default command (can be overridden by docker-compose or other tools)
CMD ["python3"]
