# Use the shared base image
ARG BASE_IMAGE=base:latest
FROM ${BASE_IMAGE}

# Set the application working directory.
WORKDIR /app

# Dependencies are installed by the shared base image. Copy every Python
# package used by this deployable so the image also works without bind mounts.
COPY --chown=appuser:appuser examples/auth /app/examples/auth
COPY --chown=appuser:appuser src /app/src
COPY --chown=appuser:appuser examples/deployment_registry /app/examples/deployment_registry
COPY --chown=appuser:appuser examples/bff /app/examples/bff
COPY --chown=appuser:appuser examples/db_management /app/examples/db_management
COPY --chown=appuser:appuser examples/shared /app/examples/shared
# Playback naming and language policy have no inference/image dependencies.
COPY --chown=appuser:appuser examples/streaming_web/__init__.py examples/streaming_web/media_paths.py examples/streaming_web/metadata_keys.py examples/streaming_web/overlay_labels.py /app/examples/streaming_web/
# Site membership changes invalidate the shared notification recipient index.
COPY --chown=appuser:appuser examples/local_notification_server/__init__.py examples/local_notification_server/site_recipient_cache.py /app/examples/local_notification_server/
# Policy-only modules: do not import the inference or violation application.
COPY --chown=appuser:appuser examples/YOLO_server_api/__init__.py examples/YOLO_server_api/model_registry.py /app/examples/YOLO_server_api/

# Expose the application port
EXPOSE 8005

# Run the database-management ASGI application.
CMD ["uvicorn", "examples.db_management.app:app", "--host", "0.0.0.0", "--port", "8005"]
