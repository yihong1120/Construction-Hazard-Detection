# Use the shared base image
FROM base:latest

# Set the application working directory.
WORKDIR /app

# Dependencies are installed by the shared base image. Copy every Python
# package used by this deployable so the image also works without bind mounts.
COPY --chown=appuser:appuser examples/auth /app/examples/auth
COPY --chown=appuser:appuser examples/bff /app/examples/bff
COPY --chown=appuser:appuser examples/db_management /app/examples/db_management

# Expose the application port
EXPOSE 8005

# Run the database-management ASGI application.
CMD ["uvicorn", "examples.db_management.app:app", "--host", "0.0.0.0", "--port", "8005"]
