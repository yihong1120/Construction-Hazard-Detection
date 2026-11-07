# Use the shared base image
ARG BASE_IMAGE=base:latest
FROM ${BASE_IMAGE}

# Set the application working directory.
WORKDIR /app

# Shared lifecycles and policies span backend packages. Copy the source tree
# together; optional ML imports remain lazy and artifacts stay build-ignored.
COPY --chown=appuser:appuser examples /app/examples
COPY --chown=appuser:appuser src /app/src

# Expose the application port
EXPOSE 8005

# Run the database-management ASGI application.
CMD ["uvicorn", "examples.db_management.app:app", "--host", "0.0.0.0", "--port", "8005"]
