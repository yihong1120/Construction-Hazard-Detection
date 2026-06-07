#!/bin/bash

# Wait for the PostgreSQL container to start and become available
echo "Waiting for PostgreSQL to be ready..."
while ! pg_isready -h postgres -U "$POSTGRES_USER" -d "$POSTGRES_DB"; do
    sleep 1  # Pause for one second before checking again
done

# Execute the SQL script to create tables in the database
echo "Creating tables..."
PGPASSWORD="$POSTGRES_PASSWORD" psql \
    -h postgres \
    -U "$POSTGRES_USER" \
    -d "$POSTGRES_DB" \
    -f /app/init.postgres.sql

# Confirm that the tables were successfully created
echo "Tables created successfully."
