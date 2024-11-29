#!/bin/bash

# Wait for the MySQL container to start and become available
echo "Waiting for MySQL to be ready..."
while ! mysqladmin ping -h mysql --silent -u"$MYSQL_USER" -p"$MYSQL_PASSWORD"; do
    sleep 1  # Pause for one second before checking again
done

# Execute the SQL script to create tables in the database
echo "Creating tables..."
mysql -u"$MYSQL_USER" -p"$MYSQL_PASSWORD" -h mysql construction_hazard_detection < /app/init.sql

# Confirm that the tables were successfully created
echo "Tables created successfully."
