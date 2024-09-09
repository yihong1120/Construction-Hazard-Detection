#!/bin/bash
# This script will run when the MySQL container starts

# Wait for MySQL to start
sleep 10

# Run SQL commands to create the table and insert the user
mysql -h db -u user -ppasscode construction_hazard_detection <<-EOSQL
    CREATE TABLE IF NOT EXISTS user (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(80) UNIQUE NOT NULL,
        password_hash VARCHAR(120) NOT NULL
    );

    INSERT INTO user (username, password_hash) VALUES
    ('user', 'scrypt:32768:8:1\$2Oy6iiKWmJiNLpbG\$9355a9329c805bf5b7252fe3062b8cc69643febf36fda93bf78ef1a1243a3879fe325db6a801c4d828068fea0f4cfa46791fa0c991cac2433bc1045e0c8c2677')
    ON DUPLICATE KEY UPDATE
    password_hash=VALUES(password_hash);
EOSQL
