-- Create the construction_hazard_detection database
CREATE DATABASE IF NOT EXISTS construction_hazard_detection;

-- Select the construction_hazard_detection database
USE construction_hazard_detection;

-- Create the users table
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(80) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'user',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Insert initial user
INSERT INTO users (username, password_hash, role, is_active) VALUES
('user', 'scrypt:32768:8:1$T2AeQL78js2oJaUz$5c51c110fca53b8f9f2d312a662b5f60cb1134ec0c1395243966b673fcd8cae59da66a66198f5880611cd9b68d2abba789d7523c4cccfa5882125667f0451f7e', 'admin', 1);

-- Create the projects table
-- mysql -u root -p < scripts/init.sql
