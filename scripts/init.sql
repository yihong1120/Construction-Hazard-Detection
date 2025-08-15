-- =============================================
-- Construction-Hazard-Detection Database Schema
-- Compatible with MySQL 8.0/8.4/9.x (fixed order/FKs)
-- violations.site -> FK to sites(name)
-- =============================================

SET NAMES utf8mb4;
SET time_zone = '+00:00';

CREATE DATABASE IF NOT EXISTS construction_hazard_detection
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE construction_hazard_detection;

-- Avoid FK conflicts during import; re-enable at the end
SET FOREIGN_KEY_CHECKS = 0;

-- Safe for repeated imports: drop child tables before parent tables
DROP TABLE IF EXISTS group_features;
DROP TABLE IF EXISTS user_sites;
DROP TABLE IF EXISTS stream_configs;
DROP TABLE IF EXISTS violations;
DROP TABLE IF EXISTS user_profiles;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS sites;
DROP TABLE IF EXISTS features;
DROP TABLE IF EXISTS group_info;

-- ========== Parent Tables ==========
CREATE TABLE group_info (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    uniform_number VARCHAR(8) UNIQUE NOT NULL COMMENT 'Unified Business Number',
    max_allowed_streams INT NOT NULL DEFAULT 8,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE features (
    id INT PRIMARY KEY AUTO_INCREMENT,
    feature_name VARCHAR(50) UNIQUE NOT NULL,
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(80) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'user',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    group_id INT,
    CONSTRAINT fk_users_group
        FOREIGN KEY (group_id) REFERENCES group_info(id)
        ON DELETE SET NULL,
    INDEX idx_users_group (group_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE sites (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(80) NOT NULL,
    group_id INT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_sites_group
        FOREIGN KEY (group_id) REFERENCES group_info(id)
        ON DELETE SET NULL,
    -- To allow violations.site to be an FK, name must be unique
    UNIQUE KEY uq_sites_name (name),
    INDEX idx_sites_group (group_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ========== Relationship/Detail Tables ==========
CREATE TABLE group_features (
    group_id INT NOT NULL,
    feature_id INT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (group_id, feature_id),
    CONSTRAINT fk_gf_group FOREIGN KEY (group_id) REFERENCES group_info(id) ON DELETE CASCADE,
    CONSTRAINT fk_gf_feature FOREIGN KEY (feature_id) REFERENCES features(id) ON DELETE CASCADE,
    INDEX idx_gf_group (group_id),
    INDEX idx_gf_feature (feature_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE user_profiles (
    user_id INT PRIMARY KEY,
    family_name VARCHAR(50) NOT NULL,
    middle_name VARCHAR(50),
    given_name VARCHAR(50) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    mobile_number VARCHAR(20) UNIQUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_up_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE user_sites (
    user_id INT NOT NULL,
    site_id INT NOT NULL,
    PRIMARY KEY (user_id, site_id),
    CONSTRAINT fk_us_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT fk_us_site FOREIGN KEY (site_id) REFERENCES sites(id) ON DELETE CASCADE,
    INDEX idx_us_user (user_id),
    INDEX idx_us_site (site_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE stream_configs (
    id INT PRIMARY KEY AUTO_INCREMENT,
    group_id INT NOT NULL,
    site_id INT NOT NULL,
    stream_name VARCHAR(80) NOT NULL,
    video_url VARCHAR(255) NOT NULL,
    model_key VARCHAR(80) NOT NULL,
    detect_no_safety_vest_or_helmet BOOLEAN DEFAULT FALSE,
    detect_near_machinery_or_vehicle BOOLEAN DEFAULT FALSE,
    detect_in_restricted_area BOOLEAN DEFAULT FALSE,
    detect_in_utility_pole_restricted_area BOOLEAN DEFAULT FALSE,
    detect_machinery_close_to_pole BOOLEAN DEFAULT FALSE,
    detect_with_server BOOLEAN DEFAULT TRUE,
    expire_date DATETIME,
    work_start_hour INT,
    work_end_hour INT,
    store_in_redis BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_sc_group FOREIGN KEY (group_id) REFERENCES group_info(id) ON DELETE CASCADE,
    CONSTRAINT fk_sc_site  FOREIGN KEY (site_id)  REFERENCES sites(id)      ON DELETE CASCADE,
    UNIQUE KEY uq_sc_site_stream (site_id, stream_name),
    INDEX idx_sc_group (group_id),
    INDEX idx_sc_site (site_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- === Key point: violations.site (VARCHAR) FK to sites(name) ===
CREATE TABLE violations (
    id INT PRIMARY KEY AUTO_INCREMENT,
    stream_name VARCHAR(80) NOT NULL,
    detection_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    image_path VARCHAR(255) NOT NULL,
    detections_json TEXT,
    cone_polygon_json TEXT,
    pole_polygon_json TEXT,
    warnings_json TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    site VARCHAR(80) NOT NULL,
    CONSTRAINT fk_vio_site_name FOREIGN KEY (site) REFERENCES sites(name) ON DELETE CASCADE,
    INDEX idx_vio_site_name (site),
    INDEX idx_vio_time (detection_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

SET FOREIGN_KEY_CHECKS = 1;

-- ========== Seed Data ==========
-- Default group (ensure id=1 exists)
INSERT INTO group_info (id, name, uniform_number, max_allowed_streams)
VALUES (1, 'Default Group', '00000001', 8)
ON DUPLICATE KEY UPDATE
    name = VALUES(name),
    max_allowed_streams = VALUES(max_allowed_streams);

-- Ensure yolo_api exists in features (update description if already present)
INSERT INTO features (feature_name, description)
VALUES ('yolo_api', 'Utilising YOLO for real-time object detection.')
ON DUPLICATE KEY UPDATE
    description = VALUES(description);

-- Guest admin user (update if already present)
INSERT INTO users (username, password_hash, role, is_active, group_id)
VALUES (
    'user',
    'scrypt:32768:8:1$HP2pOGl5dSjKGax9$9a46ec70ddd6cca8400e712487fa30f005ae5d21786847d6f71bf3afeda3a8a5f68c7ea9a3f1cbd74c2b934f21110d071389b085aa1941e2db7b59304ef8f88f',
    'admin',
    TRUE,
    1
)
ON DUPLICATE KEY UPDATE
    password_hash = VALUES(password_hash),
    role = VALUES(role),
    is_active = VALUES(is_active),
    group_id = VALUES(group_id);

-- Enable yolo_api for default group (map by name to avoid hard-coded ID)
INSERT IGNORE INTO group_features (group_id, feature_id)
SELECT 1, f.id
FROM features f
WHERE f.feature_name = 'yolo_api';
