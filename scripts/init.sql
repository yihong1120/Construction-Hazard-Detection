
-- =============================================
-- Construction-Hazard-Detection Database Schema
-- Generated from SQLAlchemy models.py
-- =============================================

-- This schema covers all major entities:
--   - features: System features (e.g., detection capabilities)
--   - group_info: User groups and permissions
--   - group_features: Many-to-many between groups and features
--   - sites: Construction sites
--   - user_profiles: User profile details
--   - users: System users and credentials
--   - user_sites: Many-to-many between users and sites
--   - stream_configs: Video stream configuration
--   - violations: Safety violation records


CREATE TABLE features (
    id INT PRIMARY KEY AUTO_INCREMENT, -- Feature ID
    feature_name VARCHAR(50) UNIQUE NOT NULL, -- Unique feature name
    description TEXT, -- Feature description
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP, -- Creation timestamp
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP -- Last update timestamp
);

CREATE TABLE group_info (
    id INT PRIMARY KEY AUTO_INCREMENT, -- Group ID
    name VARCHAR(100) NOT NULL, -- Group name
    uniform_number VARCHAR(8) UNIQUE NOT NULL COMMENT '統一編號', -- Unique identifier
    max_allowed_streams INT NOT NULL DEFAULT 8, -- Max allowed streams per group
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP, -- Creation timestamp
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP -- Last update timestamp
);

CREATE TABLE group_features (
    group_id INT NOT NULL, -- FK to group_info
    feature_id INT NOT NULL, -- FK to features
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP, -- Link creation timestamp
    PRIMARY KEY (group_id, feature_id), -- Composite PK
    FOREIGN KEY (group_id) REFERENCES group_info(id) ON DELETE CASCADE, -- Cascade delete
    FOREIGN KEY (feature_id) REFERENCES features(id) ON DELETE CASCADE -- Cascade delete
);

CREATE TABLE sites (
    id INT PRIMARY KEY AUTO_INCREMENT, -- Site ID
    name VARCHAR(80) NOT NULL, -- Site name
    group_id INT, -- FK to group_info
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, -- Creation timestamp
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP, -- Last update timestamp
    FOREIGN KEY (group_id) REFERENCES group_info(id) ON DELETE SET NULL -- Set group_id to NULL on group delete
);

CREATE TABLE user_profiles (
    user_id INT PRIMARY KEY, -- FK to users
    family_name VARCHAR(50) NOT NULL, -- Family name
    middle_name VARCHAR(50), -- Middle name (optional)
    given_name VARCHAR(50) NOT NULL, -- Given name
    email VARCHAR(255) UNIQUE NOT NULL, -- Unique email
    mobile_number VARCHAR(20) UNIQUE, -- Unique mobile number
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP, -- Creation timestamp
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP, -- Last update timestamp
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE -- Cascade delete
);

CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT, -- User ID
    username VARCHAR(80) UNIQUE NOT NULL, -- Unique username
    password_hash VARCHAR(255) NOT NULL, -- Hashed password
    role VARCHAR(20) NOT NULL DEFAULT 'user', -- User role
    is_active BOOLEAN NOT NULL DEFAULT TRUE, -- Active status
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, -- Creation timestamp
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP, -- Last update timestamp
    group_id INT, -- FK to group_info
    FOREIGN KEY (group_id) REFERENCES group_info(id) ON DELETE SET NULL -- Set group_id to NULL on group delete
);

CREATE TABLE user_sites (
    user_id INT NOT NULL, -- FK to users
    site_id INT NOT NULL, -- FK to sites
    PRIMARY KEY (user_id, site_id), -- Composite PK
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE, -- Cascade delete
    FOREIGN KEY (site_id) REFERENCES sites(id) ON DELETE CASCADE -- Cascade delete
);

CREATE TABLE stream_configs (
    id INT PRIMARY KEY AUTO_INCREMENT, -- Stream config ID
    group_id INT NOT NULL, -- FK to group_info
    site_id INT NOT NULL, -- FK to sites
    stream_name VARCHAR(80) NOT NULL, -- Stream name
    video_url VARCHAR(255) NOT NULL, -- Video stream URL
    model_key VARCHAR(80) NOT NULL, -- Detection model key
    detect_no_safety_vest_or_helmet BOOLEAN DEFAULT FALSE, -- Detect vest/helmet
    detect_near_machinery_or_vehicle BOOLEAN DEFAULT FALSE, -- Detect near machinery/vehicle
    detect_in_restricted_area BOOLEAN DEFAULT FALSE, -- Detect in restricted area
    detect_in_utility_pole_restricted_area BOOLEAN DEFAULT FALSE, -- Detect in utility pole area
    detect_machinery_close_to_pole BOOLEAN DEFAULT FALSE, -- Detect machinery close to pole
    detect_with_server BOOLEAN DEFAULT TRUE, -- Use server-side detection
    expire_date DATETIME, -- Expiration date
    work_start_hour INT, -- Work start hour
    work_end_hour INT, -- Work end hour
    store_in_redis BOOLEAN DEFAULT FALSE, -- Store in Redis
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP, -- Creation timestamp
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP, -- Last update timestamp
    FOREIGN KEY (group_id) REFERENCES group_info(id) ON DELETE CASCADE, -- Cascade delete
    FOREIGN KEY (site_id) REFERENCES sites(id) ON DELETE CASCADE, -- Cascade delete
    UNIQUE (site_id, stream_name) -- Unique constraint for site/stream
);

CREATE TABLE violations (
    id INT PRIMARY KEY AUTO_INCREMENT, -- Violation ID
    stream_name VARCHAR(80) NOT NULL, -- Stream/camera name
    detection_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, -- Detection timestamp
    image_path VARCHAR(255) NOT NULL, -- Path to violation image
    detections_json TEXT, -- JSON of detected objects
    cone_polygon_json TEXT, -- JSON of cone polygons
    pole_polygon_json TEXT, -- JSON of pole polygons
    warnings_json TEXT, -- JSON of warning content
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP, -- Record creation timestamp
    site VARCHAR(80) NOT NULL, -- Site name (FK to sites)
    FOREIGN KEY (site) REFERENCES sites(name) -- Link to sites table
);
