-- Add per-user site notification preferences.
-- Safe for MySQL 8.0+.

USE construction_hazard_detection;

CREATE TABLE IF NOT EXISTS site_notification_preferences (
    user_id INT NOT NULL,
    site_id INT NOT NULL,
    is_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, site_id),
    CONSTRAINT fk_snp_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT fk_snp_site FOREIGN KEY (site_id) REFERENCES sites(id) ON DELETE CASCADE,
    INDEX idx_snp_site_enabled (site_id, is_enabled),
    INDEX idx_snp_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
