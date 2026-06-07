-- Migration: Seed site_notification_preferences for all existing relationships
-- Date: 2026-03-27
--
-- Before this migration, SiteNotificationPreference rows were only created
-- on-demand when a user explicitly updated their preferences.  This migration
-- pre-seeds a default enabled preference for every eligible (user, site) pair

--
-- Two eligibility paths are handled:
--   1. Direct access: user is in user_sites for the site.
--   2. Group access:  user.group_id matches a group in site_groups for the site.

-- Path 1: Direct user-site access (user_sites table)
INSERT IGNORE INTO site_notification_preferences (user_id, site_id, is_enabled, created_at, updated_at)
SELECT
    us.user_id,
    us.site_id,
    1,
    NOW(),
    NOW()
FROM user_sites us
JOIN users u ON u.id = us.user_id
WHERE u.status = 'active';

-- Path 2: Group-based access (user.group_id → site_groups → sites)
INSERT IGNORE INTO site_notification_preferences (user_id, site_id, is_enabled, created_at, updated_at)
SELECT
    u.id   AS user_id,
    sg.site_id,
    1,
    NOW(),
    NOW()
FROM users u
JOIN site_groups sg ON sg.group_id = u.group_id
WHERE u.status = 'active'
  AND u.group_id IS NOT NULL;

-- Verify: count of seeded rows (informational)
SELECT COUNT(*) AS total_seeded FROM site_notification_preferences;
