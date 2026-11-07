-- Apply before deploying code that writes site_id/event_id/event_key.
-- No existing name constraint is removed during expansion.
BEGIN;
SET LOCAL lock_timeout = '5s';
ALTER TABLE violations ADD COLUMN IF NOT EXISTS site_id INTEGER;
ALTER TABLE violations ADD COLUMN IF NOT EXISTS event_id VARCHAR(36);
ALTER TABLE notifications ADD COLUMN IF NOT EXISTS event_key VARCHAR(255);
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid = 'violations'::regclass AND conname = 'fk_violations_site_id') THEN
        ALTER TABLE violations ADD CONSTRAINT fk_violations_site_id
            FOREIGN KEY (site_id) REFERENCES sites(id) ON DELETE CASCADE NOT VALID;
    END IF;
END $$;
CREATE UNIQUE INDEX IF NOT EXISTS uq_violation_event_id ON violations(event_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_notification_user_event ON notifications(user_id, event_key);
CREATE INDEX IF NOT EXISTS idx_vio_site_id_detection_id ON violations(site_id, detection_time DESC, id DESC);
COMMIT;
