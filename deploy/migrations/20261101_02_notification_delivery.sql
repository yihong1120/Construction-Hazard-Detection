BEGIN;
ALTER TABLE sites ADD COLUMN IF NOT EXISTS recipient_revision bigint NOT NULL DEFAULT 0;
CREATE OR REPLACE FUNCTION bump_notification_recipients() RETURNS trigger
LANGUAGE plpgsql SET search_path FROM CURRENT AS $$
BEGIN
    IF TG_TABLE_NAME = 'users' THEN
        UPDATE sites SET recipient_revision=recipient_revision+1
        WHERE id IN (SELECT site_id FROM site_notification_preferences WHERE user_id=NEW.id);
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE sites SET recipient_revision=recipient_revision+1 WHERE id=OLD.site_id;
    ELSIF TG_OP = 'INSERT' THEN
        UPDATE sites SET recipient_revision=recipient_revision+1 WHERE id=NEW.site_id;
    ELSE
        UPDATE sites SET recipient_revision=recipient_revision+1 WHERE id IN (OLD.site_id,NEW.site_id);
    END IF;
    RETURN NULL;
END;
$$;
DROP TRIGGER IF EXISTS notification_recipients_changed ON site_notification_preferences;
CREATE TRIGGER notification_recipients_changed AFTER INSERT OR UPDATE OR DELETE
ON site_notification_preferences FOR EACH ROW EXECUTE FUNCTION bump_notification_recipients();
DROP TRIGGER IF EXISTS notification_recipients_changed ON user_sites;
CREATE TRIGGER notification_recipients_changed AFTER INSERT OR UPDATE OR DELETE
ON user_sites FOR EACH ROW EXECUTE FUNCTION bump_notification_recipients();
DROP TRIGGER IF EXISTS notification_recipients_changed ON site_groups;
CREATE TRIGGER notification_recipients_changed AFTER INSERT OR UPDATE OR DELETE
ON site_groups FOR EACH ROW EXECUTE FUNCTION bump_notification_recipients();
DROP TRIGGER IF EXISTS notification_recipients_changed ON users;
CREATE TRIGGER notification_recipients_changed AFTER UPDATE OF status,role,group_id
ON users FOR EACH ROW EXECUTE FUNCTION bump_notification_recipients();
CREATE TABLE IF NOT EXISTS notification_delivery_receipts (
    event_digest varchar(64) NOT NULL,
    user_id integer NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash varchar(64) NOT NULL,
    registration_at timestamptz NOT NULL,
    delivered_at timestamptz NOT NULL,
    expires_at timestamptz NOT NULL,
    PRIMARY KEY(event_digest,user_id,token_hash,registration_at)
);
CREATE INDEX IF NOT EXISTS notification_receipts_expiry ON notification_delivery_receipts(expires_at);
COMMIT;
