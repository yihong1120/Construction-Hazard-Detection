-- Support keyset pagination for notification-centre history.
CREATE INDEX IF NOT EXISTS idx_notifications_user_created_id
    ON notifications (user_id, created_at DESC, id DESC);

CREATE INDEX IF NOT EXISTS idx_notifications_user_read_created_id
    ON notifications (user_id, is_read, created_at DESC, id DESC);

CREATE INDEX IF NOT EXISTS idx_notifications_user_type_created_id
    ON notifications (user_id, type, created_at DESC, id DESC);
