BEGIN;
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_attribute
        WHERE attrelid='users'::regclass AND attname='status' AND NOT attisdropped
          AND atttypid='varchar'::regtype AND atttypmod BETWEEN 4 AND 35
    ) THEN
        -- PostgreSQL requires removing the UPDATE OF trigger before widening
        -- its column. Recreate the same notification invalidation atomically.
        DROP TRIGGER notification_recipients_changed ON users;
        ALTER TABLE users ALTER COLUMN status TYPE varchar(32);
        CREATE TRIGGER notification_recipients_changed AFTER UPDATE OF status,role,group_id
        ON users FOR EACH ROW EXECUTE FUNCTION bump_notification_recipients();
    END IF;
END;
$$;
COMMIT;
