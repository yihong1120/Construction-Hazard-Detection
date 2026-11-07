BEGIN;
ALTER TABLE users ADD COLUMN IF NOT EXISTS deletion_pending boolean NOT NULL DEFAULT false;
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid='users'::regclass AND conname='user_pending_deletion_suspended') THEN
        ALTER TABLE users ADD CONSTRAINT user_pending_deletion_suspended
        CHECK (NOT deletion_pending OR status='suspended');
    END IF;
END $$;
ALTER TABLE violations ADD COLUMN IF NOT EXISTS event_fingerprint varchar(64);
CREATE TABLE IF NOT EXISTS account_operations (
    id varchar(32) PRIMARY KEY, kind varchar(20) NOT NULL,
    phase varchar(32) NOT NULL, user_id integer, tenant_id uuid NOT NULL,
    username varchar(80) NOT NULL, group_id integer,
    role varchar(32) NOT NULL DEFAULT 'user', subject varchar(255),
    attempts integer NOT NULL DEFAULT 0, last_error varchar(80),
    created_at timestamptz NOT NULL DEFAULT now(),
    next_attempt timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS account_operation_due ON account_operations(next_attempt,created_at);
CREATE UNIQUE INDEX IF NOT EXISTS account_deletion_one_pending ON account_operations(user_id)
    WHERE kind='delete' AND phase<>'completed';
CREATE TABLE IF NOT EXISTS evidence_write_operations (
    id varchar(32) PRIMARY KEY, path text UNIQUE NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS evidence_write_due ON evidence_write_operations(created_at);
COMMIT;
