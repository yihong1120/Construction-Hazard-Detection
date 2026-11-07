BEGIN;
ALTER TABLE account_operations ADD COLUMN IF NOT EXISTS sync_parts json NOT NULL DEFAULT '{}'::json;
ALTER TABLE account_operations ADD COLUMN IF NOT EXISTS completed_at timestamptz;
UPDATE account_operations SET completed_at=now() WHERE phase='completed' AND completed_at IS NULL;
DROP INDEX IF EXISTS account_operation_due;
CREATE INDEX account_operation_due ON account_operations(next_attempt,created_at)
    WHERE phase NOT IN ('completed','uncertain');
CREATE INDEX IF NOT EXISTS account_operation_completed ON account_operations(completed_at)
    WHERE phase='completed';
CREATE TABLE IF NOT EXISTS account_deletion_receipts (
    user_id integer PRIMARY KEY, tenant_id uuid NOT NULL, group_id integer,
    username varchar(80) NOT NULL, role varchar(32) NOT NULL,
    completed_at timestamptz NOT NULL, expires_at timestamptz NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_account_deletion_receipts_expires_at
    ON account_deletion_receipts(expires_at);
INSERT INTO account_deletion_receipts(user_id,tenant_id,group_id,username,role,completed_at,expires_at)
    SELECT DISTINCT ON (user_id) user_id,tenant_id,group_id,username,role,completed_at,now()+interval '365 days'
    FROM account_operations WHERE kind='delete' AND phase='completed' AND user_id IS NOT NULL
    ORDER BY user_id,completed_at DESC
    ON CONFLICT(user_id) DO NOTHING;
COMMIT;
