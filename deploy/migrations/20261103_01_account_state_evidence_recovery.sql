BEGIN;
ALTER TABLE users ADD COLUMN IF NOT EXISTS state_revision integer NOT NULL DEFAULT 1;
ALTER TABLE account_operations ADD COLUMN IF NOT EXISTS target_revision integer;
ALTER TABLE evidence_write_operations ADD COLUMN IF NOT EXISTS attempts integer NOT NULL DEFAULT 0;
ALTER TABLE evidence_write_operations ADD COLUMN IF NOT EXISTS next_attempt timestamptz NOT NULL DEFAULT now();
ALTER TABLE evidence_write_operations ADD COLUMN IF NOT EXISTS last_error varchar(80);
DROP INDEX IF EXISTS evidence_write_due;
CREATE INDEX evidence_write_due ON evidence_write_operations(next_attempt,created_at);
COMMIT;
