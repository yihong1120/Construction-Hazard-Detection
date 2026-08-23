-- Immutable compatibility repair: legacy databases may lack the lease fields
-- despite the final 20260824 definition. New bootstrap databases already have
-- this end state, so the conditional ALTER statements become no-ops.
BEGIN;

ALTER TABLE site_media_cleanup_jobs
    ADD COLUMN IF NOT EXISTS lease_token VARCHAR(36),
    ADD COLUMN IF NOT EXISTS lease_expires_at TIMESTAMP WITH TIME ZONE;

DROP INDEX IF EXISTS idx_site_media_cleanup_jobs_pending;
CREATE INDEX idx_site_media_cleanup_jobs_pending
    ON site_media_cleanup_jobs (completed_at, lease_expires_at, id)
    WHERE completed_at IS NULL;

COMMIT;
