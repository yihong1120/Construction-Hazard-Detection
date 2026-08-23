-- Durable post-commit cleanup queue for evidence images deleted with a site.
BEGIN;

CREATE TABLE IF NOT EXISTS site_media_cleanup_jobs (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    path VARCHAR(1024) NOT NULL UNIQUE,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    lease_token VARCHAR(36),
    lease_expires_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_site_media_cleanup_jobs_pending
    ON site_media_cleanup_jobs (completed_at, lease_expires_at, id)
    WHERE completed_at IS NULL;

COMMIT;
