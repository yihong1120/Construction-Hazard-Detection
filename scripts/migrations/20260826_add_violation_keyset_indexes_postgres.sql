-- Run with psql -f; CREATE INDEX CONCURRENTLY may not run inside BEGIN/COMMIT.
-- These indexes match violation list keyset ordering exactly.

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vio_site_detection_id
    ON violations (site, detection_time DESC, id DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vio_stream_config_detection_id
    ON violations (stream_config_id, detection_time DESC, id DESC);
