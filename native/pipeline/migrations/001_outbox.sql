CREATE SCHEMA IF NOT EXISTS rust_pipeline;
CREATE TABLE IF NOT EXISTS rust_pipeline.outbox_namespaces (
    namespace TEXT PRIMARY KEY CHECK (length(namespace) BETWEEN 1 AND 256),
    payload_bytes BIGINT NOT NULL DEFAULT 0 CHECK (payload_bytes >= 0),
    byte_limit BIGINT NOT NULL CHECK (byte_limit >= 1024)
);
CREATE TABLE IF NOT EXISTS rust_pipeline.outbox_events (
    namespace TEXT NOT NULL REFERENCES rust_pipeline.outbox_namespaces(namespace),
    event_id TEXT NOT NULL,
    metadata JSONB NOT NULL CHECK (jsonb_typeof(metadata) = 'object'
        AND metadata ? 'event_id' AND metadata->>'event_id' = event_id),
    image BYTEA,
    payload_bytes BIGINT NOT NULL CHECK (payload_bytes >= 0),
    violation_id TEXT,
    state TEXT NOT NULL DEFAULT 'pending' CHECK (state IN ('pending','inflight','failed','done')),
    attempts BIGINT NOT NULL DEFAULT 0 CHECK (attempts BETWEEN 0 AND 4294967295),
    next_at BIGINT NOT NULL DEFAULT 0,
    lease_until BIGINT NOT NULL DEFAULT 0,
    claim_token TEXT,
    last_error TEXT,
    created_at BIGINT NOT NULL,
    PRIMARY KEY (namespace,event_id),
    CHECK ((state = 'done') = (image IS NULL)),
    CHECK ((state = 'inflight') = (claim_token IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS outbox_pending_due
    ON rust_pipeline.outbox_events(namespace,next_at,created_at,event_id) WHERE state='pending';
CREATE INDEX IF NOT EXISTS outbox_expired_leases
    ON rust_pipeline.outbox_events(namespace,lease_until,created_at,event_id) WHERE state='inflight';
CREATE INDEX IF NOT EXISTS outbox_done_archive
    ON rust_pipeline.outbox_events(namespace,created_at DESC,event_id DESC) WHERE state='done';
CREATE TABLE IF NOT EXISTS rust_pipeline.outbox_imports (
    namespace TEXT NOT NULL REFERENCES rust_pipeline.outbox_namespaces(namespace),
    source_key TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    imported_rows BIGINT NOT NULL,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY(namespace,source_key)
);
