-- Website network policy is global, independent of construction Site records.
BEGIN;
CREATE TABLE IF NOT EXISTS website_access_policies (
    site_key VARCHAR(32) PRIMARY KEY,
    restricted BOOLEAN NOT NULL DEFAULT FALSE,
    version INTEGER NOT NULL DEFAULT 0 CHECK (version >= 0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (site_key IN ('main', 'ai', 'wiki'))
);
CREATE TABLE IF NOT EXISTS website_access_rules (
    id UUID PRIMARY KEY,
    site_key VARCHAR(32) NOT NULL REFERENCES website_access_policies(site_key) ON DELETE CASCADE,
    network VARCHAR(43) NOT NULL,
    label VARCHAR(120) NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_website_access_network UNIQUE (site_key, network)
);
CREATE TABLE IF NOT EXISTS website_access_audit_logs (
    id UUID PRIMARY KEY,
    site_key VARCHAR(32) NOT NULL REFERENCES website_access_policies(site_key),
    actor_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    actor_ip VARCHAR(45) NOT NULL,
    action VARCHAR(32) NOT NULL,
    before JSON NOT NULL,
    after JSON NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_website_access_audit_logs_created_at
    ON website_access_audit_logs(created_at);
INSERT INTO website_access_policies (site_key) VALUES ('main'), ('ai'), ('wiki')
    ON CONFLICT (site_key) DO NOTHING;
COMMIT;
