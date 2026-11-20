CREATE SCHEMA IF NOT EXISTS rust_api;

CREATE TABLE IF NOT EXISTS rust_api.evidence (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_id uuid NOT NULL,
    site_id bigint NOT NULL CHECK (site_id > 0),
    site text NOT NULL CHECK (char_length(site) BETWEEN 1 AND 80),
    metadata jsonb NOT NULL,
    image bytea NOT NULL CHECK (octet_length(image) BETWEEN 1 AND 8388608),
    image_sha256 bytea NOT NULL CHECK (octet_length(image_sha256) = 32),
    width integer NOT NULL CHECK (width > 0),
    height integer NOT NULL CHECK (height > 0),
    received_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (site_id, event_id)
);

CREATE INDEX IF NOT EXISTS evidence_site_id_idx
    ON rust_api.evidence (site_id, site, id DESC);
