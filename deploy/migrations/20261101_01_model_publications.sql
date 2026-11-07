BEGIN;
CREATE TABLE IF NOT EXISTS model_publications (
    id varchar(32) PRIMARY KEY,
    phase varchar(32) NOT NULL,
    snapshot json NOT NULL,
    updated_at varchar(40) NOT NULL
);
COMMIT;
