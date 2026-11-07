BEGIN;
CREATE TABLE IF NOT EXISTS detection_model_versions (
    model_key varchar(120) NOT NULL,
    version text NOT NULL,
    sha256 varchar(64) NOT NULL,
    classes json NOT NULL,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(model_key,version),
    CONSTRAINT model_version_sha256_length CHECK(length(sha256)=64)
);
CREATE OR REPLACE FUNCTION remember_detection_model_version(
    key text, definition json
) RETURNS void LANGUAGE plpgsql SET search_path FROM CURRENT AS $$
DECLARE
    previous detection_model_versions%ROWTYPE;
BEGIN
    INSERT INTO detection_model_versions(model_key,version,sha256,classes)
    VALUES(key,definition->>'version',definition->>'sha256',
           definition->'classes')
    ON CONFLICT(model_key,version) DO NOTHING;
    SELECT * INTO STRICT previous FROM detection_model_versions
    WHERE model_key=key AND version=definition->>'version';
    IF previous.sha256 <> definition->>'sha256'
       OR previous.classes::jsonb <> (definition->'classes')::jsonb THEN
        RAISE EXCEPTION 'Model version metadata is immutable: %/%',
            key,definition->>'version' USING ERRCODE='23514';
    END IF;
END;
$$;
CREATE OR REPLACE FUNCTION archive_detection_model_version()
RETURNS trigger LANGUAGE plpgsql SET search_path FROM CURRENT AS $$
BEGIN
    IF TG_OP <> 'INSERT' THEN
        PERFORM remember_detection_model_version(OLD.model_key,OLD.definition);
    END IF;
    IF TG_OP <> 'DELETE' THEN
        PERFORM remember_detection_model_version(NEW.model_key,NEW.definition);
        RETURN NEW;
    END IF;
    RETURN OLD;
END;
$$;
CREATE OR REPLACE FUNCTION protect_detection_model_version()
RETURNS trigger LANGUAGE plpgsql SET search_path FROM CURRENT AS $$
BEGIN
    RAISE EXCEPTION 'Model version history is immutable'
        USING ERRCODE='23514';
END;
$$;
DROP TRIGGER IF EXISTS archive_model_version ON detection_model_catalog;
CREATE TRIGGER archive_model_version BEFORE INSERT OR UPDATE OR DELETE
ON detection_model_catalog FOR EACH ROW
EXECUTE FUNCTION archive_detection_model_version();
DROP TRIGGER IF EXISTS protect_model_version ON detection_model_versions;
CREATE TRIGGER protect_model_version BEFORE UPDATE OR DELETE
ON detection_model_versions FOR EACH ROW
EXECUTE FUNCTION protect_detection_model_version();
DO $$
DECLARE item record; candidate jsonb;
BEGIN
    FOR item IN SELECT model_key,definition FROM detection_model_catalog LOOP
        PERFORM remember_detection_model_version(
            item.model_key,item.definition);
    END LOOP;
    FOR item IN SELECT snapshot FROM model_publications LOOP
        FOR candidate IN
            SELECT jsonb_build_object('id',key) || (value->'definition')
            FROM jsonb_each(coalesce(item.snapshot::jsonb->'previous','{}'))
            UNION ALL
            SELECT value->0 FROM jsonb_array_elements(
                coalesce(item.snapshot::jsonb->'prepared','[]'))
        LOOP
            PERFORM remember_detection_model_version(
                candidate->>'id',candidate::json);
        END LOOP;
    END LOOP;
END;
$$;
COMMIT;
