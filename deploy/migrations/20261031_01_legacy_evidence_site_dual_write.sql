-- Apply after 20261030_01 when old writers still omit site_id.
-- Bind both relations to their actual schemas, not the caller's search_path.
BEGIN;
SET LOCAL lock_timeout = '5s';
DO $migration$
DECLARE
    evidence_schema TEXT;
    site_schema TEXT;
BEGIN
    SELECT n.nspname INTO evidence_schema
    FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE c.oid = 'violations'::regclass;
    SELECT n.nspname INTO site_schema
    FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE c.oid = 'sites'::regclass;

    EXECUTE format($ddl$
        CREATE OR REPLACE FUNCTION %I.fill_legacy_evidence_site_id()
        RETURNS trigger LANGUAGE plpgsql AS $function$
        BEGIN
            IF NEW.site_id IS NULL THEN
                SELECT s.id INTO NEW.site_id FROM %I.sites s
                WHERE s.name = NEW.site;
                IF NEW.site_id IS NULL THEN
                    RAISE foreign_key_violation USING
                        MESSAGE = 'Evidence site name has no matching site ID',
                        CONSTRAINT = 'fk_violations_site_id';
                END IF;
            END IF;
            RETURN NEW;
        END;
        $function$
    $ddl$, evidence_schema, site_schema);

    EXECUTE format(
        'CREATE OR REPLACE TRIGGER fill_legacy_evidence_site_id '
        'BEFORE INSERT OR UPDATE OF site, site_id ON %s '
        'FOR EACH ROW EXECUTE FUNCTION %I.fill_legacy_evidence_site_id()',
        'violations'::regclass, evidence_schema
    );
END;
$migration$;
COMMIT;
