-- Run only after every writer dual-writes IDs, backfill completes, and the
-- new application has been verified. Names remain immutable evidence snapshots.
BEGIN;
SET LOCAL lock_timeout = '5s';
DO $$ DECLARE old_fk RECORD; BEGIN
    IF EXISTS (SELECT 1 FROM violations WHERE site_id IS NULL) THEN
        RAISE EXCEPTION 'Backfill incomplete; site_id remains NULL';
    END IF;
    FOR old_fk IN
        SELECT c.conname FROM pg_constraint c
        JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = ANY(c.conkey)
        WHERE c.conrelid = 'violations'::regclass AND c.contype = 'f'
          AND c.confrelid = 'sites'::regclass AND a.attname = 'site'
    LOOP
        EXECUTE format('ALTER TABLE violations DROP CONSTRAINT %I', old_fk.conname);
    END LOOP;
END $$;
ALTER TABLE violations VALIDATE CONSTRAINT fk_violations_site_id;
COMMIT;
