-- Repeat until remaining = 0. Each transaction locks at most 5,000 records.
BEGIN;
SET LOCAL lock_timeout = '5s';
WITH batch AS (
    SELECT v.id, s.id AS site_id
    FROM violations v JOIN sites s ON s.name = v.site
    WHERE v.site_id IS NULL
    ORDER BY v.id
    LIMIT 5000
    FOR UPDATE OF v SKIP LOCKED
)
UPDATE violations v SET site_id = batch.site_id
FROM batch WHERE v.id = batch.id;
COMMIT;
SELECT count(*) AS remaining FROM violations WHERE site_id IS NULL;
