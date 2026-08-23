BEGIN;

-- Do not silently merge accounts.  Resolve any historical case-only collision
-- before running this migration, then all persisted addresses are canonical.
DO $$
BEGIN
    IF EXISTS (
        SELECT lower(btrim(email))
        FROM user_profiles
        GROUP BY lower(btrim(email))
        HAVING count(*) > 1
    ) THEN
        RAISE EXCEPTION
            'Cannot normalize user profile emails: case-insensitive duplicates exist';
    END IF;
END
$$;

UPDATE user_profiles
SET email = lower(btrim(email))
WHERE email <> lower(btrim(email));

CREATE UNIQUE INDEX IF NOT EXISTS uq_user_profiles_email_lower
    ON user_profiles (lower(email));

COMMIT;
