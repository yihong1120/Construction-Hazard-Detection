ALTER TABLE users
ADD COLUMN status VARCHAR(20) NOT NULL DEFAULT 'active' AFTER role;

UPDATE users
SET status = CASE
    WHEN is_active = TRUE THEN 'active'
    ELSE 'inactive'
END;

ALTER TABLE users
DROP COLUMN is_active;
