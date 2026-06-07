-- Replace the default seeded user's Werkzeug/scrypt hash with Argon2.
-- Password remains the documented local default: password.

USE construction_hazard_detection;

UPDATE users
SET password_hash = '$argon2id$v=19$m=65536,t=3,p=4$WWrgNzRESjrJxeP6KC+jsQ$LRWIP3bk3vAJf5kSEA+gkSk1+KYvVU2VDwCKGiUtBCg'
WHERE username = 'user';
