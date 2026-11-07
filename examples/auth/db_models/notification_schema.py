from __future__ import annotations

from sqlalchemy import inspect

RECIPIENT_SCHEMA = (
    '''
CREATE OR REPLACE FUNCTION bump_notification_recipients() RETURNS
trigger
LANGUAGE plpgsql SET search_path FROM CURRENT AS $$
BEGIN
    IF TG_TABLE_NAME = 'users' THEN
        UPDATE sites SET recipient_revision=recipient_revision+1
        WHERE id IN (SELECT site_id FROM
site_notification_preferences WHERE user_id=NEW.id);
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE sites SET recipient_revision=recipient_revision+1
WHERE id=OLD.site_id;
    ELSIF TG_OP = 'INSERT' THEN
        UPDATE sites SET recipient_revision=recipient_revision+1
WHERE id=NEW.site_id;
    ELSE
        UPDATE sites SET recipient_revision=recipient_revision+1
WHERE id IN (OLD.site_id,NEW.site_id);
    END IF;
    RETURN NULL;
END;
$$
''',
    '''
DROP TRIGGER IF EXISTS notification_recipients_changed ON
site_notification_preferences
''',
    '''
CREATE TRIGGER notification_recipients_changed AFTER INSERT OR
UPDATE OR DELETE
ON site_notification_preferences FOR EACH ROW EXECUTE FUNCTION
bump_notification_recipients()
''',
    '''
DROP TRIGGER IF EXISTS notification_recipients_changed ON user_sites
''',
    '''
CREATE TRIGGER notification_recipients_changed AFTER INSERT OR
UPDATE OR DELETE
ON user_sites FOR EACH ROW EXECUTE FUNCTION
bump_notification_recipients()
''',
    '''
DROP TRIGGER IF EXISTS notification_recipients_changed ON
site_groups
''',
    '''
CREATE TRIGGER notification_recipients_changed AFTER INSERT OR
UPDATE OR DELETE
ON site_groups FOR EACH ROW EXECUTE FUNCTION
bump_notification_recipients()
''',
    '''
DROP TRIGGER IF EXISTS notification_recipients_changed ON users
''',
    '''
CREATE TRIGGER notification_recipients_changed AFTER UPDATE OF
status,role,group_id
ON users FOR EACH ROW EXECUTE FUNCTION
bump_notification_recipients()
''',
)


def install_recipient_versions(metadata, connection, **kwargs):
    if connection.dialect.name != 'postgresql':
        return
    required = {
        'sites',
        'users',
        'site_notification_preferences',
        'user_sites',
        'site_groups',
    }
    if not required.issubset(inspect(connection).get_table_names()):
        return
    for statement in RECIPIENT_SCHEMA:
        connection.exec_driver_sql(statement)
