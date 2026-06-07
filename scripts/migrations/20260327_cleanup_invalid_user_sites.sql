DELETE snp
FROM site_notification_preferences snp
JOIN users u ON u.id = snp.user_id
LEFT JOIN site_groups sg
    ON sg.site_id = snp.site_id
    AND sg.group_id = u.group_id
WHERE u.role <> 'super_admin'
  AND (u.group_id IS NULL OR sg.site_id IS NULL);

DELETE us
FROM user_sites us
JOIN users u ON u.id = us.user_id
LEFT JOIN site_groups sg
    ON sg.site_id = us.site_id
    AND sg.group_id = u.group_id
WHERE u.role <> 'super_admin'
  AND (u.group_id IS NULL OR sg.site_id IS NULL);
