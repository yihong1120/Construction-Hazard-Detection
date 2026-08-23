from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.models import User
from examples.local_notification_server.routers import (
    list_site_notification_preferences,
)
from examples.local_notification_server.routers import (
    update_site_notification_preferences,
)
from examples.local_notification_server.schemas import (
    SiteNotificationPreferenceIn,
)
from examples.local_notification_server.schemas import (
    SiteNotificationPreferenceOut,
)
from examples.local_notification_server.schemas import (
    SiteNotificationPreferenceUpdateRequest,
)


class TestSiteNotificationPreferences(unittest.IsolatedAsyncioTestCase):
    """Test suite for site notification preference endpoints."""

    def setUp(self) -> None:
        """Prepare common test mocks for each test case."""
        self.db: AsyncMock = AsyncMock(spec=AsyncSession)
        self.user: MagicMock = MagicMock(spec=User)
        self.user.role = 'admin'
        self.user.group_id = 1

    @patch(
        'examples.local_notification_server.site_preference_service.'
        'list_sites',
    )
    @patch(
        'examples.local_notification_server.site_preference_service.'
        'list_effective_sites_for_user',
        new_callable=AsyncMock,
    )
    async def test_list_site_notification_preferences(
        self,
        mock_list_effective_sites_for_user: AsyncMock,
        mock_list_sites: MagicMock,
    ) -> None:
        """List current notification subscriptions with retry defaults."""
        group = MagicMock(name='GroupA')
        group.name = 'GroupA'
        site1 = MagicMock()
        site1.id = 1
        site1.name = 'Site1'
        site1.groups = [group]
        site2 = MagicMock()
        site2.id = 2
        site2.name = 'Site2'
        site2.groups = [group]
        self.user.id = 9
        mock_list_sites.return_value = [site1, site2]
        mock_list_effective_sites_for_user.return_value = [site2]

        pref_result = MagicMock()
        pref_result.all.return_value = [(1, False)]
        self.db.execute.side_effect = [pref_result]

        result = await list_site_notification_preferences(self.db, self.user)

        self.assertEqual(
            result,
            [
                SiteNotificationPreferenceOut(
                    site_id=1,
                    site_name='Site1',
                    group_name='GroupA',
                    is_enabled=False,
                ),
                SiteNotificationPreferenceOut(
                    site_id=2,
                    site_name='Site2',
                    group_name='GroupA',
                    is_enabled=True,
                ),
            ],
        )

    @patch(
        'examples.local_notification_server.site_preference_service.'
        'refresh_site_notification_user_cache',
        new_callable=AsyncMock,
    )
    @patch(
        'examples.local_notification_server.site_preference_service.'
        'list_sites',
    )
    @patch(
        'examples.local_notification_server.site_preference_service.'
        'list_effective_sites_for_user',
        new_callable=AsyncMock,
    )
    async def test_update_site_notification_preferences(
        self,
        mock_list_effective_sites_for_user: AsyncMock,
        mock_list_sites: MagicMock,
        mock_refresh_site_cache: AsyncMock,
    ) -> None:
        """Replace notification subscriptions and invalidate changed sites."""
        group = MagicMock(name='GroupA')
        group.name = 'GroupA'
        site1 = MagicMock()
        site1.id = 1
        site1.name = 'Site1'
        site1.groups = [group]
        site2 = MagicMock()
        site2.id = 2
        site2.name = 'Site2'
        site2.groups = [group]
        self.user.id = 5
        mock_list_sites.return_value = [site1, site2]
        mock_list_effective_sites_for_user.return_value = [site2]

        pref = MagicMock(site_id=1, is_enabled=False)
        pref_result = MagicMock()
        pref_result.scalars.return_value.all.return_value = [pref]
        refreshed_pref_result = MagicMock()
        refreshed_pref_result.all.return_value = [(1, True), (2, False)]
        self.db.execute.side_effect = [
            pref_result,
            refreshed_pref_result,
        ]
        mock_redis = AsyncMock()

        result = await update_site_notification_preferences(
            SiteNotificationPreferenceUpdateRequest(
                preferences=[
                    SiteNotificationPreferenceIn(
                        site_id=1,
                        is_enabled=True,
                    ),
                    SiteNotificationPreferenceIn(
                        site_id=2,
                        is_enabled=False,
                    ),
                ],
            ),
            self.db,
            self.user,
            mock_redis,
        )

        self.assertEqual(
            result,
            [
                SiteNotificationPreferenceOut(
                    site_id=1,
                    site_name='Site1',
                    group_name='GroupA',
                    is_enabled=True,
                ),
                SiteNotificationPreferenceOut(
                    site_id=2,
                    site_name='Site2',
                    group_name='GroupA',
                    is_enabled=False,
                ),
            ],
        )
        self.assertTrue(pref.is_enabled)
        self.db.commit.assert_awaited_once()
        self.assertEqual(mock_refresh_site_cache.await_count, 2)
        mock_refresh_site_cache.assert_any_await('Site1', self.db, mock_redis)
        mock_refresh_site_cache.assert_any_await('Site2', self.db, mock_redis)

    @patch(
        'examples.local_notification_server.site_preference_service.'
        'list_sites',
    )
    @patch(
        'examples.local_notification_server.site_preference_service.'
        'list_effective_sites_for_user',
        new_callable=AsyncMock,
    )
    async def test_update_site_notification_preferences_forbidden_scope(
        self,
        _mock_list_effective_sites_for_user: AsyncMock,
        mock_list_sites: MagicMock,
    ) -> None:
        """Reject subscription updates for sites outside the user's scope."""
        site = MagicMock(id=1, name='Site1', groups=[], group=None)
        self.user.id = 5
        mock_list_sites.return_value = [site]

        with self.assertRaises(HTTPException) as ctx:
            await update_site_notification_preferences(
                SiteNotificationPreferenceUpdateRequest(
                    preferences=[
                        SiteNotificationPreferenceIn(
                            site_id=99,
                            is_enabled=True,
                        ),
                    ],
                ),
                self.db,
                self.user,
                AsyncMock(),
            )

        self.assertEqual(ctx.exception.status_code, 403)

    @patch(
        'examples.local_notification_server.site_preference_service.'
        'list_sites',
    )
    @patch(
        'examples.local_notification_server.site_preference_service.'
        'list_effective_sites_for_user',
        new_callable=AsyncMock,
    )
    async def test_list_site_preferences_group_mismatch_defaults_off(
        self,
        mock_list_effective_sites_for_user: AsyncMock,
        mock_list_sites: MagicMock,
    ) -> None:
        """Do not enable access for a mismatched direct site row."""
        group = MagicMock(name='GroupA')
        group.name = 'GroupA'
        site = MagicMock()
        site.id = 1
        site.name = 'Site1'
        site.groups = [group]
        self.user.id = 9
        mock_list_sites.return_value = [site]
        mock_list_effective_sites_for_user.return_value = []

        pref_result = MagicMock()
        pref_result.all.return_value = []
        self.db.execute.side_effect = [pref_result]

        result = await list_site_notification_preferences(
            self.db,
            self.user,
        )

        self.assertEqual(
            result,
            [
                SiteNotificationPreferenceOut(
                    site_id=1,
                    site_name='Site1',
                    group_name='GroupA',
                    is_enabled=False,
                ),
            ],
        )


if __name__ == '__main__':
    unittest.main()
