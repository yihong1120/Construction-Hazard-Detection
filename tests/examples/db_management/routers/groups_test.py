from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.models import Group
from examples.db_management.routers import groups
from examples.db_management.schemas.group import GroupCreate
from examples.db_management.schemas.group import GroupDelete
from examples.db_management.schemas.group import GroupUpdate


class TestGroupRouter(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for group management router endpoints.
    """

    async def asyncSetUp(self) -> None:
        """Set up common test variables.

        This method initialises a mock database session and an example
        group object for use in each test case.
        """
        self.db_session: AsyncMock = AsyncMock(spec=AsyncSession)
        self.example_group: Group = Group(
            id=1, name='Test Group', uniform_number='12345678',
        )
        self.super_admin: MagicMock = MagicMock(
            role='admin',
            username='ChangDar',
            group_id=None,
        )
        self.admin: MagicMock = MagicMock(
            role='admin',
            username='site-admin',
            group_id=1,
        )

    @patch('examples.db_management.routers.groups.list_groups')
    @patch(
        'examples.db_management.routers.groups.is_super_admin',
        return_value=True,
    )
    async def test_endpoint_list_groups(
        self,
        mock_is_super_admin: MagicMock,
        mock_list_groups: AsyncMock,
    ) -> None:
        """Test endpoint list groups.

        Args:
            mock_is_super_admin: Value used by this callable.
            mock_list_groups: Value used by this callable.
        """
        mock_list_groups.return_value = [self.example_group]

        result: list[Group] = await groups.endpoint_list_groups(
            db=self.db_session,
            me=self.super_admin,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, self.example_group.id)
        mock_list_groups.assert_awaited_once_with(self.db_session)

    @patch(
        'examples.db_management.routers.groups.is_super_admin',
        return_value=False,
    )
    async def test_endpoint_list_groups_admin_returns_own_group(
        self,
        mock_is_super_admin: MagicMock,
    ) -> None:
        """Test regular admin receives only their own group."""
        execute_result: MagicMock = MagicMock()
        execute_result.unique.return_value.scalar_one_or_none.return_value = (
            self.example_group
        )
        self.db_session.execute.return_value = execute_result

        result: list[Group] = await groups.endpoint_list_groups(
            db=self.db_session,
            me=self.admin,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, 1)

    @patch('examples.db_management.routers.groups.create_group')
    async def test_endpoint_create_group(
        self, mock_create_group: AsyncMock,
    ) -> None:
        """Test endpoint create group.

        Args:
            mock_create_group: Value used by this callable.
        """
        payload: GroupCreate = GroupCreate(
            name='New Group', uniform_number='87654321',
        )
        mock_create_group.return_value = self.example_group

        result: Group = await groups.endpoint_create_group(
            payload, db=self.db_session,
        )

        self.assertEqual(result.id, self.example_group.id)
        self.assertEqual(result.name, self.example_group.name)
        mock_create_group.assert_awaited_once_with(
            name=payload.name,
            uniform_number=payload.uniform_number,
            db=self.db_session,
        )

    @patch('examples.db_management.routers.groups.update_group')
    async def test_endpoint_update_group_success(
        self, mock_update_group: AsyncMock,
    ) -> None:
        """Test endpoint update group success.

        Args:
            mock_update_group: Value used by this callable.
        """
        payload: GroupUpdate = GroupUpdate(
            group_id=1, new_name='Updated Group', new_uniform_number=None,
        )
        execute_result: MagicMock = MagicMock()
        unique_result: MagicMock = MagicMock()
        unique_result.scalar_one_or_none.return_value = self.example_group
        execute_result.unique.return_value = unique_result
        self.db_session.execute.return_value = execute_result

        result: dict[str, str] = await groups.endpoint_update_group(
            payload, db=self.db_session,
        )

        self.assertEqual(result, {'message': 'Group updated successfully.'})
        mock_update_group.assert_awaited_once_with(
            grp=self.example_group,
            new_name='Updated Group',
            new_uniform_number=None,
            db=self.db_session,
        )

    async def test_endpoint_update_group_nothing_to_update(self) -> None:
        """Test endpoint update group nothing to update.
        """
        payload: GroupUpdate = GroupUpdate(group_id=1)

        with self.assertRaises(HTTPException) as ctx:
            await groups.endpoint_update_group(payload, db=self.db_session)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail, 'Nothing to update.')

    async def test_endpoint_update_group_not_found(self) -> None:
        """Test endpoint update group not found.
        """
        payload: GroupUpdate = GroupUpdate(
            group_id=999, new_name='Updated Group',
        )
        execute_result: MagicMock = MagicMock()
        unique_result: MagicMock = MagicMock()
        unique_result.scalar_one_or_none.return_value = None
        execute_result.unique.return_value = unique_result
        self.db_session.execute.return_value = execute_result

        with self.assertRaises(HTTPException) as ctx:
            await groups.endpoint_update_group(payload, db=self.db_session)

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.detail, 'Group not found.')

    @patch('examples.db_management.routers.groups.delete_group')
    async def test_endpoint_delete_group_success(
        self, mock_delete_group: AsyncMock,
    ) -> None:
        """Test endpoint delete group success.

        Args:
            mock_delete_group: Value used by this callable.
        """
        payload: GroupDelete = GroupDelete(group_id=1)
        execute_result: MagicMock = MagicMock()
        unique_result: MagicMock = MagicMock()
        unique_result.scalar_one_or_none.return_value = self.example_group
        execute_result.unique.return_value = unique_result
        self.db_session.execute.return_value = execute_result

        result: dict[str, str] = await groups.endpoint_delete_group(
            payload, db=self.db_session,
        )

        self.assertEqual(result, {'message': 'Group deleted successfully.'})
        mock_delete_group.assert_awaited_once_with(
            grp=self.example_group, db=self.db_session,
        )

    async def test_endpoint_delete_group_not_found(self) -> None:
        """Test endpoint delete group not found.
        """
        payload: GroupDelete = GroupDelete(group_id=999)
        execute_result: MagicMock = MagicMock()
        unique_result: MagicMock = MagicMock()
        unique_result.scalar_one_or_none.return_value = None
        execute_result.unique.return_value = unique_result
        self.db_session.execute.return_value = execute_result

        with self.assertRaises(HTTPException) as ctx:
            await groups.endpoint_delete_group(payload, db=self.db_session)

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.detail, 'Group not found.')


if __name__ == '__main__':
    unittest.main()

'''
pytest --cov=examples.db_management.routers.groups\
    --cov-report=term-missing\
        tests/examples/db_management/routers/groups_test.py
'''
