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

    @patch('examples.db_management.routers.groups.list_groups')
    async def test_endpoint_list_groups(
        self, mock_list_groups: AsyncMock,
    ) -> None:
        """Test listing all groups.

        Ensures that the endpoint returns a list of groups as expected.
        """
        mock_list_groups.return_value = [self.example_group]

        result: list[Group] = await groups.endpoint_list_groups(
            db=self.db_session,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, self.example_group.id)
        mock_list_groups.assert_awaited_once_with(self.db_session)

    @patch('examples.db_management.routers.groups.create_group')
    async def test_endpoint_create_group(
        self, mock_create_group: AsyncMock,
    ) -> None:
        """Test creating a new group.

        Verifies that the endpoint creates a group and returns the
        correct object.
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
        """Test successfully updating an existing group.

        Ensures that the endpoint updates the group and returns a
        success message.
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
        """Test updating a group with no fields provided raises error.

        Ensures that the endpoint raises an HTTPException if no update
        fields are provided.
        """
        payload: GroupUpdate = GroupUpdate(group_id=1)

        with self.assertRaises(HTTPException) as ctx:
            await groups.endpoint_update_group(payload, db=self.db_session)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail, 'Nothing to update.')

    async def test_endpoint_update_group_not_found(self) -> None:
        """Test updating a non-existent group raises 404 error.

        Ensures that the endpoint raises an HTTPException if the group
        does not exist in the database.
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
        """Test successfully deleting a group.

        Ensures that the endpoint deletes the group and returns a
        success message.
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
        """Test deleting a non-existent group raises 404 error.

        Ensures that the endpoint raises an HTTPException if the group
        does not exist in the database.
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
