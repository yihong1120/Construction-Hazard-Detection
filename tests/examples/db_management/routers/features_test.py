from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.models import Feature
from examples.auth.models import Group
from examples.auth.models import User
from examples.db_management.routers import features
from examples.db_management.schemas.feature import FeatureCreate
from examples.db_management.schemas.feature import FeatureDelete
from examples.db_management.schemas.feature import FeatureUpdate
from examples.db_management.schemas.feature import GroupFeatureUpdate


class TestFeatureRouter(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for feature management router endpoints.
    """

    async def asyncSetUp(self) -> None:
        """Set up common test variables.

        This method initialises a mock database session, a super admin
        user, and example feature/group objects for use in each test.
        """
        self.db_session: AsyncMock = AsyncMock(spec=AsyncSession)
        self.super_admin_user: User = User(
            id=1, username='ChangDar', role='admin',
        )
        self.example_feature: Feature = Feature(
            id=1, feature_name='Test Feature', description='A test feature',
        )
        self.example_group: Group = Group(id=1, name='Test Group')

    @patch('examples.db_management.routers.features.list_features')
    async def test_endpoint_list_features_super_admin(
        self, mock_list_features: AsyncMock,
    ) -> None:
        """Test super admin can list all features.

        Ensures that a super admin can retrieve all features.
        """
        mock_list_features.return_value = [self.example_feature]

        result: list[Feature] = await features.endpoint_list_features(
            db=self.db_session, current_user=self.super_admin_user,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, self.example_feature.id)
        mock_list_features.assert_awaited_once_with(self.db_session)

    async def test_endpoint_list_features_non_super_admin(self) -> None:
        """Test listing features raises HTTPException for non-super admin.

        Ensures that a non-super admin cannot list all features and
        receives a 403 error.
        """
        non_admin_user: User = User(id=2, username='user', role='user')

        with self.assertRaises(HTTPException) as ctx:
            await features.endpoint_list_features(
                db=self.db_session, current_user=non_admin_user,
            )

        self.assertEqual(ctx.exception.status_code, 403)

    @patch('examples.db_management.routers.features.create_feature')
    async def test_endpoint_create_feature(
        self, mock_create_feature: AsyncMock,
    ) -> None:
        """Test creating a new feature.

        Verifies that a new feature is created and returned correctly.
        """
        payload: FeatureCreate = FeatureCreate(
            feature_name='New Feature', description='Feature desc',
        )
        mock_create_feature.return_value = self.example_feature

        result: Feature = await features.endpoint_create_feature(
            payload,
            db=self.db_session,
        )

        self.assertEqual(result.id, self.example_feature.id)
        mock_create_feature.assert_awaited_once_with(
            payload.feature_name,
            payload.description,
            self.db_session,
        )

    @patch('examples.db_management.routers.features.update_feature')
    async def test_endpoint_update_feature_success(
        self, mock_update_feature: AsyncMock,
    ) -> None:
        """Test updating an existing feature successfully.

        Ensures that an existing feature is updated and a success
        message is returned.
        """
        payload: FeatureUpdate = FeatureUpdate(
            feature_id=1, new_name='Updated Feature', new_description=None,
        )
        execute_result: MagicMock = MagicMock()
        unique_result: MagicMock = MagicMock()
        unique_result.scalar_one_or_none.return_value = self.example_feature
        execute_result.unique.return_value = unique_result
        self.db_session.execute.return_value = execute_result

        result: dict[str, str] = await features.endpoint_update_feature(
            payload,
            db=self.db_session,
        )

        self.assertEqual(result, {'message': 'Feature updated successfully.'})
        mock_update_feature.assert_awaited_once_with(
            self.example_feature,
            payload.new_name,
            payload.new_description,
            self.db_session,
        )

    async def test_endpoint_update_feature_not_found(self) -> None:
        """Test updating non-existent feature raises 404 error.

        Ensures that a 404 error is raised if the feature does not exist.
        """
        payload: FeatureUpdate = FeatureUpdate(
            feature_id=999, new_name='Updated Feature',
        )
        execute_result: MagicMock = MagicMock()
        unique_result: MagicMock = MagicMock()
        unique_result.scalar_one_or_none.return_value = None
        execute_result.unique.return_value = unique_result
        self.db_session.execute.return_value = execute_result

        with self.assertRaises(HTTPException) as ctx:
            await features.endpoint_update_feature(payload, db=self.db_session)

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.detail, 'Feature not found')

    @patch('examples.db_management.routers.features.delete_feature')
    async def test_endpoint_delete_feature_success(
        self, mock_delete_feature: AsyncMock,
    ) -> None:
        """Test deleting an existing feature successfully.

        Ensures that an existing feature is deleted and a success
        message is returned.
        """
        payload: FeatureDelete = FeatureDelete(feature_id=1)
        execute_result: MagicMock = MagicMock()
        unique_result: MagicMock = MagicMock()
        unique_result.scalar_one_or_none.return_value = self.example_feature
        execute_result.unique.return_value = unique_result
        self.db_session.execute.return_value = execute_result

        result: dict[str, str] = await features.endpoint_delete_feature(
            payload,
            db=self.db_session,
        )

        self.assertEqual(result, {'message': 'Feature deleted successfully.'})
        mock_delete_feature.assert_awaited_once_with(
            self.example_feature,
            self.db_session,
        )

    async def test_endpoint_delete_feature_not_found(self) -> None:
        """Test deleting non-existent feature raises 404 error.

        Ensures that a 404 error is raised if the feature does not exist.
        """
        payload: FeatureDelete = FeatureDelete(feature_id=999)
        execute_result: MagicMock = MagicMock()
        unique_result: MagicMock = MagicMock()
        unique_result.scalar_one_or_none.return_value = None
        execute_result.unique.return_value = unique_result
        self.db_session.execute.return_value = execute_result

        with self.assertRaises(HTTPException) as ctx:
            await features.endpoint_delete_feature(payload, db=self.db_session)

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.detail, 'Feature not found')

    @patch('examples.db_management.routers.features.update_group_features')
    async def test_endpoint_update_group_feature_success(
        self, mock_update_group_features: AsyncMock,
    ) -> None:
        """Test updating group features successfully.

        Ensures that group features are updated and a success message
        is returned.
        """
        payload: GroupFeatureUpdate = GroupFeatureUpdate(
            group_id=1, feature_ids=[1, 2],
        )
        execute_result: MagicMock = MagicMock()
        unique_result: MagicMock = MagicMock()
        unique_result.scalar_one_or_none.return_value = self.example_group
        execute_result.unique.return_value = unique_result
        self.db_session.execute.return_value = execute_result

        result: dict[str, str] = await features.endpoint_update_group_feature(
            payload,
            db=self.db_session,
        )
        self.assertEqual(
            result, {'message': 'Group features updated successfully.'},
        )
        mock_update_group_features.assert_awaited_once_with(
            self.example_group,
            [1, 2],
            self.db_session,
        )

    async def test_endpoint_update_group_feature_not_found(self) -> None:
        """
        Test updating group features for non-existent group raises 404 error.
        """
        payload: GroupFeatureUpdate = GroupFeatureUpdate(
            group_id=999, feature_ids=[1, 2],
        )
        execute_result: MagicMock = MagicMock()
        unique_result: MagicMock = MagicMock()
        unique_result.scalar_one_or_none.return_value = None
        execute_result.unique.return_value = unique_result
        self.db_session.execute.return_value = execute_result

        with self.assertRaises(HTTPException) as ctx:
            await features.endpoint_update_group_feature(
                payload,
                db=self.db_session,
            )
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.detail, 'Group not found')

    @patch('examples.db_management.routers.features.list_group_features')
    async def test_endpoint_list_group_features(
        self, mock_list_group_features: AsyncMock,
    ) -> None:
        """Test listing group features returns correct data.

        Ensures that the endpoint returns the correct group-feature
        associations.
        """
        mock_list_group_features.return_value = [
            (self.example_group, [1, 2]),
        ]
        result: list[features.GroupFeatureRead] = (
            await features.endpoint_list_group_features(
                db=self.db_session,
            )
        )
        self.assertEqual(
            result,
            [
                features.GroupFeatureRead(
                    group_id=self.example_group.id,
                    group_name=self.example_group.name,
                    feature_ids=[1, 2],
                ),
            ],
        )
        mock_list_group_features.assert_awaited_once_with(self.db_session)


if __name__ == '__main__':
    unittest.main()

'''
pytest --cov=examples.db_management.routers.features\
    --cov-report=term-missing\
        tests/examples/db_management/routers/features_test.py
'''
