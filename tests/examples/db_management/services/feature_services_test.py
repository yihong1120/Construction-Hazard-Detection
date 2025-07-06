from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException

from examples.auth.models import Feature
from examples.auth.models import Group
from examples.db_management.services import feature_services


class TestFeatureServices(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for feature_services module using asynchronous mocks.
    """

    def setUp(self) -> None:
        """Set up common mock objects for each test.

        This method initialises mock database, feature, and group
        objects for use in each test case.
        """
        self.db: AsyncMock = AsyncMock()
        self.feat: MagicMock = MagicMock(spec=Feature)
        self.feat.id = 1
        self.feat.feature_name = 'Test Feature'
        self.feat.description = 'A test feature'

        self.group: MagicMock = MagicMock(spec=Group)
        self.group.id = 100

    async def test_list_features(self) -> None:
        """Test retrieving a list of all features.

        Ensures that all features are returned as expected from the
        database query.
        """
        mock_result: MagicMock = MagicMock()
        scalars_mock: MagicMock = (
            mock_result.unique.return_value.scalars.return_value
        )
        scalars_mock.all.return_value = ['feature1', 'feature2']

        self.db.execute = AsyncMock(return_value=mock_result)

        features: list = await feature_services.list_features(db=self.db)

        self.assertEqual(features, ['feature1', 'feature2'])

    async def test_create_feature_success(self) -> None:
        """Test successful creation of a feature.

        Verifies that a new feature is created and committed to the
        database without error.
        """
        self.db.commit = AsyncMock()
        self.db.refresh = AsyncMock()
        self.db.add = MagicMock()

        with patch(
            'examples.db_management.services.feature_services.Feature',
        ) as MockFeature:
            mock_feature: MagicMock = MagicMock()
            MockFeature.return_value = mock_feature

            result: MagicMock = await feature_services.create_feature(
                name='New Feature',
                description='Description here',
                db=self.db,
            )

            self.assertEqual(result, mock_feature)
            self.db.add.assert_called_with(mock_feature)
            self.db.commit.assert_awaited()
            self.db.refresh.assert_awaited_with(mock_feature)

    async def test_create_feature_exception(self) -> None:
        """Test feature creation raises HTTPException on database error.

        Ensures that an HTTPException is raised and rollback is called
        if the database commit fails during feature creation.
        """
        self.db.commit = AsyncMock(side_effect=Exception('DB error'))
        self.db.rollback = AsyncMock()
        self.db.add = MagicMock()

        with self.assertRaises(HTTPException) as context:
            await feature_services.create_feature(
                name='Fail Feature',
                description='Fail case',
                db=self.db,
            )

        self.assertEqual(context.exception.status_code, 500)
        self.db.rollback.assert_awaited()

    async def test_update_feature_success(self) -> None:
        """Test successful update of feature details.

        Verifies that the feature details are updated and committed to
        the database.
        """
        self.db.commit = AsyncMock()

        await feature_services.update_feature(
            feat=self.feat,
            new_name='Updated Name',
            new_description='Updated description',
            db=self.db,
        )

        self.db.commit.assert_awaited()
        self.assertEqual(self.feat.feature_name, 'Updated Name')
        self.assertEqual(self.feat.description, 'Updated description')

    async def test_update_feature_no_fields(self) -> None:
        """Test updating feature raises HTTPException if no fields provided.

        Ensures that an HTTPException with status 400 is raised if no
        update fields are provided.
        """
        with self.assertRaises(HTTPException) as context:
            await feature_services.update_feature(
                feat=self.feat,
                new_name=None,
                new_description=None,
                db=self.db,
            )

        self.assertEqual(context.exception.status_code, 400)

    async def test_update_feature_exception(self) -> None:
        """Test feature update raises HTTPException on database error.

        Ensures that an HTTPException is raised and rollback is called
        if the database commit fails during feature update.
        """
        self.db.commit = AsyncMock(side_effect=Exception('DB error'))
        self.db.rollback = AsyncMock()

        with self.assertRaises(HTTPException) as context:
            await feature_services.update_feature(
                feat=self.feat,
                new_name='Error Name',
                new_description=None,
                db=self.db,
            )

        self.assertEqual(context.exception.status_code, 500)
        self.db.rollback.assert_awaited()

    async def test_delete_feature_success(self) -> None:
        """Test successful deletion of a feature.

        Verifies that the feature is deleted and the transaction is
        committed.
        """
        self.db.delete = AsyncMock()
        self.db.commit = AsyncMock()

        await feature_services.delete_feature(feat=self.feat, db=self.db)

        self.db.delete.assert_awaited_with(self.feat)
        self.db.commit.assert_awaited()

    async def test_delete_feature_exception(self) -> None:
        """Test feature deletion raises HTTPException on database error.

        Ensures that an HTTPException with status 500 is raised if the
        database commit fails during feature deletion.
        """
        self.db.delete = AsyncMock()
        self.db.commit = AsyncMock(side_effect=Exception('DB error'))
        self.db.rollback = AsyncMock()

        with self.assertRaises(HTTPException) as context:
            await feature_services.delete_feature(feat=self.feat, db=self.db)

        self.assertEqual(context.exception.status_code, 500)
        self.db.rollback.assert_awaited()

    async def test_update_group_features(self) -> None:
        """Test updating features associated with a group.

        Verifies that the features associated with a group are updated
        and the transactions are committed.
        """
        self.db.execute = AsyncMock()
        self.db.commit = AsyncMock()

        feature_ids: list[int] = [1, 2, 3]

        await feature_services.update_group_features(
            group=self.group,
            feature_ids=feature_ids,
            db=self.db,
        )

        self.assertEqual(self.db.execute.call_count, 2)
        self.assertEqual(self.db.commit.call_count, 2)

    async def test_list_group_features(self) -> None:
        """Test listing groups with their associated feature IDs.

        Ensures that the correct group-feature associations are returned
        from the database query.
        """
        mock_group: MagicMock = MagicMock()
        mock_group.features = [MagicMock(id=1), MagicMock(id=2)]

        mock_result: MagicMock = MagicMock()
        mock_scalars: MagicMock = (
            mock_result.unique.return_value.scalars.return_value
        )
        mock_scalars.all.return_value = [mock_group]

        self.db.execute = AsyncMock(return_value=mock_result)

        results: list = await feature_services.list_group_features(db=self.db)

        self.assertEqual(results, [(mock_group, [1, 2])])


if __name__ == '__main__':
    unittest.main()

'''
pytest --cov=examples.db_management.services.feature_services\
    --cov-report=term-missing\
        tests/examples/db_management/services/feature_services_test.py
'''
