from __future__ import annotations

import unittest

from pydantic import ValidationError

from examples.db_management.schemas import feature


class TestFeatureCreate(unittest.TestCase):
    """Unit tests for FeatureCreate schema."""

    def test_valid(self) -> None:
        """Test creating a valid FeatureCreate instance."""
        data = feature.FeatureCreate(feature_name='f', description='desc')
        self.assertEqual(data.feature_name, 'f')
        self.assertEqual(data.description, 'desc')

    def test_optional(self) -> None:
        """
        Test creating a FeatureCreate instance without optional description.
        """
        data = feature.FeatureCreate(feature_name='f')
        self.assertIsNone(data.description)

    def test_missing(self) -> None:
        """
        Test creating a FeatureCreate instance with missing required fields.
        """
        with self.assertRaises(ValidationError):
            feature.FeatureCreate()


class TestFeatureUpdate(unittest.TestCase):
    """Unit tests for FeatureUpdate schema."""

    def test_valid(self) -> None:
        """Test creating a valid FeatureUpdate instance."""
        data = feature.FeatureUpdate(
            feature_id=1, new_name='n', new_description='d',
        )
        self.assertEqual(data.feature_id, 1)
        self.assertEqual(data.new_name, 'n')
        self.assertEqual(data.new_description, 'd')

    def test_optional(self) -> None:
        """Test creating a FeatureUpdate instance with only required fields."""
        data = feature.FeatureUpdate(feature_id=2)
        self.assertIsNone(data.new_name)
        self.assertIsNone(data.new_description)

    def test_missing(self) -> None:
        """Test FeatureUpdate instance creation without required feature_id."""
        with self.assertRaises(ValidationError):
            feature.FeatureUpdate()


class TestFeatureDelete(unittest.TestCase):
    """Unit tests for FeatureDelete schema."""

    def test_valid(self) -> None:
        """Test creating a valid FeatureDelete instance."""
        data = feature.FeatureDelete(feature_id=1)
        self.assertEqual(data.feature_id, 1)

    def test_missing(self) -> None:
        """Test FeatureDelete instance creation without required feature_id."""
        with self.assertRaises(ValidationError):
            feature.FeatureDelete()


class TestFeatureRead(unittest.TestCase):
    """Unit tests for FeatureRead schema."""

    def test_valid(self) -> None:
        """Test creating a valid FeatureRead instance."""
        data = feature.FeatureRead(id=1, feature_name='f', description='d')
        self.assertEqual(data.id, 1)
        self.assertEqual(data.feature_name, 'f')
        self.assertEqual(data.description, 'd')

    def test_optional(self) -> None:
        """
        Test creating a FeatureRead instance without optional description.
        """
        data = feature.FeatureRead(id=2, feature_name='f')
        self.assertIsNone(data.description)

    def test_missing(self) -> None:
        """Test FeatureRead instance creation without required fields."""
        with self.assertRaises(ValidationError):
            feature.FeatureRead()


class TestGroupFeatureUpdate(unittest.TestCase):
    """Unit tests for GroupFeatureUpdate schema."""

    def test_valid(self) -> None:
        """Test creating a valid GroupFeatureUpdate instance."""
        data = feature.GroupFeatureUpdate(group_id=1, feature_ids=[1, 2, 3])
        self.assertEqual(data.group_id, 1)
        self.assertEqual(data.feature_ids, [1, 2, 3])

    def test_empty_list(self) -> None:
        """
        Test creating GroupFeatureUpdate instance with empty feature_ids list.
        """
        data = feature.GroupFeatureUpdate(group_id=2, feature_ids=[])
        self.assertEqual(data.feature_ids, [])

    def test_missing(self) -> None:
        """
        Test GroupFeatureUpdate instance creation without required group_id.
        """
        with self.assertRaises(ValidationError):
            feature.GroupFeatureUpdate()


if __name__ == '__main__':
    unittest.main()

'''
pytest --cov=examples.db_management.schemas.feature\
    --cov-report=term-missing\
        tests/examples/db_management/schemas/feature_test.py
'''
