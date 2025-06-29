from __future__ import annotations

import unittest

from pydantic import ValidationError

from examples.db_management.schemas import group


class TestGroupCreate(unittest.TestCase):
    """Unit tests for the GroupCreate schema."""

    def test_valid(self) -> None:
        """Test creating a valid GroupCreate object."""
        data: group.GroupCreate = group.GroupCreate(
            name='g', uniform_number='12345678',
        )
        self.assertEqual(data.name, 'g')
        self.assertEqual(data.uniform_number, '12345678')

    def test_missing(self) -> None:
        """Test validation error when required fields are missing."""
        with self.assertRaises(ValidationError):
            group.GroupCreate(name='g')
        with self.assertRaises(ValidationError):
            group.GroupCreate(uniform_number='12345678')


class TestGroupUpdate(unittest.TestCase):
    """Unit tests for the GroupUpdate schema."""

    def test_valid(self) -> None:
        """Test creating a valid GroupUpdate object with all fields."""
        data: group.GroupUpdate = group.GroupUpdate(
            group_id=1, new_name='n', new_uniform_number='u',
        )
        self.assertEqual(data.group_id, 1)
        self.assertEqual(data.new_name, 'n')
        self.assertEqual(data.new_uniform_number, 'u')

    def test_optional(self) -> None:
        """
        Test creating a valid GroupUpdate object with optional fields omitted.
        """
        data: group.GroupUpdate = group.GroupUpdate(group_id=2)
        self.assertIsNone(data.new_name)
        self.assertIsNone(data.new_uniform_number)

    def test_missing(self) -> None:
        """Test validation error when required group_id field is missing."""
        with self.assertRaises(ValidationError):
            group.GroupUpdate()


class TestGroupDelete(unittest.TestCase):
    """Unit tests for the GroupDelete schema."""

    def test_valid(self) -> None:
        """Test creating a valid GroupDelete object."""
        data: group.GroupDelete = group.GroupDelete(group_id=1)
        self.assertEqual(data.group_id, 1)

    def test_missing(self) -> None:
        """Test validation error when required group_id field is missing."""
        with self.assertRaises(ValidationError):
            group.GroupDelete()


class TestGroupRead(unittest.TestCase):
    """Unit tests for the GroupRead schema."""

    def test_valid(self) -> None:
        """Test creating a valid GroupRead object."""
        data: group.GroupRead = group.GroupRead(
            id=1, name='g', uniform_number='12345678',
        )
        self.assertEqual(data.id, 1)
        self.assertEqual(data.name, 'g')
        self.assertEqual(data.uniform_number, '12345678')

    def test_missing(self) -> None:
        """Test validation error when required fields are missing."""
        with self.assertRaises(ValidationError):
            group.GroupRead()


class TestGroupFeatureRead(unittest.TestCase):
    """Unit tests for the GroupFeatureRead schema."""

    def test_valid(self) -> None:
        """Test creating a valid GroupFeatureRead object with feature IDs."""
        data: group.GroupFeatureRead = group.GroupFeatureRead(
            group_id=1, group_name='g', feature_ids=[1, 2],
        )
        self.assertEqual(data.group_id, 1)
        self.assertEqual(data.group_name, 'g')
        self.assertEqual(data.feature_ids, [1, 2])

    def test_empty_list(self) -> None:
        """
        Test creating a GroupFeatureRead object with an empty feature_ids list.
        """
        data: group.GroupFeatureRead = group.GroupFeatureRead(
            group_id=2, group_name='g2', feature_ids=[],
        )
        self.assertEqual(data.feature_ids, [])

    def test_missing(self) -> None:
        """Test validation error when required fields are missing."""
        with self.assertRaises(ValidationError):
            group.GroupFeatureRead()


if __name__ == '__main__':
    unittest.main()


'''
pytest --cov=examples.db_management.schemas.group\
    --cov-report=term-missing\
        tests/examples/db_management/schemas/group_test.py
'''
