from __future__ import annotations

import unittest

from pydantic import ValidationError

from examples.db_management.schemas import site


class TestSiteCreate(unittest.TestCase):
    """Unit tests for SiteCreate schema."""

    def test_valid(self) -> None:
        """Test creating a valid SiteCreate instance."""
        data: site.SiteCreate = site.SiteCreate(name='s', group_id=1)
        self.assertEqual(data.name, 's')
        self.assertEqual(data.group_id, 1)

    def test_optional(self) -> None:
        """Test creating a SiteCreate instance without optional group_id."""
        data: site.SiteCreate = site.SiteCreate(name='s')
        self.assertIsNone(data.group_id)

    def test_missing(self) -> None:
        """Test validation error if required fields are missing."""
        with self.assertRaises(ValidationError):
            site.SiteCreate()


class TestSiteUpdate(unittest.TestCase):
    """Unit tests for SiteUpdate schema."""

    def test_valid(self) -> None:
        """Test creating a valid SiteUpdate instance."""
        data: site.SiteUpdate = site.SiteUpdate(site_id=1, new_name='n')
        self.assertEqual(data.site_id, 1)
        self.assertEqual(data.new_name, 'n')

    def test_missing(self) -> None:
        """Test validation error if required fields are missing."""
        with self.assertRaises(ValidationError):
            site.SiteUpdate()


class TestSiteDelete(unittest.TestCase):
    """Unit tests for SiteDelete schema."""

    def test_valid(self) -> None:
        """Test creating a valid SiteDelete instance."""
        data: site.SiteDelete = site.SiteDelete(site_id=1)
        self.assertEqual(data.site_id, 1)

    def test_missing(self) -> None:
        """Test validation error if required field site_id is missing."""
        with self.assertRaises(ValidationError):
            site.SiteDelete()


class TestSiteUserOp(unittest.TestCase):
    """Unit tests for SiteUserOp schema."""

    def test_valid(self) -> None:
        """Test creating a valid SiteUserOp instance."""
        data: site.SiteUserOp = site.SiteUserOp(site_id=1, user_id=2)
        self.assertEqual(data.site_id, 1)
        self.assertEqual(data.user_id, 2)

    def test_missing(self) -> None:
        """Test validation error if required fields are missing."""
        with self.assertRaises(ValidationError):
            site.SiteUserOp()


class TestSiteRead(unittest.TestCase):
    """Unit tests for SiteRead schema."""

    def test_valid(self) -> None:
        """Test creating a valid SiteRead instance with all fields."""
        data: site.SiteRead = site.SiteRead(
            id=1, name='s', group_id=2,
            group_name='g', user_ids=[1, 2],
        )
        self.assertEqual(data.id, 1)
        self.assertEqual(data.name, 's')
        self.assertEqual(data.group_id, 2)
        self.assertEqual(data.group_name, 'g')
        self.assertEqual(data.user_ids, [1, 2])

    def test_optional(self) -> None:
        """Test creating a SiteRead instance with optional fields omitted."""
        data: site.SiteRead = site.SiteRead(id=2, name='s', user_ids=[])
        self.assertIsNone(data.group_id)
        self.assertIsNone(data.group_name)
        self.assertEqual(data.user_ids, [])

    def test_missing(self) -> None:
        """Test validation error if required fields are missing."""
        with self.assertRaises(ValidationError):
            site.SiteRead()


if __name__ == '__main__':
    unittest.main()

'''
pytest --cov=examples.db_management.schemas.site\
    --cov-report=term-missing\
        tests/examples/db_management/schemas/site_test.py
'''
