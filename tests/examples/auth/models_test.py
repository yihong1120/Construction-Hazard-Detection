from __future__ import annotations

import unittest
from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

from examples.auth.database import Base
from examples.auth.models import Feature
from examples.auth.models import Group
from examples.auth.models import Tenant
from examples.auth.models import User

# Define the in-memory database URI for testing
DATABASE_URL = 'sqlite:///:memory:'

# Configure the testing database and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


class TestUserModel(unittest.TestCase):
    """Test cases for the User model."""

    def setUp(self) -> None:
        """Set up a test database session."""
        self.session: Session = SessionLocal()
        self.tenant = Tenant(name=f"test-tenant-{uuid4()}")
        self.session.add(self.tenant)
        self.session.commit()

    def tearDown(self) -> None:
        """Clean up the database after each test."""
        self.session.close()

    def test_to_dict(self) -> None:
        """Test the to_dict method of the User model."""
        user = User(
            username='testuser',
            role='admin',
            status='active',
            tenant_id=self.tenant.id,
        )
        self.session.add(user)
        self.session.commit()

        # Convert user instance to dictionary
        user_dict = user.to_dict()

        # Validate dictionary contents
        self.assertEqual(user_dict['username'], 'testuser')
        self.assertEqual(user_dict['role'], 'admin')
        self.assertEqual(user_dict['status'], 'active')
        self.assertIn('created_at', user_dict)
        self.assertIn('updated_at', user_dict)

    def test_feature_repr(self) -> None:
        """Exercise this test."""
        feature = Feature(id=1, feature_name='test_feature')
        self.assertIn('Feature', repr(feature))
        self.assertIn('test_feature', repr(feature))

    def test_group_repr(self) -> None:
        """Exercise this test."""
        group = Group(
            id=1,
            name='test_group',
            uniform_number='12345678',
            max_allowed_streams=1,
        )
        self.assertIn('Group', repr(group))
        self.assertIn('test_group', repr(group))


if __name__ == '__main__':
    unittest.main()
