from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from examples.YOLO_server_api.backend.auth import create_token_logic
from examples.YOLO_server_api.backend.auth import UserLogin
from examples.YOLO_server_api.backend.models import User


class TestCreateTokenLogic(unittest.IsolatedAsyncioTestCase):
    '''
    Class to test the create_token_logic function.
    '''

    async def asyncSetUp(self):
        """
        Set up mocks before each test.
        """
        # Build Mock DB and Redis
        self.mock_db = AsyncMock(spec=AsyncSession)
        self.mock_redis = AsyncMock(spec=Redis)
        # Build mock jwt_access
        self.mock_jwt_access = MagicMock()

        # Build user_login object
        self.username = 'testuser'
        self.password = 'testpassword'
        self.user_login = UserLogin(
            username=self.username, password=self.password,
        )

        # Build mock user object with the same username and password
        self.mock_user = MagicMock(spec=User)
        self.mock_user.id = 123
        self.mock_user.username = self.username
        self.mock_user.role = 'user'
        self.mock_user.is_active = True
        # Set the check_password method to return True
        self.mock_user.check_password = AsyncMock(return_value=True)

    @patch(
        'examples.YOLO_server_api.backend.auth.get_user_data',
        new_callable=AsyncMock,
    )
    @patch(
        'examples.YOLO_server_api.backend.auth.set_user_data',
        new_callable=AsyncMock,
    )
    async def test_user_not_in_cache(
        self,
        mock_set_user_data: MagicMock,
        mock_get_user_data: MagicMock,
    ):
        """
        Test token creation when the user is not in the cache.

        Args:
            mock_set_user_data (MagicMock): Mock for set_user_data function.
            mock_get_user_data (MagicMock): Mock for get_user_data function.

        """
        mock_get_user_data.return_value = None

        # If the user is not in the cache, the database should be queried
        mock_result = MagicMock()
        mock_result.scalar.return_value = self.mock_user
        self.mock_db.execute.return_value = mock_result

        # Mock the JWT access token creation
        self.mock_jwt_access.create_access_token.return_value = 'fake_token'

        # Call the create_token_logic function
        resp = await create_token_logic(
            user=self.user_login,
            db=self.mock_db,
            redis_pool=self.mock_redis,
            jwt_access=self.mock_jwt_access,
        )

        self.assertIn('access_token', resp)
        self.assertEqual(resp['access_token'], 'fake_token')
        self.assertEqual(resp['role'], 'user')
        self.assertEqual(resp['username'], 'testuser')

        # Verify that the user data was set in the cache
        mock_set_user_data.assert_awaited_once()

    @patch(
        'examples.YOLO_server_api.backend.auth.get_user_data',
        new_callable=AsyncMock,
    )
    @patch(
        'examples.YOLO_server_api.backend.auth.set_user_data',
        new_callable=AsyncMock,
    )
    async def test_user_in_cache(
        self,
        mock_set_user_data: MagicMock,
        mock_get_user_data: MagicMock,
    ):
        """
        Test token creation when the user is in the cache.

        Args:
            mock_set_user_data (MagicMock): Mock for set_user_data function.
            mock_get_user_data (MagicMock): Mock for get_user_data function.
        """
        # If the user is in the cache, the database should not be queried
        mock_get_user_data.return_value = {
            'db_user': {
                'id': 123,
                'username': self.username,
                'role': 'user',
                'is_active': True,
            },
            'jti_list': [],
        }

        # Call the create_token_logic function
        mock_result = MagicMock()
        mock_result.scalar.return_value = self.mock_user
        self.mock_db.execute.return_value = mock_result
        self.mock_jwt_access.create_access_token.return_value = 'fake_token'

        # Call the create_token_logic function
        resp = await create_token_logic(
            user=self.user_login,
            db=self.mock_db,
            redis_pool=self.mock_redis,
            jwt_access=self.mock_jwt_access,
        )
        self.assertEqual(resp['access_token'], 'fake_token')
        self.assertEqual(resp['username'], 'testuser')
        self.assertEqual(resp['role'], 'user')

        # Verify that the user data was not set in the cache
        mock_set_user_data.assert_awaited_once()

    @patch(
        'examples.YOLO_server_api.backend.auth.get_user_data',
        new_callable=AsyncMock,
    )
    async def test_user_not_found_in_db(
        self,
        mock_get_user_data: MagicMock,
    ):
        """
        Test token creation failure when the user is not found in the database

        Args:
            mock_get_user_data (MagicMock): Mock for get_user_data function.
        """
        # Mock the get_user_data function to return None
        mock_get_user_data.return_value = None
        mock_result = MagicMock()
        mock_result.scalar.return_value = None
        self.mock_db.execute.return_value = mock_result

        # Verify that an exception is raised
        # when the user is not found in the database
        with self.assertRaisesRegex(Exception, '401'):
            await create_token_logic(
                user=self.user_login,
                db=self.mock_db,
                redis_pool=self.mock_redis,
                jwt_access=self.mock_jwt_access,
            )

    @patch(
        'examples.YOLO_server_api.backend.auth.get_user_data',
        new_callable=AsyncMock,
    )
    async def test_wrong_password(
        self,
        mock_get_user_data: MagicMock,
    ):
        """
        Test token creation failure when the user password is incorrect.

        Args:
            mock_get_user_data (MagicMock): Mock for get_user_data function.
        """
        # Set the mock user data to None
        mock_get_user_data.return_value = None
        # Set the check_password method to return False
        self.mock_user.check_password.return_value = False

        # Mock the database execute result to return the user
        mock_result = MagicMock()
        mock_result.scalar.return_value = self.mock_user
        self.mock_db.execute.return_value = mock_result

        # Verify that an exception is raised
        # when the user password is incorrect
        with self.assertRaisesRegex(Exception, '401'):
            await create_token_logic(
                user=self.user_login,
                db=self.mock_db,
                redis_pool=self.mock_redis,
                jwt_access=self.mock_jwt_access,
            )

    @patch(
        'examples.YOLO_server_api.backend.auth.get_user_data',
        new_callable=AsyncMock,
    )
    async def test_user_inactive(
        self,
        mock_get_user_data: MagicMock,
    ):
        """
        Test token creation failure when the user account is inactive.

        Args:
            mock_get_user_data (MagicMock): Mock for get_user_data function.
        """
        # Set the mock user to inactive
        mock_get_user_data.return_value = None
        self.mock_user.is_active = False

        # Mock the database execute result to return the user
        mock_result = MagicMock()
        mock_result.scalar.return_value = self.mock_user
        self.mock_db.execute.return_value = mock_result

        # Verify that an exception is raised when the user account is inactive
        with self.assertRaisesRegex(Exception, '403'):
            await create_token_logic(
                user=self.user_login,
                db=self.mock_db,
                redis_pool=self.mock_redis,
                jwt_access=self.mock_jwt_access,
            )

    @patch(
        'examples.YOLO_server_api.backend.auth.get_user_data',
        new_callable=AsyncMock,
    )
    async def test_invalid_role(
        self,
        mock_get_user_data: MagicMock,
    ):
        """
        Test token creation failure when the user role is invalid.

        Args:
            mock_get_user_data (MagicMock): Mock for get_user_data function
        """
        # Set the mock user to have an invalid role
        mock_get_user_data.return_value = None
        self.mock_user.role = 'invalid_role'

        # Mock the database execute result to return the user
        mock_result = MagicMock()
        mock_result.scalar.return_value = self.mock_user
        self.mock_db.execute.return_value = mock_result

        # Verify that an exception is raised when the user role is invalid
        with self.assertRaisesRegex(Exception, '403'):
            await create_token_logic(
                user=self.user_login,
                db=self.mock_db,
                redis_pool=self.mock_redis,
                jwt_access=self.mock_jwt_access,
            )

    @patch(
        'examples.YOLO_server_api.backend.auth.get_user_data',
        new_callable=AsyncMock,
    )
    @patch(
        'examples.YOLO_server_api.backend.auth.set_user_data',
        new_callable=AsyncMock,
    )
    async def test_jti_list_full(
        self,
        mock_set_user_data: MagicMock,
        mock_get_user_data: MagicMock,
    ):
        """
        Test token creation when the JTI list is full.

        Args:
            mock_set_user_data (MagicMock): Mock for set_user_data function.
            mock_get_user_data (MagicMock): Mock for get_user_data function.
        """
        # Mock the user data with a full JTI list
        mock_get_user_data.return_value = {
            'db_user': {
                'id': 123,
                'username': self.username,
                'role': 'user',
                'is_active': True,
            },
            'jti_list': ['jti1', 'jti2'],  # Assume max_jti=2
        }

        # Mock the database execute result to return the user
        mock_result = MagicMock()
        mock_result.scalar.return_value = self.mock_user
        self.mock_db.execute.return_value = mock_result

        # Mock the JWT access token creation
        self.mock_jwt_access.create_access_token.return_value = 'fake_token'

        # Call the create_token_logic function with max_jti=2
        resp = await create_token_logic(
            user=self.user_login,
            db=self.mock_db,
            redis_pool=self.mock_redis,
            jwt_access=self.mock_jwt_access,
            max_jti=2,
        )

        # Validate the response
        self.assertEqual(resp['access_token'], 'fake_token')
        self.assertEqual(resp['role'], 'user')
        self.assertEqual(resp['username'], self.username)

        # Verify that the JTI list was updated correctly
        # Extract user data
        updated_user_data = mock_set_user_data.call_args[0][2]
        self.assertEqual(len(updated_user_data['jti_list']), 2)  # Max length
        # Oldest JTI removed
        self.assertNotIn('jti1', updated_user_data['jti_list'])
        # Second JTI remains
        self.assertIn('jti2', updated_user_data['jti_list'])
        self.assertIn(
            updated_user_data['jti_list'][-1],  # New JTI added
            updated_user_data['jti_list'],
        )


if __name__ == '__main__':
    unittest.main()
