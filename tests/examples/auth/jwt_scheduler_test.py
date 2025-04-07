from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import FastAPI

from examples.auth.jwt_scheduler import start_jwt_scheduler


class TestJwtScheduler(unittest.TestCase):
    """
    Test suite for verifying the behaviour of `start_jwt_scheduler`.
    """

    @patch('examples.auth.jwt_scheduler.update_secret_key')
    @patch('examples.auth.jwt_scheduler.BackgroundScheduler')
    def test_start_jwt_scheduler(
        self,
        mock_scheduler_cls: MagicMock,
        mock_update_secret_key: MagicMock,
    ) -> None:
        """
        Ensure `start_jwt_scheduler` sets up the scheduler to call
        `update_secret_key(app)` every 30 days and starts it.

        Args:
            mock_scheduler_cls (MagicMock):
                Mocked class for the scheduler.
            mock_update_secret_key (MagicMock):
                Mocked function for updating the secret key.
        """
        # Create a mock scheduler instance
        mock_scheduler_instance: MagicMock = MagicMock()
        mock_scheduler_cls.return_value = mock_scheduler_instance

        # Prepare a FastAPI app to pass into the function
        app: FastAPI = FastAPI()

        # Call the function under test
        returned_scheduler = start_jwt_scheduler(app)

        # The returned object should match our mock scheduler instance
        self.assertEqual(
            returned_scheduler,
            mock_scheduler_instance,
            'Returned scheduler should match the mock scheduler instance.',
        )

        # Ensure a job was added with the correct trigger and interval
        mock_scheduler_instance.add_job.assert_called_once()
        add_job_call = mock_scheduler_instance.add_job.call_args
        job_kwargs = add_job_call.kwargs

        # 'add_job' should specify an interval trigger with days=30
        self.assertEqual(
            job_kwargs['trigger'],
            'interval',
            "The job's trigger should be set to 'interval'.",
        )
        self.assertEqual(
            job_kwargs['days'],
            30,
            'The interval should be set to 30 days.',
        )

        # The function to run is a lambda: update_secret_key(app)
        job_func = job_kwargs['func']
        self.assertTrue(
            callable(job_func),
            'Scheduled job function must be callable.',
        )

        # Invoke the job_func manually to
        # simulate the scheduler running the job
        job_func()
        mock_update_secret_key.assert_called_once_with(app)

        # Finally, ensure that `scheduler.start()` was called
        mock_scheduler_instance.start.assert_called_once()


if __name__ == '__main__':
    unittest.main()

'''
pytest \
    --cov=examples.auth.jwt_scheduler \
    --cov-report=term-missing tests/examples/auth/jwt_scheduler_test.py
'''
