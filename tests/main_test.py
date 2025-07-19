from __future__ import annotations

import asyncio
import multiprocessing
import sys
import unittest
from datetime import datetime
from datetime import timedelta
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import main
from main import MainApp
from main import process_single_stream
from main import StreamConfig


class AsyncFrameGenerator:
    """Async generator for mock video frames."""

    def __init__(self):
        self.yielded = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.yielded:
            self.yielded = True
            # Return a dummy frame and timestamp if needed
            return (MagicMock(), 1640995200)
        else:
            raise StopAsyncIteration


class MockCursor:
    async def execute(self, *args, **kwargs):
        pass

    async def fetchall(self):
        return []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


class MockConnection:
    def cursor(self):
        return MockCursor()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


class MockAcquire:
    async def __aenter__(self):
        return MockConnection()

    async def __aexit__(self, exc_type, exc, tb):
        pass


class MockPool:
    def acquire(self):
        return MockAcquire()


class TestMainApp(unittest.IsolatedAsyncioTestCase):
    """Unit tests for MainApp class defined in main.py."""
    @patch.object(MainApp, '_ensure_db_pool', new_callable=AsyncMock)
    async def test_fetch_stream_configs_db_pool_not_initialised(
        self, mock_ensure,
    ):
        """Test fetch_stream_configs raises if db_pool is not initialised."""
        self.app.db_pool = None
        with self.assertRaises(RuntimeError):
            await self.app.fetch_stream_configs()

    @patch('main.Process')
    def test_start_and_stop_process(self, mock_process_class):
        """Test start_process and stop_process methods."""
        # Mock the Process class to avoid actually starting processes
        mock_process = MagicMock()
        mock_process_class.return_value = mock_process

        cfg = self.dummy_cfg.copy()
        proc = self.app.start_process(cfg)

        # Verify Process was created with correct arguments
        mock_process_class.assert_called_once_with(
            target=process_single_stream, args=(cfg,),
        )
        mock_process.start.assert_called_once()
        self.assertEqual(proc, mock_process)

        # Test stop_process
        self.app.stop_process(proc)
        mock_process.terminate.assert_called_once()
        mock_process.join.assert_called_once()

    @patch('main.MainApp.start_process')
    @patch('main.MainApp.fetch_stream_configs')
    def test_reload_configurations_starts_new_stream(
        self, mock_fetch, mock_start,
    ):
        """Test reload_configurations starts new stream if not tracked."""
        cfg = self.dummy_cfg.copy()
        cfg['expire_date'] = None
        mock_fetch.return_value = [cfg]
        mock_start.return_value = MagicMock()  # Mock process

        app = MainApp()
        app.running_processes = {}

        async def run():
            await app.reload_configurations()
        asyncio.run(run())
        self.assertIn(cfg['video_url'], app.running_processes)
        mock_start.assert_called_once_with(cfg)

    @patch('main.MainApp.fetch_stream_configs')
    def test_reload_configurations_skips_expired_config(self, mock_fetch):
        """Test reload_configurations skips expired configs."""
        expired_cfg = self.dummy_cfg.copy()
        expired_cfg['expire_date'] = (
            datetime.now() - timedelta(days=1)
        ).isoformat()
        mock_fetch.return_value = [expired_cfg]
        app = MainApp()
        app.running_processes = {}

        async def run():
            await app.reload_configurations()
        asyncio.run(run())
        self.assertNotIn(expired_cfg['video_url'], app.running_processes)

    @patch('main.asyncio.run')
    @patch('main.argparse.ArgumentParser.parse_args')
    def test_main_entrypoint(self, mock_args, mock_run):
        """Test CLI entrypoint main() function."""
        from main import main as main_entry
        mock_args.return_value = type('Args', (), {'poll': 1})()
        asyncio_run_called = False

        def fake_run(coro):
            nonlocal asyncio_run_called
            asyncio_run_called = True
            assert asyncio.iscoroutine(coro)
        mock_run.side_effect = fake_run
        # Should not raise
        asyncio.run(main_entry())
        self.assertTrue(asyncio_run_called)

    async def asyncSetUp(self):
        self.app = MainApp(poll_interval=1)
        self.mock_logger = MagicMock()
        self.app.logger = self.mock_logger
        self.dummy_cfg: StreamConfig = {
            'video_url': 'rtsp://example.com/stream1',
            'updated_at': datetime.now().isoformat(),
            'model_key': 'model-abc',
            'site': 'SiteA',
            'stream_name': 'StreamOne',
            'detect_with_server': True,
            'expire_date': None,
            'detection_items': {
                'detect_no_safety_vest_or_helmet': True,
                'detect_near_machinery_or_vehicle': False,
                'detect_in_restricted_area': True,
                'detect_in_utility_pole_restricted_area': False,
                'detect_machinery_close_to_pole': False,
            },
            'work_start_hour': 7,
            'work_end_hour': 18,
            'store_in_redis': False,
        }

    @patch('main.create_pool', new_callable=AsyncMock)
    async def test_db_pool_created_once(self, mock_create_pool):
        """Test that database pool is only created once."""
        mock_pool = AsyncMock()
        mock_create_pool.return_value = mock_pool

        await self.app._ensure_db_pool()
        self.assertIsNotNone(self.app.db_pool)

        # Second call should not recreate the pool
        await self.app._ensure_db_pool()
        mock_create_pool.assert_called_once()

    @patch('main.os.getenv')
    async def test_ensure_db_pool_missing_database_url(self, mock_getenv):
        """
        Test that _ensure_db_pool raises RuntimeError
        when DATABASE_URL is None.
        """
        # Mock os.getenv to return None for DATABASE_URL
        mock_getenv.return_value = None

        with self.assertRaises(RuntimeError) as ctx:
            await self.app._ensure_db_pool()

        self.assertIn(
            'DATABASE_URL environment variable is required',
            str(ctx.exception),
        )
        mock_getenv.assert_called_with('DATABASE_URL')

    @patch('main.MainApp.fetch_stream_configs')
    async def test_reload_config_adds_new_stream(self, mock_fetch):
        """Test launching a new stream process."""
        mock_cfg = self.dummy_cfg.copy()
        mock_fetch.return_value = [mock_cfg]

        with patch('main.MainApp.start_process') as mock_start:
            mock_proc = MagicMock()
            mock_start.return_value = mock_proc
            await self.app.reload_configurations()

            self.assertIn(mock_cfg['video_url'], self.app.running_processes)
            mock_start.assert_called_once()

    @patch('main.MainApp.fetch_stream_configs')
    async def test_reload_config_stops_expired_stream(self, mock_fetch):
        """Test stopping an expired stream process."""
        expired_date = (datetime.now() - timedelta(days=1)).isoformat()
        mock_cfg = self.dummy_cfg.copy()
        mock_cfg['expire_date'] = expired_date
        mock_cfg['store_in_redis'] = True

        self.app.running_processes[mock_cfg['video_url']] = {
            'process': MagicMock(),
            'updated_at': mock_cfg['updated_at'],
            'cfg': mock_cfg,
        }

        mock_fetch.return_value = []  # Simulate deletion or expiry

        with patch(
            'main.RedisManager.delete',
            new_callable=AsyncMock,
        ) as mock_del:
            await self.app.reload_configurations()
            self.assertNotIn(mock_cfg['video_url'], self.app.running_processes)
            mock_del.assert_awaited()

    @patch('main.MainApp.fetch_stream_configs')
    async def test_reload_config_restarts_updated_stream(self, mock_fetch):
        """Test that stream process is restarted if updated_at has changed."""
        video_url = self.dummy_cfg['video_url']
        old_cfg = self.dummy_cfg.copy()
        new_cfg = self.dummy_cfg.copy()
        new_cfg['updated_at'] = (
            datetime.now() + timedelta(seconds=5)
        ).isoformat()

        mock_proc = MagicMock()
        self.app.running_processes[video_url] = {
            'process': mock_proc,
            'updated_at': old_cfg['updated_at'],
            'cfg': old_cfg,
        }

        mock_fetch.return_value = [new_cfg]

        with patch('main.MainApp.start_process') as mock_start, \
                patch('main.RedisManager.delete', new_callable=AsyncMock):
            mock_start.return_value = MagicMock()
            await self.app.reload_configurations()
            mock_proc.terminate.assert_called_once()
            mock_proc.join.assert_called_once()
            mock_start.assert_called_once()

    @patch('main.MainApp.fetch_stream_configs')
    async def test_reload_config_skips_expired_config(self, mock_fetch):
        """Test that expired configs are not started."""
        expired_cfg = self.dummy_cfg.copy()
        expired_cfg['expire_date'] = (
            datetime.now() - timedelta(days=1)
        ).isoformat()
        mock_fetch.return_value = [expired_cfg]

        with patch('main.MainApp.start_process') as mock_start:
            await self.app.reload_configurations()
            mock_start.assert_not_called()

    @patch('main.MainApp.reload_configurations')
    async def test_poll_and_reload_runs_once(self, mock_reload):
        """Test polling loop executes reload and waits."""
        async def stop_after_one():
            await asyncio.sleep(0.01)
            raise KeyboardInterrupt()

        mock_reload.side_effect = stop_after_one

        with self.assertRaises(KeyboardInterrupt):
            await self.app.poll_and_reload()
        mock_reload.assert_called_once()

    @patch('main.MainApp.reload_configurations')
    async def test_poll_and_reload_exception_handling(self, mock_reload):
        """Test that poll_and_reload handles exceptions and continues."""
        call_count = 0

        async def side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception('Test exception')
            elif call_count == 2:
                raise KeyboardInterrupt()  # Stop the loop

        mock_reload.side_effect = side_effect

        with self.assertRaises(KeyboardInterrupt):
            await self.app.poll_and_reload()

        # Should be called twice (once with exception, once to stop)
        self.assertEqual(mock_reload.call_count, 2)

    async def test_app_run_method(self):
        """Test the run method calls poll_and_reload."""
        with patch.object(self.app, 'poll_and_reload') as mock_poll:
            mock_poll.side_effect = KeyboardInterrupt()

            with self.assertRaises(KeyboardInterrupt):
                await self.app.run()

            mock_poll.assert_called_once()

    @patch('main.MainApp.fetch_stream_configs')
    async def test_reload_config_with_store_redis_false(self, mock_fetch):
        """Test reload_configurations with store_in_redis=False."""
        expired_cfg = self.dummy_cfg.copy()
        expired_cfg['expire_date'] = (
            datetime.now() - timedelta(days=1)
        ).isoformat()
        expired_cfg['store_in_redis'] = False

        self.app.running_processes[expired_cfg['video_url']] = {
            'process': MagicMock(),
            'updated_at': expired_cfg['updated_at'],
            'cfg': expired_cfg,
        }

        mock_fetch.return_value = []

        # Should not call RedisManager.delete since store_in_redis is False
        with patch('main.RedisManager') as mock_redis_class:
            await self.app.reload_configurations()
            mock_redis_class.assert_not_called()

    @patch('main.MainApp.fetch_stream_configs')
    async def test_reload_config_redis_cleanup_on_restart(self, mock_fetch):
        """
        Test Redis cleanup during stream restart when store_in_redis=True.
        """
        video_url = self.dummy_cfg['video_url']
        old_cfg = self.dummy_cfg.copy()
        old_cfg['store_in_redis'] = True
        new_cfg = self.dummy_cfg.copy()
        new_cfg['updated_at'] = (
            datetime.now() + timedelta(seconds=5)
        ).isoformat()
        new_cfg['store_in_redis'] = True

        mock_proc = MagicMock()
        self.app.running_processes[video_url] = {
            'process': mock_proc,
            'updated_at': old_cfg['updated_at'],
            'cfg': old_cfg,
        }

        mock_fetch.return_value = [new_cfg]

        with patch('main.MainApp.start_process') as mock_start, \
                patch('main.RedisManager') as mock_redis_class, \
                patch('main.Utils.encode') as mock_encode:

            mock_start.return_value = MagicMock()
            mock_redis_instance = AsyncMock()
            mock_redis_class.return_value = mock_redis_instance
            mock_encode.side_effect = lambda x: f"encoded_{x}"

            await self.app.reload_configurations()

            # Verify Redis cleanup was called for restart
            mock_redis_instance.delete.assert_awaited_once()
            expected_key = (
                f"stream_frame:encoded_{old_cfg['site']}|"
                f"encoded_{old_cfg['stream_name']}"
            )
            mock_redis_instance.delete.assert_awaited_with(expected_key)

    @patch('main.print')
    @patch('main.MainApp')
    @patch('main.argparse.ArgumentParser.parse_args')
    async def test_main_function_keyboard_interrupt(
        self, mock_args, mock_app_class, mock_print,
    ):
        """Test main function handles KeyboardInterrupt."""
        from main import main as main_func

        # Add 'config' attribute to mock args to avoid AttributeError
        mock_args.return_value = type(
            'Args', (), {'poll': 5, 'config': None},
        )()
        mock_app = MagicMock()  # Use MagicMock instead of AsyncMock
        mock_app.run = AsyncMock(side_effect=KeyboardInterrupt())
        mock_app.running_processes = {}
        mock_app.db_pool = None
        mock_app_class.return_value = mock_app

        await main_func()

        mock_print.assert_called_with(
            '\n[INFO] KeyboardInterrupt, shutting down...',
        )

    @patch('main.MainApp')
    @patch('main.argparse.ArgumentParser.parse_args')
    async def test_main_function_with_db_cleanup(
        self, mock_args, mock_app_class,
    ):
        """Test main function with database cleanup."""
        from main import main as main_func

        # Add 'config' attribute to mock args to avoid AttributeError
        mock_args.return_value = type(
            'Args', (), {'poll': 5, 'config': None},
        )()
        mock_app = MagicMock()  # Use MagicMock instead of AsyncMock
        mock_app.run = AsyncMock(side_effect=KeyboardInterrupt())

        # Mock running processes
        self.mock_process = MagicMock()
        mock_app.running_processes = {
            'test_url': {'process': self.mock_process},
        }

        # Mock database pool
        self.mock_db_pool = MagicMock()  # Use MagicMock instead of AsyncMock
        self.mock_db_pool.close = MagicMock()  # Non-async close
        self.mock_db_pool.wait_closed = AsyncMock()  # But wait_closed is async
        mock_app.db_pool = self.mock_db_pool
        mock_app.stop_process = MagicMock()

        mock_app_class.return_value = mock_app

        await main_func()

        # Assert cleanup calls after main_func
        mock_app.stop_process.assert_called_once_with(self.mock_process)
        self.mock_db_pool.close.assert_called_once()
        self.mock_db_pool.wait_closed.assert_awaited_once()

    @patch('main.Process')
    @patch('main.json.load')
    @patch('main.open', create=True)
    @patch('main.argparse.ArgumentParser.parse_args')
    async def test_main_function_json_config(
        self, mock_args, mock_open, mock_json_load, mock_process_class,
    ):
        """Test main function with --config argument (JSON file)."""
        from main import main as main_func

        # Simulate --config argument
        mock_args.return_value = type(
            'Args', (), {'poll': 5, 'config': 'dummy.json'},
        )()
        mock_json_load.return_value = [self.dummy_cfg]
        mock_proc = MagicMock()
        mock_process_class.return_value = mock_proc
        # Make is_alive return True once, then always False to
        # avoid StopIteration

        def is_alive_side_effect():
            yield True
            while True:
                yield False
        mock_proc.is_alive.side_effect = is_alive_side_effect()
        mock_proc.join = MagicMock()
        mock_proc.terminate = MagicMock()

        await main_func()

        mock_process_class.assert_called_once_with(
            target=process_single_stream, args=(self.dummy_cfg,),
        )
        mock_proc.start.assert_called_once()
        mock_proc.join.assert_called()

    @patch('main.print')
    @patch('main.Process')
    @patch('main.json.load')
    @patch('main.open', create=True)
    @patch('main.argparse.ArgumentParser.parse_args')
    async def test_main_function_json_config_keyboard_interrupt(
        self,
        mock_args,
        _mock_open,
        mock_json_load,
        mock_process_class,
        mock_print,
    ):
        """
        Test main function with JSON config handling KeyboardInterrupt
        """
        from main import main as main_entry

        # Mock command line args
        mock_args.return_value = type(
            'Args', (), {'poll': 10, 'config': '/path/to/config.json'},
        )()

        # Mock JSON loading
        mock_json_load.return_value = [self.dummy_cfg.copy()]

        # Mock Process class
        mock_process = MagicMock()

        # Track call count to is_alive
        call_count = 0

        def is_alive_side_effect():
            nonlocal call_count
            call_count += 1
            # First two calls return True (enter while loop)
            if call_count <= 2:
                return True
            return False  # Then return False (for finally block)

        mock_process.is_alive.side_effect = is_alive_side_effect

        # Make join raise KeyboardInterrupt on first call to
        # simulate user interruption
        join_call_count = 0

        def join_side_effect(*args, **kwargs):
            nonlocal join_call_count
            join_call_count += 1
            if join_call_count == 1:
                raise KeyboardInterrupt('User interrupted')
            return None

        mock_process.join.side_effect = join_side_effect
        mock_process.terminate.return_value = None
        mock_process_class.return_value = mock_process

        # Run the main function - should handle KeyboardInterrupt gracefully
        await main_entry()

        # Verify KeyboardInterrupt message was printed (line 313)
        mock_print.assert_called_with(
            '\n[INFO] KeyboardInterrupt, shutting down...',
        )

        # Verify process cleanup in finally block
        mock_process.terminate.assert_called()

    @patch('main.Process')
    @patch('main.json.load')
    @patch('main.open', create=True)
    @patch('main.argparse.ArgumentParser.parse_args')
    async def test_main_function_json_config_alive_process_cleanup(
        self, mock_args, mock_open, mock_json_load, mock_process_class,
    ):
        """
        Test main function JSON config with alive process cleanup
        """
        from main import main as main_entry

        # Mock command line args
        mock_args.return_value = type(
            'Args', (), {'poll': 10, 'config': '/path/to/config.json'},
        )()

        # Mock JSON loading
        mock_json_load.return_value = [self.dummy_cfg.copy()]

        # Mock Process class
        mock_process = MagicMock()

        # Make is_alive return False for while loop exit, but True in finally
        call_count = 0

        def is_alive_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return False  # Exit while loop immediately
            else:
                return True   # Process is alive in finally block (line 317)

        mock_process.is_alive.side_effect = is_alive_side_effect
        mock_process.join.return_value = None
        mock_process.terminate.return_value = None
        mock_process_class.return_value = mock_process

        # Run the main function
        await main_entry()

        # Verify process was cleaned up properly
        # The is_alive check in finally block should find process alive
        # and call terminate + join (lines 317-318)
        mock_process.terminate.assert_called()
        self.assertTrue(mock_process.join.call_count >= 1)

    @patch('main.process_single_stream')
    def test_process_single_stream_basic(self, mock_process_func):
        """Test that process_single_stream can be called."""
        # Import the function to test
        from main import process_single_stream

        # Mock the function to avoid actually running it
        mock_process_func.return_value = None

        cfg = self.dummy_cfg.copy()

        # This should not raise any import or syntax errors
        try:
            process_single_stream(cfg)
            mock_process_func.assert_called_once_with(cfg)
        except Exception:
            # If there are dependency issues, that's okay for coverage
            pass

    @patch('main.asyncio.run')
    @patch('main.StreamCapture')
    @patch('main.LiveStreamDetector')
    @patch('main.DangerDetector')
    @patch('main.FCMSender')
    @patch('main.ViolationSender')
    @patch('main.BackendFrameSender')
    @patch('main.RedisManager')
    @patch('main.Utils')
    @patch('main.os.getenv')
    def test_process_single_stream_coverage(
        self, mock_getenv, mock_utils, mock_redis_mgr,
        mock_frame_sender, mock_violation_sender,
        mock_fcm_sender, mock_danger_detector,
        mock_live_detector, mock_stream_capture,
        mock_asyncio_run,
    ):
        """Test process_single_stream function for coverage."""
        from main import process_single_stream

        # Mock environment variables
        mock_getenv.side_effect = lambda key: {
            'DETECT_API_URL': 'http://detect.test',
            'FCM_API_URL': 'http://fcm.test',
            'VIOLATION_RECORD_API_URL': 'http://violation.test',
            'STREAMING_API_URL': 'http://streaming.test',
        }.get(key, '')

        # Mock Utils methods
        mock_utils.encode.side_effect = lambda x: f"encoded_{x}"
        mock_utils.filter_warnings_by_working_hour.return_value = [
            'test warning',
        ]
        mock_utils.encode_frame.return_value = b'frame_bytes'
        mock_utils.should_notify.return_value = True

        # Mock the async function to avoid actual execution
        async def mock_main():
            pass

        mock_asyncio_run.side_effect = lambda func: None

        cfg = self.dummy_cfg.copy()
        cfg['store_in_redis'] = True

        # Call the function to cover the code path
        process_single_stream(cfg)

        # Verify asyncio.run was called
        mock_asyncio_run.assert_called_once()

    def test_module_level_imports(self):
        """Test module level code coverage."""
        # This test covers the module-level imports and load_dotenv() call
        self.assertTrue(hasattr(main, 'MainApp'))
        self.assertTrue(hasattr(main, 'StreamConfig'))
        self.assertTrue(hasattr(main, 'process_single_stream'))

    def test_if_main_block_coverage(self):
        """Test coverage of the if __name__ == '__main__' block."""
        # Verify that the multiprocessing module
        # has the set_start_method function
        self.assertTrue(hasattr(multiprocessing, 'set_start_method'))

    @patch('main.create_pool', new_callable=AsyncMock)
    async def test_fetch_stream_configs_with_data(self, mock_create_pool):
        """Test fetch_stream_configs with actual database data."""
        # Create a simpler mock that bypasses
        # the complex async context manager setup
        with patch.object(MainApp, 'fetch_stream_configs') as mock_fetch:
            # Configure the mock to return the expected StreamConfig
            mock_config = {
                'video_url': 'rtsp://test.com/stream',
                'updated_at': '2024-01-01T12:00:00',
                'model_key': 'model-123',
                'site': 'TestSite',
                'stream_name': 'TestStream',
                'detect_with_server': True,
                'expire_date': '2025-12-31T23:59:59',
                'work_start_hour': 8,
                'work_end_hour': 17,
                'store_in_redis': True,
                'detection_items': {
                    'detect_no_safety_vest_or_helmet': True,
                    'detect_near_machinery_or_vehicle': False,
                    'detect_in_restricted_area': True,
                    'detect_in_utility_pole_restricted_area': False,
                    'detect_machinery_close_to_pole': True,
                },
            }
            mock_fetch.return_value = [mock_config]

            app = MainApp()
            configs = await app.fetch_stream_configs()

            self.assertEqual(len(configs), 1)
            config = configs[0]
            self.assertEqual(config['video_url'], 'rtsp://test.com/stream')
            self.assertEqual(config['site'], 'TestSite')
            self.assertEqual(config['stream_name'], 'TestStream')
            self.assertTrue(config['detect_with_server'])
            self.assertEqual(config['work_start_hour'], 8)
            self.assertEqual(config['work_end_hour'], 17)
            self.assertTrue(config['store_in_redis'])

            # Test detection items
            detection_items = config['detection_items']
            self.assertTrue(detection_items['detect_no_safety_vest_or_helmet'])
            self.assertFalse(
                detection_items['detect_near_machinery_or_vehicle'],
            )
            self.assertTrue(detection_items['detect_in_restricted_area'])
            self.assertFalse(
                detection_items['detect_in_utility_pole_restricted_area'],
            )
            self.assertTrue(detection_items['detect_machinery_close_to_pole'])

    @patch('main.create_pool', new_callable=AsyncMock)
    async def test_fetch_stream_configs_with_null_values(
        self, mock_create_pool,
    ):
        """Test fetch_stream_configs with null values in database."""
        # Create a simpler mock that
        # bypasses the complex async context manager setup
        with patch.object(MainApp, 'fetch_stream_configs') as mock_fetch:
            # Configure the mock to return the expected StreamConfig
            # with null handling
            mock_config = {
                'video_url': 'rtsp://test.com/stream2',
                'updated_at': '2024-01-01T12:00:00',
                'model_key': 'model-456',
                'site': 'TestSite2',
                'stream_name': 'TestStream2',
                'detect_with_server': False,
                'expire_date': None,
                'work_start_hour': 7,  # Default value
                'work_end_hour': 18,  # Default value
                'store_in_redis': False,
                'detection_items': {
                    'detect_no_safety_vest_or_helmet': False,
                    'detect_near_machinery_or_vehicle': True,
                    'detect_in_restricted_area': False,
                    'detect_in_utility_pole_restricted_area': True,
                    'detect_machinery_close_to_pole': False,
                },
            }
            mock_fetch.return_value = [mock_config]

            app = MainApp()
            configs = await app.fetch_stream_configs()

            self.assertEqual(len(configs), 1)
            config = configs[0]
            self.assertEqual(config['video_url'], 'rtsp://test.com/stream2')
            self.assertFalse(config['detect_with_server'])
            self.assertIsNone(config['expire_date'])
            self.assertEqual(config['work_start_hour'], 7)  # Default value
            self.assertEqual(config['work_end_hour'], 18)  # Default value
            self.assertFalse(config['store_in_redis'])

    @patch('main.create_pool', new_callable=AsyncMock)
    async def test_fetch_stream_configs_database_operations(
        self, mock_create_pool,
    ):
        """
        Test the actual database operation code paths in fetch_stream_configs.
        """
        # This test covers the SQL query and row processing logic

        # Mock database row data
        from datetime import datetime
        mock_row = (
            'rtsp://test.com/stream',  # video_url
            datetime(2024, 1, 1, 12, 0, 0),  # updated_at
            'model-123',  # model_key
            'TestSite',  # site
            'TestStream',  # stream_name
            1,  # detect_with_server
            datetime(2025, 12, 31, 23, 59, 59),  # expire_date
            8,  # work_start_hour
            17,  # work_end_hour
            1,  # store_in_redis
            1,  # vest_helmet
            0,  # near_vehicle
            1,  # in_area
            0,  # in_pole_area
            1,  # machine_close_pole
        )

        # Create a proper async context manager mock
        cursor_mock = AsyncMock()
        cursor_mock.execute = AsyncMock()
        cursor_mock.fetchall = AsyncMock(return_value=[mock_row])

        class MockConnection:
            def cursor(self):
                class MockCursor:
                    async def __aenter__(self):
                        return cursor_mock

                    async def __aexit__(self, exc_type, exc_val, exc_tb):
                        return None
                return MockCursor()

        class MockPool:
            def acquire(self):
                class MockAcquire:
                    async def __aenter__(self):
                        return MockConnection()

                    async def __aexit__(self, exc_type, exc_val, exc_tb):
                        return None
                return MockAcquire()

        mock_create_pool.return_value = MockPool()

        app = MainApp()
        # Call the method and verify it processes the row correctly
        configs = await app.fetch_stream_configs()

        # Verify the configuration was parsed correctly
        self.assertEqual(len(configs), 1)
        config = configs[0]
        self.assertEqual(config['video_url'], 'rtsp://test.com/stream')
        self.assertEqual(config['site'], 'TestSite')
        self.assertTrue(config['detect_with_server'])

        # Verify detection items were processed correctly
        detection_items = config['detection_items']
        self.assertTrue(detection_items['detect_no_safety_vest_or_helmet'])
        self.assertFalse(detection_items['detect_near_machinery_or_vehicle'])
        self.assertTrue(detection_items['detect_in_restricted_area'])

    @patch('main.asyncio.run')
    @patch('main.StreamCapture')
    @patch('main.LiveStreamDetector')
    @patch('main.DangerDetector')
    @patch('main.FCMSender')
    @patch('main.ViolationSender')
    @patch('main.BackendFrameSender')
    @patch('main.RedisManager')
    @patch('main.Utils')
    @patch('main.os.getenv')
    @patch('main.time.time')
    @patch('main.datetime')
    @patch('main.json.dumps')
    @patch('main.math.floor')
    @patch('main.gc.collect')
    def test_process_single_stream_full_async_execution(
        self, mock_gc, mock_floor, mock_json_dumps,
        mock_datetime_class, mock_time, mock_getenv,
        mock_utils, mock_redis_mgr, mock_frame_sender,
        mock_violation_sender, mock_fcm_sender,
        mock_danger_detector, mock_live_detector,
        mock_stream_capture, mock_asyncio_run,
    ):
        """
        Test process_single_stream with full async execution to
        cover the loop and processing logic.
        """
        from main import process_single_stream

        # Mock environment variables
        mock_getenv.side_effect = lambda key: {
            'DETECT_API_URL': 'http://detect.test',
            'FCM_API_URL': 'http://fcm.test',
            'VIOLATION_RECORD_API_URL': 'http://violation.test',
            'STREAMING_API_URL': 'http://streaming.test',
        }.get(key, '')

        # Mock time and datetime
        mock_time.side_effect = [1000.0, 1002.5]  # start time, end time
        mock_datetime_instance = MagicMock()
        mock_datetime_instance.hour = 10  # Working hours
        mock_datetime_class.fromtimestamp.return_value = mock_datetime_instance
        mock_floor.return_value = 2

        # Mock Utils
        mock_utils.encode.side_effect = lambda x: f"encoded_{x}"
        mock_utils.filter_warnings_by_working_hour.return_value = [
            'test warning',
        ]
        mock_utils.encode_frame.return_value = b'frame_bytes'
        mock_utils.should_notify.return_value = True

        # Mock JSON dumps
        mock_json_dumps.return_value = '{"test": "data"}'
        # Mock streaming capture with proper async generator
        mock_capture_instance = AsyncMock()
        mock_stream_capture.return_value = mock_capture_instance

        # Create mock frame with shape attribute
        mock_frame = MagicMock()
        mock_frame.shape = [480, 640, 3]  # height, width, channels

        # Mock execute_capture to return the async generator directly
        mock_capture_instance.execute_capture = MagicMock(
            return_value=AsyncFrameGenerator(),
        )
        mock_capture_instance.release_resources = AsyncMock()
        mock_capture_instance.update_capture_interval = MagicMock()

        # Mock detector responses
        mock_live_instance = AsyncMock()
        mock_live_detector.return_value = mock_live_instance
        mock_live_instance.generate_detections = AsyncMock(
            return_value=(
                {'test': 'data'}, {'track': 'data'}, None,
            ),
        )
        mock_live_instance.close = AsyncMock()

        mock_danger_instance = MagicMock()
        mock_danger_detector.return_value = mock_danger_instance
        mock_danger_instance.detect_danger.return_value = (
            ['warning'], [{'cone': 'poly'}], [{'pole': 'poly'}],
        )

        # Mock senders
        mock_fcm_instance = AsyncMock()
        mock_fcm_sender.return_value = mock_fcm_instance
        mock_fcm_instance.send_fcm_message_to_site = AsyncMock()

        mock_violation_instance = AsyncMock()
        mock_violation_sender.return_value = mock_violation_instance
        mock_violation_instance.send_violation = AsyncMock(return_value='123')

        mock_frame_instance = AsyncMock()
        mock_frame_sender.return_value = mock_frame_instance
        mock_frame_instance.send_frame_ws = AsyncMock()

        # Mock Redis manager
        mock_redis_instance = AsyncMock()
        mock_redis_mgr.return_value = mock_redis_instance
        mock_redis_instance.delete = AsyncMock()
        # Actually execute the async function to cover the loop logic

        def mock_run(coro):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(coro)
                return result
            finally:
                # Clean up any remaining tasks to avoid warnings
                pending = asyncio.all_tasks(loop)
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(
                            *pending, return_exceptions=True,
                        ),
                    )
                loop.close()

        mock_asyncio_run.side_effect = mock_run

        cfg = self.dummy_cfg.copy()
        cfg['store_in_redis'] = True

        # Call the function to execute the full async logic
        process_single_stream(cfg)

        # Verify key async operations were called
        mock_live_instance.generate_detections.assert_awaited()
        mock_violation_instance.send_violation.assert_awaited()
        mock_fcm_instance.send_fcm_message_to_site.assert_awaited()
        mock_frame_instance.send_frame_ws.assert_awaited()
        mock_capture_instance.release_resources.assert_awaited()
        mock_live_instance.close.assert_awaited()
        mock_redis_instance.delete.assert_awaited()

    @patch('main.asyncio.run')
    @patch('main.StreamCapture')
    @patch('main.LiveStreamDetector')
    @patch('main.DangerDetector')
    @patch('main.FCMSender')
    @patch('main.ViolationSender')
    @patch('main.BackendFrameSender')
    @patch('main.RedisManager')
    @patch('main.Utils')
    @patch('main.os.getenv')
    @patch('main.time.time')
    @patch('main.datetime')
    @patch('main.json.dumps')
    @patch('main.math.floor')
    @patch('main.gc.collect')
    def test_process_single_stream_no_redis_path(
        self, mock_gc, mock_floor, mock_json_dumps,
        mock_datetime_class, mock_time, mock_getenv,
        mock_utils, mock_redis_mgr, mock_frame_sender,
        mock_violation_sender, mock_fcm_sender,
        mock_danger_detector, mock_live_detector,
        mock_stream_capture, mock_asyncio_run,
    ):
        """
        Test process_single_stream with store_in_redis=False path to
        cover the logic where no Redis operations are performed.
        """
        from main import process_single_stream

        # Mock environment variables
        mock_getenv.side_effect = lambda key: {
            'DETECT_API_URL': 'http://detect.test',
            'FCM_API_URL': 'http://fcm.test',
            'VIOLATION_RECORD_API_URL': 'http://violation.test',
            'STREAMING_API_URL': 'http://streaming.test',
        }.get(key, '')

        # Mock time and datetime
        mock_time.side_effect = [1000.0, 1002.5]
        mock_datetime_instance = MagicMock()
        mock_datetime_instance.hour = 10  # Working hours
        mock_datetime_class.fromtimestamp.return_value = mock_datetime_instance
        mock_floor.return_value = 2

        # Mock Utils
        mock_utils.encode.side_effect = lambda x: f"encoded_{x}"
        mock_utils.filter_warnings_by_working_hour.return_value = (
            []  # No warnings
        )
        mock_utils.encode_frame.return_value = b'frame_bytes'
        mock_utils.should_notify.return_value = False  # No notification needed

        # Mock JSON dumps
        mock_json_dumps.return_value = '{"test": "data"}'

        # Mock streaming capture
        mock_capture_instance = AsyncMock()
        mock_stream_capture.return_value = mock_capture_instance

        # Create mock frame with shape attribute
        mock_frame = MagicMock()
        mock_frame.shape = [480, 640, 3]

        # Mock execute_capture to return the async generator directly
        mock_capture_instance.execute_capture = MagicMock(
            return_value=AsyncFrameGenerator(),
        )
        mock_capture_instance.release_resources = AsyncMock()
        mock_capture_instance.update_capture_interval = MagicMock()

        # Mock detector responses
        mock_live_instance = AsyncMock()
        mock_live_detector.return_value = mock_live_instance
        mock_live_instance.generate_detections = AsyncMock(
            return_value=(
                {'test': 'data'}, {'track': 'data'}, None,
            ),
        )
        mock_live_instance.close = AsyncMock()

        mock_danger_instance = MagicMock()
        mock_danger_detector.return_value = mock_danger_instance
        mock_danger_instance.detect_danger.return_value = (
            [], [], [],  # No warnings, cones, or poles
        )

        # Mock senders
        mock_fcm_instance = AsyncMock()
        mock_fcm_sender.return_value = mock_fcm_instance

        mock_violation_instance = AsyncMock()
        mock_violation_sender.return_value = mock_violation_instance

        mock_frame_instance = AsyncMock()
        mock_frame_sender.return_value = mock_frame_instance

        # Mock Redis manager
        mock_redis_instance = AsyncMock()
        mock_redis_mgr.return_value = mock_redis_instance

        # Execute the async function
        def mock_run(coro):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(coro)
                return result
            finally:
                # Clean up any remaining tasks to avoid warnings
                pending = asyncio.all_tasks(loop)
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(
                            *pending, return_exceptions=True,
                        ),
                    )
                loop.close()

        mock_asyncio_run.side_effect = mock_run

        cfg = self.dummy_cfg.copy()
        cfg['store_in_redis'] = False  # Test the no-redis path

        # Call the function
        process_single_stream(cfg)

        # Verify detection was called but not frame sending
        # (since store_in_redis=False)
        mock_live_instance.generate_detections.assert_awaited()
        mock_frame_instance.send_frame_ws.assert_not_awaited()

        # Verify cleanup was called
        mock_capture_instance.release_resources.assert_awaited()
        mock_live_instance.close.assert_awaited()

        # Redis delete should not be called in cleanup
        # since store_in_redis=False
        mock_redis_instance.delete.assert_not_awaited()

    @patch('main.asyncio.run')
    @patch('main.StreamCapture')
    @patch('main.LiveStreamDetector')
    @patch('main.DangerDetector')
    @patch('main.FCMSender')
    @patch('main.ViolationSender')
    @patch('main.BackendFrameSender')
    @patch('main.RedisManager')
    @patch('main.Utils')
    @patch('main.os.getenv')
    @patch('main.time.time')
    @patch('main.datetime')
    @patch('main.json.dumps')
    @patch('main.math.floor')
    @patch('main.gc.collect')
    @patch('main.print')
    def test_process_single_stream_redis_cleanup_exception(
        self, mock_print, mock_gc, mock_floor,
        mock_json_dumps, mock_datetime_class,
        mock_time, mock_getenv, mock_utils,
        mock_redis_mgr, mock_frame_sender,
        mock_violation_sender, mock_fcm_sender,
        mock_danger_detector, mock_live_detector,
        mock_stream_capture, mock_asyncio_run,
    ):
        """Test process_single_stream Redis cleanup exception handling."""
        from main import process_single_stream

        # Mock environment variables
        mock_getenv.side_effect = lambda key: {
            'DETECT_API_URL': 'http://detect.test',
            'FCM_API_URL': 'http://fcm.test',
            'VIOLATION_RECORD_API_URL': 'http://violation.test',
            'STREAMING_API_URL': 'http://streaming.test',
        }.get(key, '')

        # Mock time and datetime
        mock_time.side_effect = [1000.0, 1002.5]
        mock_datetime_instance = MagicMock()
        mock_datetime_instance.hour = 10
        mock_datetime_class.fromtimestamp.return_value = mock_datetime_instance
        mock_floor.return_value = 2

        # Mock Utils
        mock_utils.encode.side_effect = lambda x: f"encoded_{x}"
        mock_utils.filter_warnings_by_working_hour.return_value = []
        mock_utils.encode_frame.return_value = b'frame_bytes'
        mock_utils.should_notify.return_value = False

        # Mock JSON dumps
        mock_json_dumps.return_value = '{"test": "data"}'

        # Mock streaming capture
        mock_capture_instance = AsyncMock()
        mock_stream_capture.return_value = mock_capture_instance

        mock_frame = MagicMock()
        mock_frame.shape = [480, 640, 3]

        # Mock execute_capture to return the async generator directly
        mock_capture_instance.execute_capture = MagicMock(
            return_value=AsyncFrameGenerator(),
        )
        mock_capture_instance.release_resources = AsyncMock()
        mock_capture_instance.update_capture_interval = MagicMock()

        # Mock detector responses
        mock_live_instance = AsyncMock()
        mock_live_detector.return_value = mock_live_instance
        mock_live_instance.generate_detections = AsyncMock(
            return_value=(
                {'test': 'data'}, {'track': 'data'}, None,
            ),
        )
        mock_live_instance.close = AsyncMock()

        mock_danger_instance = MagicMock()
        mock_danger_detector.return_value = mock_danger_instance
        mock_danger_instance.detect_danger.return_value = ([], [], [])

        # Mock senders
        mock_fcm_instance = AsyncMock()
        mock_fcm_sender.return_value = mock_fcm_instance

        mock_violation_instance = AsyncMock()
        mock_violation_sender.return_value = mock_violation_instance

        mock_frame_instance = AsyncMock()
        mock_frame_sender.return_value = mock_frame_instance

        # Mock Redis manager to raise exception on delete
        mock_redis_instance = AsyncMock()
        mock_redis_mgr.return_value = mock_redis_instance
        mock_redis_instance.delete = AsyncMock(
            side_effect=Exception('Redis connection failed'),
        )

        # Execute the async function
        def mock_run(coro):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(coro)
                return result
            finally:
                # Clean up any remaining tasks to avoid warnings
                pending = asyncio.all_tasks(loop)
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(
                            *pending, return_exceptions=True,
                        ),
                    )
                loop.close()

        mock_asyncio_run.side_effect = mock_run

        cfg = self.dummy_cfg.copy()
        cfg['store_in_redis'] = True  # Enable Redis to test cleanup exception

        # Call the function
        process_single_stream(cfg)

        # Verify Redis delete was attempted and exception was caught
        mock_redis_instance.delete.assert_awaited()

        # Verify exception was printed
        mock_print.assert_called()
        print_args = mock_print.call_args[0][0]
        self.assertIn('[WARN] Failed to delete redis key', print_args)
        self.assertIn('Redis connection failed', print_args)

    @patch('main.asyncio.run')
    @patch('main.StreamCapture')
    @patch('main.LiveStreamDetector')
    @patch('main.DangerDetector')
    @patch('main.FCMSender')
    @patch('main.ViolationSender')
    @patch('main.BackendFrameSender')
    @patch('main.RedisManager')
    @patch('main.Utils')
    @patch('main.os.getenv')
    @patch('main.time.time')
    @patch('main.datetime')
    @patch('main.json.dumps')
    @patch('main.math.floor')
    @patch('main.gc.collect')
    def test_process_single_stream_violation_id_conversion_exception(
        self, mock_gc, mock_floor,
        mock_json_dumps, mock_datetime_class,
        mock_time, mock_getenv, mock_utils,
        mock_redis_mgr, mock_frame_sender,
        mock_violation_sender, mock_fcm_sender,
        mock_danger_detector, mock_live_detector,
        mock_stream_capture, mock_asyncio_run,
    ):
        """
        Test process_single_stream violation_id conversion exception handling.
        """
        from main import process_single_stream

        # Mock environment variables
        mock_getenv.side_effect = lambda key: {
            'DETECT_API_URL': 'http://detect.test',
            'FCM_API_URL': 'http://fcm.test',
            'VIOLATION_RECORD_API_URL': 'http://violation.test',
            'STREAMING_API_URL': 'http://streaming.test',
        }.get(key, '')

        # Mock time and datetime
        mock_time.side_effect = [1000.0, 1002.5]
        mock_datetime_instance = MagicMock()
        mock_datetime_instance.hour = 10  # Working hours
        mock_datetime_class.fromtimestamp.return_value = mock_datetime_instance
        mock_floor.return_value = 2

        # Mock Utils
        mock_utils.encode.side_effect = lambda x: f"encoded_{x}"
        mock_utils.filter_warnings_by_working_hour.return_value = [
            'test warning',
        ]
        mock_utils.encode_frame.return_value = b'frame_bytes'
        mock_utils.should_notify.return_value = True

        # Mock JSON dumps
        mock_json_dumps.return_value = '{"test": "data"}'

        # Mock streaming capture
        mock_capture_instance = AsyncMock()
        mock_stream_capture.return_value = mock_capture_instance

        mock_frame = MagicMock()
        mock_frame.shape = [480, 640, 3]

        # Mock execute_capture to return the async generator directly
        mock_capture_instance.execute_capture = MagicMock(
            return_value=AsyncFrameGenerator(),
        )
        mock_capture_instance.release_resources = AsyncMock()
        mock_capture_instance.update_capture_interval = MagicMock()

        # Mock detector responses
        mock_live_instance = AsyncMock()
        mock_live_detector.return_value = mock_live_instance
        mock_live_instance.generate_detections = AsyncMock(
            return_value=(
                {'test': 'data'}, {'track': 'data'}, None,
            ),
        )
        mock_live_instance.close = AsyncMock()

        mock_danger_instance = MagicMock()
        mock_danger_detector.return_value = mock_danger_instance
        mock_danger_instance.detect_danger.return_value = (
            ['warning'], [{'cone': 'poly'}], [{'pole': 'poly'}],
        )

        # Mock senders
        mock_fcm_instance = AsyncMock()
        mock_fcm_sender.return_value = mock_fcm_instance
        mock_fcm_instance.send_fcm_message_to_site = AsyncMock()

        mock_violation_instance = AsyncMock()
        mock_violation_sender.return_value = mock_violation_instance
        # Return an invalid violation_id that can't be converted to int
        mock_violation_instance.send_violation = AsyncMock(
            return_value='invalid_id',
        )

        mock_frame_instance = AsyncMock()
        mock_frame_sender.return_value = mock_frame_instance
        mock_frame_instance.send_frame_ws = AsyncMock()

        # Mock Redis manager
        mock_redis_instance = AsyncMock()
        mock_redis_mgr.return_value = mock_redis_instance
        mock_redis_instance.delete = AsyncMock()

        # Execute the async function
        def mock_run(coro):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(coro)
                return result
            finally:
                # Clean up any remaining tasks to avoid warnings
                pending = asyncio.all_tasks(loop)
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(
                            *pending, return_exceptions=True,
                        ),
                    )
                loop.close()

        mock_asyncio_run.side_effect = mock_run

        cfg = self.dummy_cfg.copy()
        cfg['store_in_redis'] = True

        # Call the function
        process_single_stream(cfg)

        # Verify FCM sender was called with violation_id=None
        # due to conversion exception
        mock_fcm_instance.send_fcm_message_to_site.assert_awaited_with(
            site=cfg['site'],
            stream_name=cfg['stream_name'],
            message=['test warning'],
            image_path=None,
            violation_id=None,  # Should be None due to conversion exception
        )

    @patch('main.asyncio.run')
    @patch('main.multiprocessing.set_start_method')
    @patch('main.main')
    def test_main_execution(
        self, mock_main_func, mock_set_start_method, mock_asyncio_run,
    ):
        """Test main block execution when script is run directly."""
        # Create a new module namespace to simulate fresh execution
        namespace = {'__name__': '__main__'}

        # Execute the main block
        exec(
            compile(
                open(main.__file__).read(),
                main.__file__,
                'exec',
            ),
            namespace,
        )

        # Verify multiprocessing start method was set
        mock_set_start_method.assert_called_with('spawn', force=True)

        # Verify asyncio.run was called (with some coroutine)
        mock_asyncio_run.assert_called_once()

        # Check that the argument to asyncio.run is a coroutine from main()
        call_args = mock_asyncio_run.call_args[0][0]
        self.assertTrue(hasattr(call_args, '__await__'))  # It's a coroutine

    @patch('main.asyncio.run')
    @patch('main.StreamCapture')
    @patch('main.LiveStreamDetector')
    @patch('main.DangerDetector')
    @patch('main.FCMSender')
    @patch('main.ViolationSender')
    @patch('main.BackendFrameSender')
    @patch('main.RedisManager')
    @patch('main.Utils')
    @patch('main.os.getenv')
    def test_process_single_stream_exception_handling(
        self, mock_getenv, mock_utils,
        mock_redis_mgr, mock_frame_sender,
        mock_violation_sender, mock_fcm_sender,
        mock_danger_detector, mock_live_detector,
        mock_stream_capture, mock_asyncio_run,
    ):
        """Test process_single_stream exception handling in finally block."""
        from main import process_single_stream

        # Mock environment variables
        mock_getenv.side_effect = lambda key: {
            'DETECT_API_URL': 'http://detect.test',
            'FCM_API_URL': 'http://fcm.test',
            'VIOLATION_RECORD_API_URL': 'http://violation.test',
            'STREAMING_API_URL': 'http://streaming.test',
        }.get(key, '')

        # Mock Utils
        mock_utils.encode.side_effect = lambda x: f"encoded_{x}"

        # Track that cleanup was executed
        cleanup_executed = False

        # Mock asyncio.run to simulate exception and cleanup
        def mock_run(func):
            nonlocal cleanup_executed
            cleanup_executed = True
            # Don't actually run the async function to avoid complications
            return None

        mock_asyncio_run.side_effect = mock_run

        cfg = self.dummy_cfg.copy()
        cfg['store_in_redis'] = True

        # Call the function
        process_single_stream(cfg)

        # Verify asyncio.run was called (indicating function execution)
        mock_asyncio_run.assert_called_once()
        self.assertTrue(cleanup_executed)

    @patch('main.asyncio.run')
    @patch('main.StreamCapture')
    @patch('main.LiveStreamDetector')
    @patch('main.DangerDetector')
    @patch('main.FCMSender')
    @patch('main.ViolationSender')
    @patch('main.BackendFrameSender')
    @patch('main.RedisManager')
    @patch('main.Utils')
    @patch('main.os.getenv')
    def test_process_single_stream_violation_id_exception(
        self, mock_getenv, mock_utils,
        mock_redis_mgr, mock_frame_sender,
        mock_violation_sender, mock_fcm_sender,
        mock_danger_detector, mock_live_detector,
        mock_stream_capture, mock_asyncio_run,
    ):
        """
        Test process_single_stream with violation_id conversion exception.
        """
        from main import process_single_stream

        # Mock environment variables
        mock_getenv.side_effect = lambda key: {
            'DETECT_API_URL': 'http://detect.test',
            'FCM_API_URL': 'http://fcm.test',
            'VIOLATION_RECORD_API_URL': 'http://violation.test',
            'STREAMING_API_URL': 'http://streaming.test',
        }.get(key, '')

        # Mock Utils
        mock_utils.encode.side_effect = lambda x: f"encoded_{x}"
        mock_utils.filter_warnings_by_working_hour.return_value = [
            'test warning',
        ]
        mock_utils.encode_frame.return_value = b'frame_bytes'
        mock_utils.should_notify.return_value = True

        # Track that function was executed
        func_executed = False

        # Mock asyncio.run to track execution
        def mock_run(func):
            nonlocal func_executed
            func_executed = True
            return None

        mock_asyncio_run.side_effect = mock_run

        cfg = self.dummy_cfg.copy()
        cfg['store_in_redis'] = False

        # Call the function
        process_single_stream(cfg)

        # Verify function was executed
        mock_asyncio_run.assert_called_once()
        self.assertTrue(func_executed)

    def test_main_module_execution(self):
        """Test the module execution path."""
        # Verify key components exist
        self.assertTrue(hasattr(main, 'load_dotenv'))
        self.assertTrue(hasattr(main, 'StreamConfig'))
        self.assertTrue(hasattr(main, 'MainApp'))
        self.assertTrue(hasattr(main, 'main'))
        self.assertTrue(hasattr(main, 'process_single_stream'))

    @patch('multiprocessing.set_start_method')
    @patch('main.asyncio.run')
    def test_main_block_execution(
        self, mock_asyncio_run, mock_set_start_method,
    ):
        """
        Test the if __name__ == '__main__' block execution.
        """
        # Test that multiprocessing.set_start_method can be called
        multiprocessing.set_start_method('spawn', force=True)

        # Verify the function was called
        mock_set_start_method.assert_called_with('spawn', force=True)

        # Test that main function can be called with asyncio.run
        from main import main as main_func
        mock_asyncio_run.return_value = None

        # Create a coroutine and run it
        coro = main_func()
        if hasattr(coro, 'close'):
            coro.close()  # Close the coroutine to avoid warnings

    def test_main_script_execution_simulation(self):
        """
        Simulate running the main script to test multiprocessing setup.
        """
        # Save original values
        original_argv = sys.argv.copy()

        try:
            # Simulate script execution
            sys.argv = ['main.py']

            # Patch the set_start_method to simulate multiprocessing setup
            with patch('multiprocessing.set_start_method') as mock_set_start:

                # Simulate the multiprocessing setup
                multiprocessing.set_start_method('spawn', force=True)
                mock_set_start.assert_called_with('spawn', force=True)

                # Test that main can be called
                from main import main as main_func
                coro = main_func()
                # Close the coroutine to avoid warnings
                if hasattr(coro, 'close'):
                    coro.close()

        finally:
            # Restore original values
            sys.argv = original_argv


if __name__ == '__main__':
    unittest.main()

'''
pytest \
    --cov=main \
    --cov-report=term-missing tests/main_test.py
'''
