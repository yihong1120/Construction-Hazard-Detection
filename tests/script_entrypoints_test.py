from __future__ import annotations

import logging
import runpy
import sys
import unittest
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_as_script(relative_path: str) -> dict[str, object]:
    """Execute a project script exactly as ``python path/to/script.py``
    would."""
    return runpy.run_path(
        str(PROJECT_ROOT / relative_path),
        run_name='__main__',
    )


class ScriptEntrypointTests(unittest.TestCase):
    """Verify direct-execution guards delegate to their controlled runners."""

    def test_fastapi_entrypoints_delegate_to_uvicorn(self) -> None:
        """Service scripts should hand their application to Uvicorn once."""
        scripts = (
            'examples/db_management/app.py',
            'examples/local_notification_server/app.py',
            'examples/YOLO_server_api/app.py',
            'examples/streaming_web/app.py',
            'examples/violation_records/app.py',
        )

        with patch('uvicorn.run') as run_server:
            for script in scripts:
                _run_as_script(script)

        self.assertEqual(run_server.call_count, len(scripts))

    def test_async_entrypoints_delegate_to_asyncio_runner(self) -> None:
        """Async scripts should schedule their main coroutine when run
        directly."""
        scripts = (
            'src/notifiers/telegram_notifier.py',
            'src/notifiers/line_notifier_message_api.py',
            'src/stream_capture.py',
            'src/yolo_detector.py',
        )

        def close_coroutine(coroutine: object) -> None:
            close = getattr(coroutine, 'close', None)
            if not callable(close):
                self.fail('asyncio.run received a non-coroutine')
                return
            close()

        with patch('asyncio.run', side_effect=close_coroutine) as run_async:
            for script in scripts:
                _run_as_script(script)

        self.assertEqual(run_async.call_count, len(scripts))

    def test_cli_entrypoints_support_help_without_starting_workloads(
        self,
    ) -> None:
        """Direct CLI execution should expose help before any workload
        starts."""
        scripts = (
            'src/stream_viewer.py',
            'examples/YOLO_train/export_int8_engine.py',
            'examples/YOLO_train/train.py',
            'examples/YOLO_data_augmentation/'
            'data_augmentation_albumentations.py',
            'examples/YOLO_data_augmentation/visualise_bounding_boxes.py',
            'examples/YOLO_evaluation/convert_yolo_to_coco.py',
            'examples/YOLO_evaluation/evaluate_sahi_yolo.py',
            'examples/YOLO_evaluation/evaluate_yolo.py',
        )

        for script in scripts:
            with self.subTest(script=script):
                output = StringIO()
                with (
                    patch.object(sys, 'argv', [script, '--help']),
                    redirect_stdout(output),
                    redirect_stderr(output),
                    self.assertRaises(SystemExit) as exit_context,
                ):
                    _run_as_script(script)

                self.assertEqual(exit_context.exception.code, 0)
                self.assertIn('usage:', output.getvalue())

    def test_sample_notification_entrypoints_use_mocked_http(self) -> None:
        """Example notification scripts must not make real HTTP requests."""
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            'access_token': 'token',
            'media_id': 'id',
        }

        with (
            patch('requests.get', return_value=response) as get_request,
            patch('requests.post', return_value=response) as post_request,
            patch('logging.basicConfig') as configure_logging,
            redirect_stdout(StringIO()),
        ):
            _run_as_script('src/notifiers/wechat_notifier.py')
            _run_as_script('src/notifiers/messenger_notifier.py')
            _run_as_script('src/notifiers/broadcast_notifier.py')

        get_request.assert_called_once()
        self.assertGreaterEqual(post_request.call_count, 3)
        configure_logging.assert_called()

    def test_model_fetcher_entrypoint_registers_then_runs_scheduler(
        self,
    ) -> None:
        """The scheduler entrypoint registers work before entering its loop."""
        schedule_job = MagicMock()

        with (
            patch('schedule.every', return_value=schedule_job) as every,
            patch('schedule.run_pending', side_effect=KeyboardInterrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            _run_as_script('src/model_fetcher.py')

        every.assert_called_once_with(1)
        schedule_job.hour.do.assert_called_once()

    def test_remaining_sample_entrypoints_do_not_touch_external_resources(
        self,
    ) -> None:
        """Local demonstrations can start directly with only local doubles."""
        with (
            patch(
                'logging.handlers.RotatingFileHandler',
                return_value=logging.NullHandler(),
            ),
            patch('pathlib.Path.mkdir'),
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()),
        ):
            _run_as_script('src/danger_detector.py')
            _run_as_script('src/monitor_logger.py')
            _run_as_script('examples/local_notification_server/lang_config.py')


if __name__ == '__main__':
    unittest.main()
