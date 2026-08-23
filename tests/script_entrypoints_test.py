from __future__ import annotations

import logging
import runpy
import sys
import unittest
from collections.abc import Callable
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import cast
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_as_script(relative_path: str) -> dict[str, object]:
    """Execute a project script exactly as ``python path/to/script.py``
    would."""
    return runpy.run_path(
        str(PROJECT_ROOT / relative_path),
        run_name='__main__',
    )


def _load_migration_runner() -> dict[str, object]:
    """Load migration-runner helpers without invoking its command line.

    Returns:
        The migration runner's module namespace.
    """
    return runpy.run_path(
        str(PROJECT_ROOT / 'scripts/apply_postgres_migrations.py'),
        run_name='postgres_migration_runner_test',
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
            'src/local_yolo_detector.py',
        )

        def close_coroutine(coroutine: object) -> None:
            """Perform close coroutine.

            Args:
                coroutine: Value used by this callable.
            """
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

    def test_migration_sql_splitter_preserves_postgresql_dollar_blocks(
        self,
    ) -> None:
        """Concurrent-index SQL must not share an asyncpg query batch."""
        namespace = _load_migration_runner()
        splitter = cast(
            Callable[[str], list[str]],
            namespace['split_sql_statements'],
        )
        execution_units = cast(
            Callable[[str], list[str]],
            namespace['statements_for_execution'],
        )
        source = '''
        DO $body$
        BEGIN
            PERFORM 1;
        END
        $body$;
        CREATE INDEX CONCURRENTLY example_index ON example_table (id);
        INSERT INTO schema_migrations (version, checksum)
        VALUES ('example', 'checksum');
        '''

        statements = splitter(source)

        self.assertEqual(len(statements), 3)
        self.assertIn('PERFORM 1;', statements[0])
        self.assertIn('CREATE INDEX CONCURRENTLY', statements[1])
        self.assertIn('INSERT INTO schema_migrations', statements[2])
        self.assertEqual(execution_units(source), statements)
        self.assertEqual(
            execution_units('SELECT 1; SELECT 2;'),
            ['SELECT 1; SELECT 2;'],
        )


if __name__ == '__main__':
    unittest.main()
