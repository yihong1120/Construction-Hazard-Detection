from __future__ import annotations

from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

from src import stream_detection
from src import stream_processor as processor


class ClusterPipelineTest(IsolatedAsyncioTestCase):
    async def test_one_analysis_result_feeds_live_overlay_and_persistence(
        self,
    ):
        cones = [[[10, 10], [100, 10], [100, 100], [10, 10]]]
        poles = [[[200, 10], [300, 10], [300, 100], [200, 10]]]
        warnings = {'warning_people_in_controlled_area': {'count': 1}}
        detector = MagicMock()
        detector.detect_danger.return_value = warnings, cones, poles
        state = processor._LatestDetectionState()
        with (
            patch.object(stream_detection, 'should_notify', return_value=True),
            patch.object(
                stream_detection,
                '_store_media_server_viewer_data',
                AsyncMock(),
            ),
            patch.object(
                stream_detection,
                '_send_violation_and_notification',
                AsyncMock(return_value=123),
            ) as send,
        ):
            await stream_detection._record_detection_result(
                frame=np.zeros((360, 640, 3), dtype=np.uint8),
                timestamp=123,
                sequence=1,
                track_data=[],
                danger_detector=detector,
                fcm_sender=AsyncMock(),
                violation_sender=AsyncMock(),
                redis_manager=MagicMock(),
                latest_detection=state,
                site='Site',
                stream_name='Cam',
                work_start_hour=0,
                work_end_hour=24,
                metadata_key='key',
                last_notification_time=0,
                last_warning_event_time=None,
            )
        detector.detect_danger.assert_called_once_with([])
        self.assertIs(state.cone_polys, cones)
        self.assertIs(state.pole_polys, poles)
        self.assertIs(send.await_args.kwargs['cone_polys'], cones)
        self.assertIs(send.await_args.kwargs['pole_polys'], poles)
        self.assertIs(send.await_args.kwargs['warnings'], warnings)
