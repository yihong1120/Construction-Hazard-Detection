from __future__ import annotations

import base64
import threading
from unittest.mock import MagicMock
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from examples.mcp_server.tools.streaming import StreamingTools


@pytest.mark.anyio
@pytest.mark.parametrize('frame_format', ['base64', 'array'])
async def test_capture_and_release_off_event_loop(frame_format):
    cap = MagicMock()
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    main_thread = threading.get_ident()

    def read():
        assert threading.get_ident() != main_thread
        return True, frame

    cap.read.side_effect = read
    with patch('cv2.VideoCapture', return_value=cap) as factory:
        result = await StreamingTools().capture_frame(
            'rtsp://camera', frame_format,
        )
    assert result['success']
    cap.release.assert_called_once()
    assert cv2.CAP_PROP_READ_TIMEOUT_MSEC in factory.call_args.args[2]
    if frame_format == 'array':
        assert result['frame_data'] == frame.tolist()
    else:
        decoded = cv2.imdecode(
            np.frombuffer(
                base64.b64decode(
                    result['frame_data'],
                ), np.uint8,
            ), cv2.IMREAD_COLOR,
        )
        assert decoded.shape == frame.shape


@pytest.mark.anyio
@pytest.mark.parametrize('result', [(False, None), (True, None)])
async def test_missing_frame_returns_failure_and_releases(result):
    cap = MagicMock()
    cap.read.return_value = result
    with patch('cv2.VideoCapture', return_value=cap):
        assert not (await StreamingTools().capture_frame('camera'))['success']
    cap.release.assert_called_once()


@pytest.mark.anyio
async def test_read_failure_and_encoder_failure_release_capture():
    cap = MagicMock()
    cap.read.side_effect = RuntimeError('read')
    with patch('cv2.VideoCapture', return_value=cap):
        with pytest.raises(RuntimeError):
            await StreamingTools().capture_frame('camera')
    cap.release.assert_called_once()
    cap.read.side_effect = None
    cap.read.return_value = (True, np.zeros((2, 2, 3), np.uint8))
    with (
        patch('cv2.VideoCapture', return_value=cap),
        patch('cv2.imencode', return_value=(False, None)),
    ):
        assert not (await StreamingTools().capture_frame('camera'))['success']


@pytest.mark.anyio
@pytest.mark.parametrize('url,fmt', [('', 'base64'), ('camera', 'bytes')])
async def test_invalid_inputs(url, fmt):
    with pytest.raises(ValueError):
        await StreamingTools().capture_frame(url, fmt)
