from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from src.stream_capture import StreamCapture

@pytest.fixture
def stream_capture():
    return StreamCapture("http://example.com/stream")

def test_initialise_stream_success(stream_capture):
    with patch('cv2.VideoCapture') as mock_video_capture:
        mock_video_capture.return_value.isOpened.return_value = True
        stream_capture.initialise_stream(stream_capture.stream_url)
        assert stream_capture.cap is not None
        mock_video_capture.assert_called_once_with(stream_capture.stream_url)
        stream_capture.release_resources()

def test_initialise_stream_retry(stream_capture):
    with patch('cv2.VideoCapture') as mock_video_capture:
        instance = mock_video_capture.return_value
        instance.isOpened.side_effect = [False, True]  # Fail the first time, succeed the second time
        with patch('time.sleep') as mock_sleep:
            stream_capture.initialise_stream(stream_capture.stream_url)
            assert stream_capture.cap is not None
            mock_sleep.assert_called_once_with(5)
            assert mock_video_capture.call_count == 1  # Only one instance creation
            assert instance.open.call_count == 1  # One retry to open the stream

def test_release_resources(stream_capture):
    with patch.object(stream_capture, 'cap', create=True) as mock_cap:
        stream_capture.release_resources()
        mock_cap.release.assert_called_once()
        assert stream_capture.cap is None

@patch('cv2.VideoCapture')
@patch('cv2.Mat')
def test_execute_capture(mock_mat, mock_video_capture, stream_capture):
    mock_video_capture.return_value.read.return_value = (True, mock_mat)
    mock_video_capture.return_value.isOpened.return_value = True
    with patch('time.sleep', return_value=None):
        generator = stream_capture.execute_capture()
        frame, timestamp = next(generator)
        assert frame is not None
        assert isinstance(timestamp, float)
    stream_capture.release_resources()

def test_check_internet_speed(stream_capture):
    with patch('speedtest.Speedtest') as mock_speedtest:
        mock_speedtest.return_value.download.return_value = 50_000_000
        mock_speedtest.return_value.upload.return_value = 10_000_000
        download_speed, upload_speed = stream_capture.check_internet_speed()
        assert download_speed == 50.0
        assert upload_speed == 10.0

@patch('streamlink.streams')
def test_select_quality_based_on_speed_high_speed(mock_streams, stream_capture):
    mock_streams.return_value = {
        'best': MagicMock(url='http://best.stream'),
        '1080p': MagicMock(url='http://1080p.stream'),
        '720p': MagicMock(url='http://720p.stream'),
        '480p': MagicMock(url='http://480p.stream')
    }
    with patch.object(stream_capture, 'check_internet_speed', return_value=(20, 5)):
        selected_quality = stream_capture.select_quality_based_on_speed()
        assert selected_quality == 'http://best.stream'

@patch('streamlink.streams')
def test_select_quality_based_on_speed_medium_speed(mock_streams, stream_capture):
    mock_streams.return_value = {
        '720p': MagicMock(url='http://720p.stream'),
        '480p': MagicMock(url='http://480p.stream'),
        '360p': MagicMock(url='http://360p.stream')
    }
    with patch.object(stream_capture, 'check_internet_speed', return_value=(7, 5)):
        selected_quality = stream_capture.select_quality_based_on_speed()
        assert selected_quality == 'http://720p.stream'

@patch('streamlink.streams')
def test_select_quality_based_on_speed_low_speed(mock_streams, stream_capture):
    mock_streams.return_value = {
        '480p': MagicMock(url='http://480p.stream'),
        '360p': MagicMock(url='http://360p.stream'),
        '240p': MagicMock(url='http://240p.stream')
    }
    with patch.object(stream_capture, 'check_internet_speed', return_value=(3, 5)):
        selected_quality = stream_capture.select_quality_based_on_speed()
        assert selected_quality == 'http://480p.stream'

@patch('streamlink.streams', return_value={})
@patch.object(StreamCapture, 'check_internet_speed', return_value=(20, 5))
def test_select_quality_based_on_speed_no_quality(mock_check_speed, mock_streams, stream_capture):
    selected_quality = stream_capture.select_quality_based_on_speed()
    assert selected_quality is None

@patch('streamlink.streams', return_value={
    'best': MagicMock(url='http://best.stream'),
    '720p': MagicMock(url='http://720p.stream'),
    '480p': MagicMock(url='http://480p.stream')
})
@patch('cv2.VideoCapture')
@patch('cv2.Mat')
def test_capture_generic_frames(mock_mat, mock_video_capture, mock_streams, stream_capture):
    mock_video_capture.return_value.read.return_value = (True, mock_mat)
    mock_video_capture.return_value.isOpened.return_value = True
    with patch('time.sleep', return_value=None):
        generator = stream_capture.capture_generic_frames()
        frame, timestamp = next(generator)
        assert frame is not None
        assert isinstance(timestamp, float)
    stream_capture.release_resources()

def test_update_capture_interval(stream_capture):
    stream_capture.update_capture_interval(20)
    assert stream_capture.capture_interval == 20

if __name__ == "__main__":
    pytest.main()
