import pytest
from unittest.mock import patch, MagicMock
from src.live_stream_detection import LiveStreamDetector

# Mock the requests.post method to simulate API responses
@patch('src.live_stream_detection.requests.post')
def test_live_stream_detector_initialisation(mock_post):
    """
    Test the initialisation of the LiveStreamDetector class.
    """
    # Set up the mock response for the authentication
    mock_response = MagicMock()
    mock_response.json.return_value = {'access_token': 'fake_token'}
    mock_post.return_value = mock_response

    # Initialise the detector
    detector = LiveStreamDetector(api_url='http://mock_api', model_key='yolov8l', output_folder='test_output', output_filename='test_frame.jpg')

    # Check if the initialisation is correct
    assert detector.api_url == 'http://mock_api'
    assert detector.model_key == 'yolov8l'
    assert detector.output_folder == 'test_output'
    assert detector.output_filename == 'test_frame.jpg'
    assert detector.access_token == 'fake_token'

@patch('src.live_stream_detection.requests.post')
def test_live_stream_detector_authentication(mock_post):
    """
    Test the authentication method of the LiveStreamDetector class.
    """
    # Set up the mock response for the authentication
    mock_response = MagicMock()
    mock_response.json.return_value = {'access_token': 'new_fake_token'}
    mock_post.return_value = mock_response

    # Initialise the detector
    detector = LiveStreamDetector(api_url='http://mock_api', model_key='yolov8l')

    # Call the authenticate method
    detector.authenticate()

    # Check if the access token is updated
    assert detector.access_token == 'new_fake_token'

@patch('src.live_stream_detection.cv2.imwrite')
@patch('src.live_stream_detection.LiveStreamDetector.generate_detections')
def test_live_stream_detector_draw_detections_on_frame(mock_generate_detections, mock_imwrite):
    """
    Test the draw_detections_on_frame method of the LiveStreamDetector class.
    """
    # Set up the mock response for the generate_detections method
    mock_generate_detections.return_value = ([], MagicMock())

    # Initialise the detector
    detector = LiveStreamDetector(api_url='http://mock_api', model_key='yolov8l')

    # Create a fake frame
    fake_frame = MagicMock()

    # Call the draw_detections_on_frame method
    detector.draw_detections_on_frame(fake_frame, [])

    # Check if the frame was saved
    mock_imwrite.assert_called_once()

# Run the tests
if __name__ == '__main__':
    pytest.main()
