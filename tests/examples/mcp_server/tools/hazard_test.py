from __future__ import annotations

import unittest
from typing import cast
from unittest.mock import patch

from examples.mcp_server.tools.hazard import HazardTools


class DetectViolationsListInputTests(unittest.IsolatedAsyncioTestCase):
    """Tests for detect_violations with list-based detections."""

    async def test_detect_violations_with_list_passthrough(self) -> None:
        """List inputs should be passed through unmodified to the detector."""
        detections: list[list[float]] = [
            [0.0, 1.0, 2.0, 3.0, 0.9, 1.0],
            [10.0, 11.0, 12.0, 13.0, 0.8, 2.0],
        ]
        fake_warnings: dict[str, dict[str, int]] = {'zone_a': {'no_helmet': 2}}
        fake_cones: list[list[list[float]]] = [[[0.0, 0.0], [1.0, 0.0]]]
        fake_poles: list[list[list[float]]] = [[[2.0, 2.0], [3.0, 3.0]]]

        with patch(
            'examples.mcp_server.tools.hazard.DangerDetector',
        ) as mock_dd:
            inst = mock_dd.return_value
            inst.detect_danger.return_value = (
                fake_warnings,
                fake_cones,
                fake_poles,
            )

            tool = HazardTools()
            result = await tool.detect_violations(
                detections=detections,
            )

            # Detector should be initialised once with default detection items.
            mock_dd.assert_called_once()
            inst.detect_danger.assert_called_once_with(detections)

            # Response shape
            self.assertEqual(result['warnings'], fake_warnings)
            self.assertEqual(result['cone_polygons'], fake_cones)
            self.assertEqual(result['pole_polygons'], fake_poles)
            self.assertNotIn('meta', result)


class DetectViolationsDictInputTests(unittest.IsolatedAsyncioTestCase):
    """Tests for detect_violations with dict-based detections that require
    normalisation."""

    async def test_detect_violations_normalises_bbox_conf_class_(self) -> None:
        """Normalise keys: bbox + confidence + class_."""
        detections = [
            {
                'bbox': [1.0, 2.0, 3.0, 4.0],
                'confidence': 0.95,
                'class_': 5,
            },
        ]
        expected: list[list[float]] = [[1.0, 2.0, 3.0, 4.0, 0.95, 5.0]]

        with patch(
            'examples.mcp_server.tools.hazard.DangerDetector',
        ) as mock_dd:
            inst = mock_dd.return_value
            inst.detect_danger.return_value = ({}, [], [])

            tool = HazardTools()
            res = await tool.detect_violations(
                detections=cast(list, detections),
            )

            inst.detect_danger.assert_called_once_with(expected)
            self.assertNotIn('meta', res)

    async def test_detect_violations_normalises_dict_input(self) -> None:
        """Normalise explicit detection dict keys."""
        detections = [
            {
                'bbox': [10, 20, 30, 40],
                'confidence': 0.88,
                'class_': 7,
            },
        ]
        expected: list[list[float]] = [[10.0, 20.0, 30.0, 40.0, 0.88, 7.0]]

        with patch(
            'examples.mcp_server.tools.hazard.DangerDetector',
        ) as mock_dd:
            inst = mock_dd.return_value
            inst.detect_danger.return_value = ({}, [], [])
            tool = HazardTools()
            await tool.detect_violations(detections=cast(list, detections))
            inst.detect_danger.assert_called_once_with(expected)

    async def test_detect_violations_skips_invalid_bbox(self) -> None:
        """Entries with invalid bbox should be skipped silently."""
        detections = [
            {'bbox': [1, 2, 3]},  # fewer than 4 coords -> invalid
            {'bbox': 'not-a-list'},  # wrong type
            {'box': [0, 0, 0, 0], 'confidence': 0.9, 'class_': 1},
        ]
        expected: list[list[float]] = []

        with patch(
            'examples.mcp_server.tools.hazard.DangerDetector',
        ) as mock_dd:
            inst = mock_dd.return_value
            inst.detect_danger.return_value = ({}, [], [])
            tool = HazardTools()
            await tool.detect_violations(detections=cast(list, detections))
            inst.detect_danger.assert_called_once_with(expected)

    async def test_detect_violations_non_numeric_values(self) -> None:
        """Non-numeric confidence/class_ types should normalise to zeros."""
        detections = [
            {
                'bbox': [0, 0, 1, 1],
                'confidence': {'x': 1},
                'class_': {'y': 2},
            },
        ]
        expected: list[list[float]] = [[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]]

        with patch(
            'examples.mcp_server.tools.hazard.DangerDetector',
        ) as mock_dd:
            inst = mock_dd.return_value
            inst.detect_danger.return_value = ({}, [], [])

            tool = HazardTools()
            await tool.detect_violations(detections=cast(list, detections))

            inst.detect_danger.assert_called_once_with(expected)

    async def test_detect_violations_invalid_numeric_strings_fall_back_to_zero(
        self,
    ) -> None:
        """Unparseable string confidence and class values are harmless."""
        detections = [
            {'bbox': [0, 0, 1, 1]},
            {
                'bbox': [0, 0, 1, 1],
                'confidence': 'not-a-number',
                'class_': 'not-a-class',
            },
        ]
        with patch(
            'examples.mcp_server.tools.hazard.DangerDetector',
        ) as mock_dd:
            detector = mock_dd.return_value
            detector.detect_danger.return_value = ({}, [], [])
            tool = HazardTools()
            await tool.detect_violations(detections=cast(list, detections))

        detector.detect_danger.assert_called_once_with(
            [[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]],
        )

    async def test_detect_violations_propagates_detector_exception(
        self,
    ) -> None:
        """If underlying detector fails, the exception should propagate."""
        with patch(
            'examples.mcp_server.tools.hazard.DangerDetector',
        ) as mock_dd:
            inst = mock_dd.return_value
            inst.detect_danger.side_effect = RuntimeError('boom')
            tool = HazardTools()
            with self.assertRaises(RuntimeError):
                await tool.detect_violations(
                    detections=[[0.0, 0.0, 1.0, 1.0, 0.9, 1.0]],
                )


class InitDetectorTests(unittest.IsolatedAsyncioTestCase):
    """Tests for the private initialiser _init_detector."""

    async def test_init_detector_default_items(self) -> None:
        """When no detection_items provided, defaults should be used."""
        with patch(
            'examples.mcp_server.tools.hazard.DangerDetector',
        ) as mock_dd:
            tool = HazardTools()
            await tool._init_detector(None)
            # Called exactly once with a dict that contains expected defaults.
            self.assertTrue(mock_dd.called)
            (args, _kwargs) = mock_dd.call_args
            self.assertIsInstance(args[0], dict)
            defaults = args[0]
            # Check a representative subset of default keys set to True.
            for key in (
                'detect_no_safety_vest_or_helmet',
                'detect_near_machinery_or_vehicle',
                'detect_in_restricted_area',
                'detect_in_utility_pole_restricted_area',
                'detect_machinery_close_to_pole',
            ):
                self.assertIn(key, defaults)
                self.assertTrue(defaults[key])

    async def test_init_detector_uses_user_items(self) -> None:
        """Detector should be created with user-provided detection_items."""
        custom = {'detect_in_restricted_area': False, 'custom_flag': True}
        with patch(
            'examples.mcp_server.tools.hazard.DangerDetector',
        ) as mock_dd:
            tool = HazardTools()
            await tool._init_detector(custom)
            mock_dd.assert_called_once_with(custom)


if __name__ == '__main__':
    unittest.main()
