from __future__ import annotations

import unittest

from examples.mcp_server.tools.utils import bbox_intersection
from examples.mcp_server.tools.utils import calculate_polygon_area
from examples.mcp_server.tools.utils import point_in_polygon
from examples.mcp_server.tools.utils import validate_detection_data


class UtilityFunctionTests(unittest.TestCase):
    """The MCP utilities are synchronous pure functions, not service wrappers."""

    def test_polygon_area(self) -> None:
        """Test polygon area.
        """
        result = calculate_polygon_area([[0, 0], [4, 0], [0, 3]])
        self.assertEqual(result['area'], 6.0)

    def test_point_and_bbox_operations(self) -> None:
        """Test point and bbox operations.
        """
        self.assertTrue(
            point_in_polygon([5, 5], [[0, 0], [10, 0], [10, 10], [0, 10]])[
                'is_inside'
            ],
        )
        self.assertEqual(
            bbox_intersection([0, 0, 2, 2], [1, 1, 3, 3])[
                'intersection_area'
            ],
            1.0,
        )

    def test_detection_validation(self) -> None:
        """Test detection validation.
        """
        valid = validate_detection_data(
            [{'bbox': [0.1, 0.2, 0.5, 0.8], 'confidence': 0.9}], 100, 100,
        )
        invalid = validate_detection_data([{'bbox': [2, 0, 1, 0]}], 100, 100)
        self.assertTrue(valid['is_valid'])
        self.assertFalse(invalid['is_valid'])
