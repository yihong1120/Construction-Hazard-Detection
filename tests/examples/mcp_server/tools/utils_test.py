from __future__ import annotations

import base64
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from examples.mcp_server.tools.utils import UtilsTools


class TestCalculatePolygonArea(unittest.IsolatedAsyncioTestCase):
    async def test_triangle_area(self):
        tool = UtilsTools()
        points = [[0, 0], [4, 0], [0, 3]]
        res = await tool.calculate_polygon_area(points)
        self.assertAlmostEqual(res['area'], 6.0)

    async def test_less_than_three_points(self):
        tool = UtilsTools()
        res = await tool.calculate_polygon_area([[1, 1], [2, 2]])
        self.assertEqual(res['area'], 0.0)

    async def test_area_exception(self):
        with patch(
            'examples.mcp_server.tools.utils.logging.getLogger',
        ) as mock_logger:
            logger = mock_logger.return_value
            tool = UtilsTools()
            tool.logger = logger
            with patch(
                'examples.mcp_server.tools.utils.abs',
                side_effect=RuntimeError('boom'),
            ):
                with self.assertRaises(RuntimeError):
                    await tool.calculate_polygon_area(
                        [[0, 0], [1, 0], [0, 1]],
                    )
                logger.error.assert_called_once()


class TestPointInPolygon(unittest.IsolatedAsyncioTestCase):
    async def test_inside(self):
        tool = UtilsTools()
        poly = [[0, 0], [10, 0], [10, 10], [0, 10]]
        res = await tool.point_in_polygon([5, 5], poly)
        self.assertTrue(res['is_inside'])

    async def test_outside(self):
        tool = UtilsTools()
        poly = [[0, 0], [10, 0], [10, 10], [0, 10]]
        res = await tool.point_in_polygon([15, 5], poly)
        self.assertFalse(res['is_inside'])

    async def test_point_exception(self):
        with patch(
            'examples.mcp_server.tools.utils.logging.getLogger',
        ) as mock_logger:
            logger = mock_logger.return_value
            tool = UtilsTools()
            tool.logger = logger
            with patch(
                'examples.mcp_server.tools.utils.range',
                side_effect=RuntimeError('boom'),
            ):
                with self.assertRaises(RuntimeError):
                    await tool.point_in_polygon(
                        [1, 1],
                        [[0, 0], [1, 0], [0, 1]],
                    )
                logger.error.assert_called_once()


class TestBBoxIntersection(unittest.IsolatedAsyncioTestCase):
    async def test_overlap(self):
        tool = UtilsTools()
        res = await tool.bbox_intersection([0, 0, 2, 2], [1, 1, 3, 3])
        self.assertTrue(res['intersection_area'] > 0)

    async def test_no_overlap(self):
        tool = UtilsTools()
        res = await tool.bbox_intersection([0, 0, 1, 1], [2, 2, 3, 3])
        self.assertEqual(res['intersection_area'], 0.0)

    async def test_bbox_exception(self):
        with patch(
            'examples.mcp_server.tools.utils.logging.getLogger',
        ) as mock_logger:
            logger = mock_logger.return_value
            tool = UtilsTools()
            tool.logger = logger
            with patch(
                'examples.mcp_server.tools.utils.max',
                side_effect=RuntimeError('boom'),
            ):
                with self.assertRaises(RuntimeError):
                    await tool.bbox_intersection(
                        [0, 0, 1, 1],
                        [0, 0, 1, 1],
                    )
                logger.error.assert_called_once()


class TestDistanceBetweenPoints(unittest.IsolatedAsyncioTestCase):
    async def test_all_metrics(self):
        tool = UtilsTools()
        p1, p2 = [0, 0], [3, 4]
        self.assertAlmostEqual(
            (await tool.distance_between_points(p1, p2, 'euclidean'))[
                'distance'
            ],
            5.0,
        )
        self.assertEqual(
            (await tool.distance_between_points(p1, p2, 'manhattan'))[
                'distance'
            ],
            7,
        )
        self.assertEqual(
            (await tool.distance_between_points(p1, p2, 'chebyshev'))[
                'distance'
            ],
            4,
        )

    async def test_invalid_metric(self):
        tool = UtilsTools()
        with self.assertRaises(ValueError):
            await tool.distance_between_points([0, 0], [1, 1], 'invalid')

    async def test_distance_exception(self):
        with patch(
            'examples.mcp_server.tools.utils.logging.getLogger',
        ) as mock_logger:
            logger = mock_logger.return_value
            tool = UtilsTools()
            tool.logger = logger
            with patch(
                'examples.mcp_server.tools.utils.sqrt',
                side_effect=RuntimeError('boom'),
            ):
                with self.assertRaises(RuntimeError):
                    await tool.distance_between_points([0, 0], [1, 1])
                logger.error.assert_called_once()


class TestCreateSafetyZone(unittest.IsolatedAsyncioTestCase):
    async def test_circle_square(self):
        tool = UtilsTools()
        self.assertEqual(
            len(
                (await tool.create_safety_zone([0, 0], 1, 'circle'))[
                    'zone_points'
                ],
            ),
            32,
        )
        self.assertEqual(
            len(
                (await tool.create_safety_zone([0, 0], 1, 'square'))[
                    'zone_points'
                ],
            ),
            4,
        )

    async def test_invalid_shape(self):
        tool = UtilsTools()
        with self.assertRaises(ValueError):
            await tool.create_safety_zone([0, 0], 1, 'triangle')

    async def test_zone_exception(self):
        # __import__ 是 builtins，而不是模組屬性
        with (
            patch('builtins.__import__', side_effect=RuntimeError('boom')),
            patch(
                'examples.mcp_server.tools.utils.logging.getLogger',
            ) as mock_logger,
        ):
            logger = mock_logger.return_value
            tool = UtilsTools()
            tool.logger = logger
            with self.assertRaises(RuntimeError):
                await tool.create_safety_zone([0, 0], 1, 'circle')
            logger.error.assert_called_once()


class TestNormalizeCoordinates(unittest.IsolatedAsyncioTestCase):
    async def test_all_formats(self):
        tool = UtilsTools()
        self.assertTrue(
            (
                await tool.normalize_coordinates(
                    [[50, 50]], 100, 100, 'normalized',
                )
            )['success'],
        )
        self.assertTrue(
            (
                await tool.normalize_coordinates(
                    [[0, 0, 100, 200]], 200, 400, 'yolo',
                )
            )['success'],
        )
        self.assertTrue(
            (
                await tool.normalize_coordinates(
                    [[0, 0, 100, 200]], 200, 400, 'coco',
                )
            )['success'],
        )

    async def test_invalid_cases(self):
        tool = UtilsTools()
        with self.assertRaises(ValueError):
            await tool.normalize_coordinates(
                [[1, 2, 3]], 100, 100, 'yolo',
            )
        with self.assertRaises(ValueError):
            await tool.normalize_coordinates(
                [[1, 2, 3]], 100, 100, 'coco',
            )
        with self.assertRaises(ValueError):
            await tool.normalize_coordinates(
                [[0, 0]], 100, 100, 'invalid',
            )

    async def test_normalize_exception(self):
        with patch(
            'examples.mcp_server.tools.utils.logging.getLogger',
        ) as mock_logger:
            logger = mock_logger.return_value
            tool = UtilsTools()
            tool.logger = logger
            # 傳入合法 bbox，讓內部 _clip 調用 max()，再把 max 打爆
            with patch(
                'examples.mcp_server.tools.utils.max',
                side_effect=RuntimeError('boom'),
            ):
                with self.assertRaises(RuntimeError):
                    await tool.normalize_coordinates(
                        [[0, 0, 1, 1]], 1, 1,
                    )
                logger.error.assert_called_once()


class TestConvertImageFormat(unittest.IsolatedAsyncioTestCase):
    async def test_successful_conversion(self):
        fake_img_data = base64.b64encode(b'fakeimg').decode()
        fake_image = MagicMock()
        fake_image.convert.return_value = fake_image

        # 模擬 PIL 寫入資料到 BytesIO
        def _fake_save(buf, format=None, **kw):
            buf.write(b'newdata')

        fake_image.save.side_effect = _fake_save

        with (
            patch('PIL.Image.open', return_value=fake_image),
            patch('io.BytesIO', MagicMock()),
            patch('base64.b64decode', return_value=b'fakeimg'),
            patch('base64.b64encode', return_value=b'encoded'),
        ):
            tool = UtilsTools()
            res = await tool.convert_image_format(fake_img_data)

        self.assertTrue(res['success'])
        self.assertEqual(res['converted_image'], 'encoded')

    async def test_pil_fallback(self):
        fake_img_data = 'abc123'
        # 讓 PIL 失敗 → 走 fallback，並驗證 logger.warning
        with (
            patch('PIL.Image.open', side_effect=RuntimeError('fail')),
            patch(
                'examples.mcp_server.tools.utils.logging.getLogger',
            ) as mock_logger,
        ):
            logger = mock_logger.return_value
            tool = UtilsTools()
            tool.logger = logger
            res = await tool.convert_image_format(fake_img_data)
        self.assertTrue(res['success'])
        logger.warning.assert_called_once()

    async def test_convert_exception(self):
        with patch(
            'examples.mcp_server.tools.utils.logging.getLogger',
        ) as mock_logger:
            logger = mock_logger.return_value
            tool = UtilsTools()
            tool.logger = logger
            # 在最外層先把 len() 打爆，直接走 except
            with patch(
                'examples.mcp_server.tools.utils.len',
                side_effect=RuntimeError('boom'),
            ):
                with self.assertRaises(RuntimeError):
                    await tool.convert_image_format('a')
                logger.error.assert_called_once()


class TestValidateDetectionData(unittest.IsolatedAsyncioTestCase):
    async def test_valid_detection(self):
        tool = UtilsTools()
        res = await tool.validate_detection_data(
            [{'bbox': [0, 0, 10, 10]}],
            100,
            100,
        )
        self.assertTrue(res['is_valid'])

    async def test_invalid_detections(self):
        tool = UtilsTools()
        dets = [
            {'bbox': [10, 10, 5, 5]},
            {'box': [0, 0, 10, 10], 'confidence': 'bad'},
            {'bbox': [0, 0, 10, 10], 'class': 'wrong'},
            'not_dict',
        ]
        res = await tool.validate_detection_data(dets, 50, 50)
        self.assertFalse(res['is_valid'])
        self.assertGreater(len(res['validation_errors']), 1)

    async def test_validate_exception(self):
        # 避免打掉 isinstance（mock 自己會用），改在 float() 轉換處觸發錯誤
        with (
            patch(
                'examples.mcp_server.tools.utils.float',
                side_effect=RuntimeError('boom'),
            ),
            patch(
                'examples.mcp_server.tools.utils.logging.getLogger',
            ) as mock_logger,
        ):
            logger = mock_logger.return_value
            tool = UtilsTools()
            tool.logger = logger
            with self.assertRaises(RuntimeError):
                await tool.validate_detection_data(
                    [{'bbox': [0, 0, 1, 1]}],
                    10,
                    10,
                )
            logger.error.assert_called_once()

    async def test_more_invalid_branches_and_normalized(self):
        tool = UtilsTools()
        dets = [
            {},  # missing 'bbox'/'box'
            {'bbox': [0, 'a', 1, 1]},  # non-numeric in bbox
            {'bbox': [1, 2, 3]},  # wrong length
            {'bbox': [0.1, 0.2, 0.3, 0.4]},  # normalized coords branch (valid)
            {'bbox': [-1, -1, 60, 60]},  # out-of-range on all sides
            {'box': [0, 0, 10, 10], 'conf': 'bad'},  # conf wrong type
            {'box': [0, 0, 10, 10], 'cls': 'bad'},  # cls wrong type
        ]
        res = await tool.validate_detection_data(dets, 50, 50)
        self.assertFalse(res['is_valid'])  # because some entries are invalid
        joined = '\n'.join(
            res['validation_errors'],
        )  # make assertions easier
        self.assertIn("missing 'bbox'/'box'", joined)
        self.assertIn("'bbox' values must be numbers", joined)
        self.assertIn("'bbox' must be a list of 4 numbers", joined)
        self.assertIn('x1 out of range [0,50]', joined)
        self.assertIn('x2 out of range [0,50]', joined)
        self.assertIn('y1 out of range [0,50]', joined)
        self.assertIn('y2 out of range [0,50]', joined)
        self.assertIn("'conf' must be a number", joined)
        self.assertIn("'cls' must be an integer", joined)

    async def test_valid_normalized_only(self):
        tool = UtilsTools()
        dets = [{'bbox': [0.2, 0.2, 0.8, 0.9]}]
        res = await tool.validate_detection_data(dets, 1920, 1080)
        self.assertTrue(res['is_valid'])  # normalized and valid geometry


class TestEnsureUtils(unittest.IsolatedAsyncioTestCase):
    async def test_ensure_once(self):
        with patch('examples.mcp_server.tools.utils.Utils') as mock_utils:
            tool = UtilsTools()
            await tool._ensure_utils()
            await tool._ensure_utils()
            mock_utils.assert_called_once()


if __name__ == '__main__':
    unittest.main()

'''
pytest --cov=examples.mcp_server.tools.utils\
    --cov-report=term-missing\
    tests/examples/mcp_server/tools/utils_test.py
'''
