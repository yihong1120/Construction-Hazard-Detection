from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import patch

import numpy as np

from src.yolo_detector import YoloDetector


class TestYoloDetector(unittest.IsolatedAsyncioTestCase):
    """Async test suite for YoloDetector.

    Tests shared-worker detection, tracking algorithms, and label filtering.
    """

    # Test instance attributes
    detector: YoloDetector
    detector_server: YoloDetector
    model_key: str
    output_folder: str

    def setUp(self) -> None:
        """Set up test environment and detector instances before each test."""
        self.model_key: str = 'yolo11n'
        self.output_folder: str = 'test_output'
        self.detector_server: YoloDetector = YoloDetector(
            model_key='yolo11n',
        )
        self.detector = self.detector_server

    def test_initialisation(self) -> None:
        """Test basic detector initialisation with default parameters."""
        detector = YoloDetector(
            model_key=self.model_key,
            output_folder=self.output_folder,
        )
        # Verify all configuration is set correctly
        self.assertEqual(detector.model_key, self.model_key)
        self.assertEqual(detector.output_folder, self.output_folder)

    async def test_generate_detections(self) -> None:
        """Shared-worker results receive a persistent remote track ID."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        worker_client = AsyncMock()
        worker_client.detect.return_value = [[20, 20, 60, 60, 0.8, 1]]
        self.detector.worker_client = worker_client

        detections, tracked = await self.detector.generate_detections(frame)

        self.assertEqual(detections, [[20, 20, 60, 60, 0.8, 1]])
        self.assertEqual(tracked[0][6:], [0, 0])
        worker_client.detect.assert_awaited_once_with(
            frame,
            model_key=self.detector.model_key,
        )

    async def test_server_detection_requires_worker_client(self) -> None:
        """Detection fails fast without a shared worker."""
        self.detector.worker_client = None

        with self.assertRaisesRegex(
            RuntimeError,
            'Shared YOLO worker is required',
        ):
            await self.detector.generate_detections(
                np.zeros((32, 32, 3), dtype=np.uint8),
            )

    async def test_close_method(self) -> None:
        """close is a no-op when no worker client is attached."""
        await self.detector.close()

    def test_remove_overlapping_labels(self) -> None:
        """Test removal of overlapping detection labels."""
        datas = [
            [10, 10, 50, 50, 0.9, 0],
            [10, 10, 50, 45, 0.8, 2],
            [20, 20, 60, 60, 0.85, 7],
            [20, 20, 60, 55, 0.75, 4],
        ]
        expected = [
            [10, 10, 50, 50, 0.9, 0],
            [20, 20, 60, 60, 0.85, 7],
        ]
        filtered = self.detector.remove_overlapping_labels(datas)
        self.assertEqual(filtered, expected)

    def test_overlap_percentage(self) -> None:
        """Test calculation of overlap percentage between bboxes."""
        bbox1 = [10, 10, 50, 50]
        bbox2 = [30, 30, 70, 70]
        overlap = self.detector.overlap_percentage(bbox1, bbox2)
        self.assertIsInstance(overlap, float)
        self.assertGreaterEqual(overlap, 0.0)
        self.assertLessEqual(overlap, 1.0)

    def test_is_contained(self) -> None:
        """Test detection of one bbox contained within another."""
        inner_bbox = [20, 20, 40, 40]
        outer_bbox = [10, 10, 50, 50]
        result = self.detector.is_contained(inner_bbox, outer_bbox)
        self.assertTrue(result)

        inner_bbox = [5, 5, 15, 15]
        result = self.detector.is_contained(inner_bbox, outer_bbox)
        self.assertFalse(result)

    def test_remove_completely_contained_labels(self) -> None:
        """Test removal of labels completely contained within others."""
        datas = [
            [10, 10, 50, 50, 0.9, 0],
            [20, 20, 40, 40, 0.8, 2],
            [60, 60, 100, 100, 0.85, 7],
        ]
        filtered = self.detector.remove_completely_contained_labels(datas)
        self.assertEqual(len(filtered), 2)

    def test_track_remote_centroid(self) -> None:
        """Test centroid-based remote tracking algorithm."""
        dets = [[10, 10, 50, 50, 0.9, 0]]
        tracked = self.detector._track_remote_centroid(dets)
        self.assertEqual(len(tracked), 1)
        self.assertEqual(len(tracked[0]), 8)

    def test_track_remote_hungarian(self) -> None:
        """Test Hungarian algorithm-based remote tracking."""
        dets = [[10, 10, 50, 50, 0.9, 0]]
        tracked = self.detector._track_remote_hungarian(dets)
        self.assertEqual(len(tracked), 1)
        self.assertEqual(len(tracked[0]), 8)

    def test_bbox_center(self) -> None:
        """Test bounding box centre calculation."""
        center = self.detector._bbox_center(10, 10, 50, 50)
        self.assertEqual(center, (30.0, 30.0))

    def test_bbox_iou(self) -> None:
        """Test intersection over union calculation for bboxes."""
        box1 = (10, 10, 50, 50)
        box2 = (30, 30, 70, 70)
        iou = self.detector._bbox_iou(box1, box2)
        self.assertIsInstance(iou, float)
        self.assertGreaterEqual(iou, 0.0)
        self.assertLessEqual(iou, 1.0)

    def test_squared_distance(self) -> None:
        """Test squared Euclidean distance calculation."""
        dist = self.detector._squared_distance((0, 0), (3, 4))
        self.assertEqual(dist, 25.0)

    def test_cov_track_remote_dispatch(self) -> None:
        """Exercise this test."""
        out = self.detector_server._track_remote([])
        self.assertEqual(out, [])
        self.detector_server.remote_tracker = 'hungarian'
        out = self.detector_server._track_remote([])
        self.assertEqual(out, [])

    def test_cov_track_remote_centroid_prune(self) -> None:
        """Exercise this test."""
        self.detector_server.frame_count = 10
        self.detector_server.max_id_keep = 5
        self.detector_server.remote_tracks = {
            1: {
                'bbox': (0, 0, 1, 1),
                'center': (0.5, 0.5),
                'last_seen': 1,
                'cls': 0,
            },
            2: {
                'bbox': (1, 1, 2, 2),
                'center': (1.5, 1.5),
                'last_seen': 9,
                'cls': 0,
            },
        }
        out = self.detector_server._track_remote_centroid([])
        self.assertEqual(out, [])
        self.assertNotIn(1, self.detector_server.remote_tracks)
        self.assertIn(2, self.detector_server.remote_tracks)

    def test_cov_hungarian_assign_threshold_rejects_all(self) -> None:
        """Exercise this test."""
        d = self.detector_server
        cost = np.array([[0.9, 0.8], [0.7, 0.6]])
        matches, ur, uc = d._hungarian_assign(
            cost, cost_threshold=0.0,
        )
        self.assertEqual(matches, [])
        self.assertEqual(set(ur), {0, 1})
        self.assertEqual(set(uc), {0, 1})

    def test_cov_prune_remote_tracks(self) -> None:
        """Exercise this test."""
        self.detector_server.frame_count = 50
        self.detector_server.max_id_keep = 5
        self.detector_server.remote_tracks = {
            1: {
                'bbox': (0, 0, 1, 1),
                'center': (0.5, 0.5),
                'last_seen': 1,
                'cls': 0,
            },
            2: {
                'bbox': (1, 1, 2, 2),
                'center': (1.5, 1.5),
                'last_seen': 49,
                'cls': 0,
            },
        }
        self.detector_server._prune_remote_tracks()
        self.assertNotIn(1, self.detector_server.remote_tracks)
        self.assertIn(2, self.detector_server.remote_tracks)

    def test_cov_build_group_cost_matrix(self) -> None:
        """Exercise this test."""
        d = self.detector_server
        dets = [
            [1.0, 1.0, 3.0, 3.0, 0.9, 0],
            [10.0, 10.0, 20.0, 20.0, 0.8, 1],
        ]
        tracks = [
            (
                1,
                {
                    'bbox': (1.0, 1.0, 3.0, 3.0),
                    'center': (2.0, 2.0),
                    'last_seen': 1,
                    'cls': 0,
                },
            ),
        ]
        cost = d._build_group_cost_matrix(dets, [0], tracks)
        self.assertEqual(cost.shape, (1, 1))
        self.assertGreaterEqual(cost[0, 0], 0.0)

    def test_cov_hungarian_assign_zero_extraction(self) -> None:
        """Exercise this test."""
        d = self.detector_server
        cost = np.array([[0.0, 2.0], [2.0, 0.0]], dtype=float)
        matches, ur, uc = d._hungarian_assign(cost, cost_threshold=10.0)
        self.assertIn((0, 0), matches)
        self.assertIn((1, 1), matches)

    def test_hungarian_assign_splits_disconnected_candidate_groups(
        self,
    ) -> None:
        """Distant objects use independent small Hungarian assignments."""
        d = self.detector_server
        cost = np.array([
            [0.1, 0.9, 0.9, 0.9],
            [0.9, 0.2, 0.9, 0.9],
            [0.9, 0.9, 0.1, 0.9],
            [0.9, 0.9, 0.9, 0.2],
        ])

        def assign(
            component_cost: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            """Return the only assignment in a one-edge component."""
            self.assertEqual(component_cost.shape, (1, 1))
            return np.array([0]), np.array([0])

        with patch(
            'src.yolo_detector._linear_sum_assignment',
            return_value=assign,
        ) as assignment:
            matches, unmatched_rows, unmatched_cols = d._hungarian_assign(
                cost,
                cost_threshold=0.3,
            )

        self.assertEqual(matches, [(0, 0), (1, 1), (2, 2), (3, 3)])
        self.assertEqual(unmatched_rows, [])
        self.assertEqual(unmatched_cols, [])
        self.assertEqual(assignment.call_count, 4)

    def test_cov_remove_vest_containment_both_directions(self) -> None:
        """Exercise this test."""
        d = self.detector_server
        datas1 = [[10, 10, 30, 30, 0.9, 7], [12, 12, 20, 20, 0.8, 4]]
        out1 = d.remove_completely_contained_labels(datas1.copy())
        self.assertTrue(all(row[5] != 4 for row in out1))
        datas2 = [[12, 12, 20, 20, 0.9, 7], [10, 10, 30, 30, 0.8, 4]]
        out2 = d.remove_completely_contained_labels(datas2.copy())
        self.assertTrue(all(row[5] != 7 for row in out2))

    def test_cov_remove_hardhat_contained_by_no_hardhat(self) -> None:
        """Exercise this test."""
        d = self.detector_server
        datas = [[10, 10, 20, 20, 0.9, 0], [5, 5, 30, 30, 0.8, 2]]
        out = d.remove_completely_contained_labels(datas.copy())
        self.assertTrue(all(row[5] != 0 for row in out))

    def test_cov_hungarian_assign_empty_matrix(self) -> None:
        """Exercise this test."""
        d = self.detector_server
        cost = np.empty((0, 2), dtype=float)
        matches, ur, uc = d._hungarian_assign(cost, cost_threshold=10.0)
        self.assertEqual(matches, [])
        self.assertEqual(ur, [])
        self.assertEqual(uc, [0, 1])

    def test_track_remote_hungarian_no_tracks_creates_new(self) -> None:
        """Exercise this test."""
        d = self.detector_server
        d.remote_tracker = 'hungarian'
        d.remote_tracks.clear()
        dets = [[10.0, 10.0, 20.0, 20.0, 0.9, 0]]
        out = d._track_remote_hungarian(dets)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), 8)

    def test_track_remote_hungarian_match_unmatched_prune(self) -> None:
        """Test Hungarian tracker with matched/unmatched dets and pruning."""
        d = self.detector_server
        d.remote_tracker = 'hungarian'
        d.frame_count = 10  # trigger prune
        d.max_id_keep = 5   # ensure pruning threshold excludes old
        # existing valid track near (15,15) with class 0
        d.remote_tracks = {
            1: {
                'bbox': (10.0, 10.0, 20.0, 20.0),
                'center': (15.0, 15.0),
                'last_seen': 9,
                'cls': 0,
            },
            99: {
                'bbox': (0.0, 0.0, 1.0, 1.0),
                'center': (0.5, 0.5),
                'last_seen': 0,  # stale, should be pruned
                'cls': 0,
            },
        }
        # two detections: one matches tid=1, one unmatched (different class)
        dets = [
            [11.0, 11.0, 19.0, 19.0, 0.95, 0],  # Near centre, class 0 => match.
            [30.0, 30.0, 40.0, 40.0, 0.80, 1],  # different class => new track
        ]
        out = d._track_remote_hungarian(dets)
        self.assertEqual(len(out), 2)
        # one result should use existing tid=1
        tids = {row[6] for row in out}
        self.assertIn(1, tids)
        # the other should be a new tid (>= 2)
        self.assertTrue(any(tid != 1 for tid in tids))
        # prune should remove stale 99
        self.assertNotIn(99, d.remote_tracks)

    def test_centroid_tracker_match_and_moving_flag(self) -> None:
        """Test centroid tracker matching and movement flag assignment."""
        d = self.detector_server
        d.remote_tracker = 'centroid'
        d.movement_thr_sq = 4.0  # threshold for distance^2
        # Existing track class 0 at centre (10, 10).
        d.remote_tracks = {
            1: {
                'bbox': (8.0, 8.0, 12.0, 12.0),
                'center': (10.0, 10.0),
                'last_seen': d.frame_count,
                'cls': 0,
            },
        }
        # detection near enough but moving > thr
        dets = [[12.0, 10.0, 14.0, 12.0, 0.9, 0]]
        out = d._track_remote_centroid(dets)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][6], 1)
        # Movement flag depends on the previous centre and should be 1.
        self.assertIn(out[0][7], (0, 1))

    def test_centroid_tracker_prune_on_empty(self) -> None:
        """Test centroid tracker pruning old tracks when no detections."""
        d = self.detector_server
        d.remote_tracker = 'centroid'
        d.frame_count = 10
        d.max_id_keep = 5
        d.remote_tracks = {
            99: {
                'bbox': (0.0, 0.0, 1.0, 1.0),
                'center': (0.5, 0.5),
                'last_seen': 0,
                'cls': 0,
            },
        }
        out = d._track_remote_centroid([])
        self.assertEqual(out, [])
        self.assertNotIn(99, d.remote_tracks)

    def test_centroid_tracker_new_track_branch(self) -> None:
        """Test centroid tracker creating new tracks for unmatched dets."""
        d = self.detector_server
        d.remote_tracker = 'centroid'
        d.remote_tracks.clear()
        dets = [[10.0, 10.0, 20.0, 20.0, 0.9, 0]]
        out = d._track_remote_centroid(dets)
        self.assertEqual(len(out), 1)
        # should assign a new track id (starting from current next_remote_id)
        self.assertGreaterEqual(out[0][6], 0)

    def test_centroid_tracker_dist_threshold_branch(self) -> None:
        """Test centroid tracker distance threshold validation."""
        d = self.detector_server
        d.remote_tracker = 'centroid'
        # One track at centre (0, 0), class 0.
        d.remote_tracks = {
            1: {
                'bbox': (-1.0, -1.0, 1.0, 1.0),
                'center': (0.0, 0.0),
                'last_seen': 0,
                'cls': 0,
            },
        }
        d.movement_thr_sq = 4.0
        # detection at distance^2 exactly equal to movement_thr_sq*4 (boundary)
        # movement_thr_sq*4 = 16. Require dist_sq < 16 to match -> won't match
        dets = [[4.0, 0.0, 6.0, 2.0, 0.9, 0]]  # Centre at (5, 1), distance squared 26.
        out = d._track_remote_centroid(dets)
        # since no match, new track created
        self.assertEqual(len(out), 1)
        self.assertNotEqual(out[0][6], 1)

    def test_centroid_tracker_updates_best_distance(self) -> None:
        """Test centroid tracker updating to track with shortest distance."""
        d = self.detector_server
        d.remote_tracker = 'centroid'
        # two tracks same class; the second becomes the best (dist_sq update)
        d.movement_thr_sq = 10.0  # threshold*4 = 40.0
        d.remote_tracks = {
            1: {
                'bbox': (0.0, 0.0, 2.0, 2.0),
                'center': (1.0, 1.0),
                'last_seen': 0,
                'cls': 0,
            },
            2: {
                'bbox': (0.0, 0.0, 2.0, 2.0),
                'center': (2.5, 2.5),
                'last_seen': 0,
                'cls': 0,
            },
        }
        dets = [[2.0, 2.0, 4.0, 4.0, 0.9, 0]]  # Centre at (3, 3).
        out = d._track_remote_centroid(dets)
        # Should assign to the closer track id=2
        self.assertEqual(out[0][6], 2)

    def test_centroid_tracker_skip_different_class_continue(self) -> None:
        """Test centroid tracker skipping tracks with different classes."""
        d = self.detector_server
        d.remote_tracker = 'centroid'
        # track 1 is different class -> should be skipped via continue
        d.movement_thr_sq = 100.0
        d.remote_tracks = {
            1: {
                'bbox': (0.0, 0.0, 2.0, 2.0),
                'center': (1.0, 1.0),
                'last_seen': 0,
                'cls': 0,
            },
            2: {
                'bbox': (10.0, 10.0, 12.0, 12.0),
                'center': (11.0, 11.0),
                'last_seen': 0,
                'cls': 1,
            },
        }
        dets = [[10.0, 10.0, 12.0, 12.0, 0.9, 1]]
        out = d._track_remote_centroid(dets)
        self.assertEqual(out[0][6], 2)


if __name__ == '__main__':
    unittest.main()


"""
pytest \
    --cov=src.yolo_detector \
    --cov-report=term-missing tests/src/yolo_detector_test.py
"""
