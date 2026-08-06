from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from src.yolo_worker import YoloWorkerClient


def linear_sum_assignment(*args: Any, **kwargs: Any) -> Any:
    """Run SciPy linear assignment, importing SciPy lazily."""
    from scipy.optimize import (
        linear_sum_assignment as _assignment,
    )

    return _assignment(*args, **kwargs)


def _linear_sum_assignment() -> Any:
    """Return SciPy's Hungarian assignment function, importing it lazily."""
    return linear_sum_assignment


class YoloDetector:
    """Submit production frames to a shared worker and track its results.

    Passing `detect_with_server=False` remains a compatibility bridge for
    external callers. It lazily delegates to `LocalYoloDetector`, keeping
    Ultralytics and SAHI out of the production stream process.
    """

    def __init__(
        self,
        model_key: str = 'yolo26n',
        output_folder: str | None = None,
        detect_with_server: bool = True,
        use_ultralytics: bool = True,
        movement_thr: float = 40.0,
        fps: int = 1,
        max_id_keep: int = 10,
        remote_tracker: str = 'centroid',
        remote_cost_threshold: float = 0.7,
        worker_client: YoloWorkerClient | Any | None = None,
    ) -> None:
        """Initialise shared-worker inference and per-camera tracking state."""
        self.model_key = model_key
        self.output_folder = output_folder
        self.detect_with_server = detect_with_server
        self.worker_client = worker_client
        self.movement_thr = movement_thr
        self.movement_thr_sq = movement_thr * movement_thr
        self.frame_count = 0
        self.max_id_keep = max_id_keep
        self.remote_tracker = remote_tracker
        self.remote_cost_threshold = remote_cost_threshold
        self.remote_tracks: dict[int, dict] = {}
        self.next_remote_id = 0
        self._local_detector: Any | None = None

        if not detect_with_server:
            from src.local_yolo_detector import LocalYoloDetector

            self._local_detector = LocalYoloDetector(
                model_key=model_key,
                output_folder=output_folder,
                use_ultralytics=use_ultralytics,
                movement_thr=movement_thr,
                fps=fps,
                max_id_keep=max_id_keep,
            )

    async def _detect_local(self, frame: np.ndarray) -> list[list[float]]:
        """Delegate legacy local calls to the optional local detector."""
        if self._local_detector is None:
            raise RuntimeError(
                'Local inference is not enabled. '
                'Use LocalYoloDetector instead.',
            )
        return await self._local_detector._detect_local(frame)

    async def generate_detections(
        self,
        frame: np.ndarray,
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Generate detections through the shared worker and track them."""
        if self._local_detector is not None:
            return await self._local_detector.generate_detections(frame)

        self.frame_count += 1
        detections = await self._detect_remote(frame)
        return detections, self._track_remote(detections)

    async def _detect_remote(
        self,
        frame: np.ndarray,
    ) -> list[list[float]]:
        """Detect with the shared worker process."""
        if self.worker_client is None:
            raise RuntimeError(
                'Shared YOLO worker is required for server detection mode',
            )
        return await self.worker_client.detect(
            frame,
            model_key=self.model_key,
        )

    def track_detections(
        self,
        detections: list[list[float]],
    ) -> list[list[float]]:
        """Attach this detector's persistent remote track IDs to detections."""
        self.frame_count += 1
        return self._track_remote(detections)

    def _track_remote(self, dets: list[list[float]]) -> list[list[float]]:
        """Dispatch to the configured remote tracker implementation."""
        if self.remote_tracker == 'hungarian':
            return self._track_remote_hungarian(dets)
        return self._track_remote_centroid(dets)

    def _bbox_center(
            self, x1: float, y1: float, x2: float, y2: float,
    ) -> tuple[float, float]:
        """
        Return the center point (cx, cy) of a bbox.

        Args:
            x1: The x1 coordinate of the bbox.
            y1: The y1 coordinate of the bbox.
            x2: The x2 coordinate of the bbox.
            y2: The y2 coordinate of the bbox.

        Returns:
            The center point (cx, cy) of the bbox.
        """
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5

    def _bbox_iou(
        self,
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
    ) -> float:
        """
        Compute IoU for two boxes (x1,y1,x2,y2).

        Args:
            a: The first box (x1, y1, x2, y2).
            b: The second box (x1, y1, x2, y2).

        Returns:
            The IoU (Intersection over Union) of the two boxes.
        """
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        else:
            inter_area = 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    def _squared_distance(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
    ) -> float:
        """
        Return squared Euclidean distance between points p1 and p2.

        Args:
            p1: The first point (x, y).
            p2: The second point (x, y).

        Returns:
            The squared Euclidean distance.
        """
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx * dx + dy * dy

    def _set_remote_track(
        self,
        tid: int,
        bbox: tuple[float, float, float, float],
        cls_id: int,
        center: tuple[float, float],
    ) -> None:
        """
        Upsert a remote track's latest state.

        Args:
            tid: The track ID.
            bbox: The bounding box (x1, y1, x2, y2).
            cls_id: The class ID.
            center: The center point (cx, cy).

        Returns:
            None
        """
        self.remote_tracks[tid] = {
            'bbox': bbox,
            'center': center,
            'last_seen': self.frame_count,
            'cls': cls_id,
        }

    def _new_track_for_det(self, det: list[float]) -> list[float]:
        """
        Create a new track entry for a single detection.

        Args:
            det: A single detection [x1, y1, x2, y2, conf, cls_id].

        Returns:
            A tracked row [x1, y1, x2, y2, conf, cls_id, track_id, is_moving].
        """
        x1, y1, x2, y2, conf, cls_id = det
        cx, cy = self._bbox_center(x1, y1, x2, y2)
        tid = self.next_remote_id
        self.next_remote_id += 1
        self._set_remote_track(tid, (x1, y1, x2, y2), int(cls_id), (cx, cy))
        return [x1, y1, x2, y2, conf, cls_id, tid, 0]

    # Centroid tracker (original simple implementation)
    def _track_remote_centroid(
        self, dets: list[list[float]],
    ) -> list[list[float]]:
        """Simple centroid-based tracker for remote detections.

        Args:
            dets: List of detections [x1, y1, x2, y2, conf, cls].

        Returns:
            List with tracking info [x1, y1, x2, y2, conf, cls, track_id,
            is_moving].
        """
        if not dets:
            if self.frame_count % 10 == 0:
                self._prune_remote_tracks()
            return []

        assigned_tracks: list[list[float]] = []
        used_track_ids: set[int] = set()
        match_threshold_sq = self.movement_thr_sq * 4

        for det in dets:
            x1, y1, x2, y2, conf, cls_id = det
            cx, cy = self._bbox_center(x1, y1, x2, y2)
            best_tid = None
            best_dist_sq = float('inf')
            for tid, info in self.remote_tracks.items():
                if tid in used_track_ids or info['cls'] != cls_id:
                    continue
                tcx, tcy = info['center']
                dx = cx - tcx
                dy = cy - tcy
                dist_sq = dx * dx + dy * dy
                if dist_sq < best_dist_sq and dist_sq < match_threshold_sq:
                    best_dist_sq = dist_sq
                    best_tid = tid
            if best_tid is None:
                # new track
                tracked_row = self._new_track_for_det(det)
                assigned_tracks.append(tracked_row)
                continue
            else:
                tid = best_tid
                used_track_ids.add(tid)
                prev_center = self.remote_tracks[tid]['center']
                dist_sq_move = self._squared_distance((cx, cy), prev_center)
                moving_flag = 1 if dist_sq_move > self.movement_thr_sq else 0
            self._set_remote_track(
                tid, (x1, y1, x2, y2),
                int(cls_id), (cx, cy),
            )
            assigned_tracks.append(
                [x1, y1, x2, y2, conf, cls_id, tid, moving_flag],
            )
        if self.frame_count % 10 == 0:
            self._prune_remote_tracks()
        return assigned_tracks

    # Hungarian (global) assignment tracker
    def _track_remote_hungarian(
        self, dets: list[list[float]],
    ) -> list[list[float]]:
        """
        Global assignment tracker using Hungarian algorithm.

        Args:
            dets: List of detections [x1, y1, x2, y2, conf, cls].

        Returns:
            List with tracking info [x1, y1, x2, y2, conf, cls, track_id,
            is_moving].
        """
        if not dets:
            if self.frame_count % 10 == 0:
                self._prune_remote_tracks()
            return []

        track_items = list(self.remote_tracks.items())  # (tid, info)
        if not track_items:
            return self._assign_new_tracks_for_all(dets)

        dets_by_class: dict[int, list[int]] = {}
        for det_index, det in enumerate(dets):
            dets_by_class.setdefault(int(det[5]), []).append(det_index)

        tracks_by_class: dict[int, list[tuple[int, dict]]] = {}
        for tid, info in track_items:
            tracks_by_class.setdefault(int(info['cls']), []).append(
                (tid, info),
            )

        matched_pairs: list[tuple[int, tuple[int, dict]]] = []
        unmatched_dets: list[int] = []
        for cls_id, det_indices in dets_by_class.items():
            class_tracks = tracks_by_class.get(cls_id)
            if not class_tracks:
                unmatched_dets.extend(det_indices)
                continue

            cost_matrix = self._build_group_cost_matrix(
                dets,
                det_indices,
                class_tracks,
            )
            matches, unmatched_group_dets, _ = self._hungarian_assign(
                cost_matrix, self.remote_cost_threshold,
            )
            matched_pairs.extend(
                (det_indices[det_pos], class_tracks[track_pos])
                for det_pos, track_pos in matches
            )
            unmatched_dets.extend(
                det_indices[det_pos] for det_pos in unmatched_group_dets
            )

        assigned = self._update_matched_track_pairs(dets, matched_pairs)
        assigned += self._create_tracks_for_unmatched(dets, unmatched_dets)
        if self.frame_count % 10 == 0:
            self._prune_remote_tracks()
        return assigned

    def _assign_new_tracks_for_all(
        self, dets: list[list[float]],
    ) -> list[list[float]]:
        """
        Create brand-new tracks for each detection when no tracks exist.

        Args:
            dets: List of detections [x1, y1, x2, y2, conf, cls].

        Returns:
            List with tracking info [x1, y1, x2, y2, conf, cls, track_id,
            is_moving].
        """
        assigned: list[list[float]] = []
        for det in dets:
            assigned.append(self._new_track_for_det(det))
        return assigned

    def _build_group_cost_matrix(
        self,
        dets: list[list[float]],
        det_indices: list[int],
        track_items: list[tuple[int, dict]],
    ) -> np.ndarray:
        """
        Compute the cost matrix for one class of detections and tracks.

        Args:
            dets: List of detections [x1, y1, x2, y2, conf, cls].
            det_indices: Detection indexes for the current class.
            track_items: List of track items [(track_id, info), ...].

        Returns:
            Cost matrix as a NumPy array.
        """
        num_dets = len(det_indices)
        num_tracks = len(track_items)
        if num_dets == 0 or num_tracks == 0:
            return np.empty((num_dets, num_tracks), dtype=float)

        det_boxes = np.asarray(
            [dets[index][:4] for index in det_indices],
            dtype=float,
        )
        track_boxes = np.asarray(
            [info['bbox'] for _, info in track_items],
            dtype=float,
        )
        track_centers = np.asarray(
            [info['center'] for _, info in track_items],
            dtype=float,
        )
        inter_x1 = np.maximum(det_boxes[:, None, 0], track_boxes[None, :, 0])
        inter_y1 = np.maximum(det_boxes[:, None, 1], track_boxes[None, :, 1])
        inter_x2 = np.minimum(det_boxes[:, None, 2], track_boxes[None, :, 2])
        inter_y2 = np.minimum(det_boxes[:, None, 3], track_boxes[None, :, 3])
        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        det_area = (
            (det_boxes[:, 2] - det_boxes[:, 0]) *
            (det_boxes[:, 3] - det_boxes[:, 1])
        )
        track_area = (
            (track_boxes[:, 2] - track_boxes[:, 0]) *
            (track_boxes[:, 3] - track_boxes[:, 1])
        )
        union = det_area[:, None] + track_area[None, :] - inter_area
        iou = np.divide(
            inter_area,
            union,
            out=np.zeros_like(inter_area),
            where=union > 0,
        )

        det_centers = np.column_stack(
            (
                (det_boxes[:, 0] + det_boxes[:, 2]) * 0.5,
                (det_boxes[:, 1] + det_boxes[:, 3]) * 0.5,
            ),
        )
        dist = det_centers[:, None, :] - track_centers[None, :, :]
        dist_sq = np.sum(dist * dist, axis=2)
        denom = max(self.movement_thr_sq * 4, 1e-12)
        dist_norm = np.minimum(dist_sq / denom, 1.0)
        return 0.5 * (1 - iou) + 0.5 * dist_norm

    def _update_matched_track_pairs(
        self,
        dets: list[list[float]],
        matches: list[tuple[int, tuple[int, dict]]],
    ) -> list[list[float]]:
        """
        Update matched tracks and compute moving flags.

        Args:
            dets:
                List of detections [x1, y1, x2, y2, conf, cls].
            matches:
                List of matched pairs [(detection_index, track_item), ...].

        Returns:
            List of assigned tracks.
        """
        assigned: list[list[float]] = []

        for d_idx, track_item in matches:
            x1, y1, x2, y2, conf, cls_id = dets[d_idx]
            tid, info = track_item
            cx, cy = self._bbox_center(x1, y1, x2, y2)
            prev_center = info['center']
            dist_sq_move = self._squared_distance((cx, cy), prev_center)
            moving_flag = 1 if dist_sq_move > self.movement_thr_sq else 0
            self._set_remote_track(
                tid, (x1, y1, x2, y2),
                int(cls_id), (cx, cy),
            )
            assigned.append([x1, y1, x2, y2, conf, cls_id, tid, moving_flag])
        return assigned

    def _create_tracks_for_unmatched(
        self, dets: list[list[float]], unmatched_dets: list[int],
    ) -> list[list[float]]:
        """
        Create tracks for unmatched detections.

        Args:
            dets: List of detections [x1, y1, x2, y2, conf, cls].
            unmatched_dets: List of unmatched detection indices.

        Returns:
            List of new track representations.
        """
        assigned: list[list[float]] = []
        for d_idx in unmatched_dets:
            assigned.append(self._new_track_for_det(dets[d_idx]))
        return assigned

    # Hungarian assignment helper
    def _hungarian_assign(
        self, cost: np.ndarray, cost_threshold: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """
        Apply Hungarian algorithm on cost matrix and filter by threshold.

        Args:
            cost: Cost matrix as a 2D numpy array.
            cost_threshold: Cost threshold for filtering matches.

        Returns:
            (matches, unmatched_rows, unmatched_cols)
        """
        num_rows, num_cols = cost.shape
        if num_rows == 0 or num_cols == 0:
            return [], list(range(num_rows)), list(range(num_cols))

        candidate_mask = np.isfinite(cost) & (cost <= cost_threshold)
        if not candidate_mask.any():
            return [], list(range(num_rows)), list(range(num_cols))

        matches: list[tuple[int, int]] = []
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        remaining_rows = {
            int(row)
            for row in np.flatnonzero(candidate_mask.any(axis=1))
        }
        while remaining_rows:
            component_rows, component_cols = self._assignment_component(
                candidate_mask,
                remaining_rows.pop(),
            )
            remaining_rows.difference_update(component_rows)
            row_indices = sorted(component_rows)
            col_indices = sorted(component_cols)
            component_cost = cost[np.ix_(row_indices, col_indices)]
            component_candidates = candidate_mask[
                np.ix_(
                    row_indices,
                    col_indices,
                )
            ]
            # Invalid edges must never displace valid matches in this
            # component. The solver only sees small connected components.
            blocked_cost = max(cost_threshold + 1.0, 1.0)
            solver_cost = np.where(
                component_candidates,
                component_cost,
                blocked_cost,
            )
            row_ind, col_ind = _linear_sum_assignment()(solver_cost)
            for row, col in zip(row_ind, col_ind, strict=True):
                r = row_indices[int(row)]
                c = col_indices[int(col)]
                if not candidate_mask[r, c]:
                    continue
                matches.append((r, c))
                used_rows.add(r)
                used_cols.add(c)

        unmatched_rows = [r for r in range(num_rows) if r not in used_rows]
        unmatched_cols = [c for c in range(num_cols) if c not in used_cols]
        return matches, unmatched_rows, unmatched_cols

    @staticmethod
    def _assignment_component(
        candidate_mask: np.ndarray,
        start_row: int,
    ) -> tuple[set[int], set[int]]:
        """Return one connected valid-edge component of a cost matrix."""
        rows = {start_row}
        cols: set[int] = set()
        row_stack = [start_row]
        while row_stack:
            row = row_stack.pop()
            for col_value in np.flatnonzero(candidate_mask[row]):
                col = int(col_value)
                if col in cols:
                    continue
                cols.add(col)
                for neighbour_value in np.flatnonzero(candidate_mask[:, col]):
                    neighbour = int(neighbour_value)
                    if neighbour in rows:
                        continue
                    rows.add(neighbour)
                    row_stack.append(neighbour)
        return rows, cols

    def _prune_remote_tracks(self) -> None:
        """
        Remove remote tracks that have not been updated
        within max_id_keep frames.
        """
        threshold = self.frame_count - self.max_id_keep
        stale = [
            tid for tid, info in self.remote_tracks.items()
            if info['last_seen'] < threshold
        ]
        for tid in stale:
            self.remote_tracks.pop(tid, None)

    async def close(self) -> None:
        """Release the active worker or compatibility local detector."""
        if self._local_detector is not None:
            await self._local_detector.close()
            return
        close = getattr(self.worker_client, 'close', None)
        if close is not None:
            await close()

    def remove_overlapping_labels(
        self,
        datas: list[list[float]],
    ) -> list[list[float]]:
        """Remove overlapping non-compliance labels.

        Args:
            datas: Detection rows in YOLO ``xyxy`` format.

        Returns:
            Detection rows with redundant no-hardhat and no-vest labels
            removed.
        """
        indices = self._label_indices(datas)

        to_remove = set()
        for hardhat_index in indices[0]:
            for no_hardhat_index in indices[2]:
                overlap = self.overlap_percentage(
                    datas[hardhat_index], datas[no_hardhat_index],
                )
                if overlap > 0.8:
                    to_remove.add(no_hardhat_index)

        for safety_vest_index in indices[7]:
            for no_safety_vest_index in indices[4]:
                overlap = self.overlap_percentage(
                    datas[safety_vest_index],
                    datas[no_safety_vest_index],
                )
                if overlap > 0.8:
                    to_remove.add(no_safety_vest_index)

        return self._without_indices(datas, to_remove)

    def overlap_percentage(
        self,
        bbox1: Sequence[float],
        bbox2: Sequence[float],
    ) -> float:
        """Calculate the overlap ratio between two bounding boxes.

        Args:
            bbox1: First bounding box as ``[x1, y1, x2, y2, ...]``.
            bbox2: Second bounding box as ``[x1, y1, x2, y2, ...]``.

        Returns:
            Intersection-over-union ratio. Returns ``0.0`` when the union is
            empty.
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
        bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

        union = bbox1_area + bbox2_area - intersection_area
        return intersection_area / float(union) if union > 0 else 0.0

    def is_contained(
        self,
        inner_bbox: Sequence[float],
        outer_bbox: Sequence[float],
    ) -> bool:
        """Return whether one bounding box is contained inside another.

        Args:
            inner_bbox: Candidate inner bounding box.
            outer_bbox: Candidate outer bounding box.

        Returns:
            ``True`` when the inner box lies fully inside the outer box.
        """
        return (
            inner_bbox[0] >= outer_bbox[0]
            and inner_bbox[2] <= outer_bbox[2]
            and inner_bbox[1] >= outer_bbox[1]
            and inner_bbox[3] <= outer_bbox[3]
        )

    def remove_completely_contained_labels(
        self,
        datas: list[list[float]],
    ) -> list[list[float]]:
        """Remove labels fully contained in matching compliance boxes.

        Args:
            datas: Detection rows in YOLO ``xyxy`` format.

        Returns:
            Detection rows with redundant contained labels removed.
        """
        indices = self._label_indices(datas)

        to_remove = set()
        # Check hardhats
        for hardhat_index in indices[0]:
            for no_hardhat_index in indices[2]:
                if self.is_contained(
                    datas[no_hardhat_index],
                    datas[hardhat_index],
                ):
                    to_remove.add(no_hardhat_index)
                elif self.is_contained(
                    datas[hardhat_index],
                    datas[no_hardhat_index],
                ):
                    to_remove.add(hardhat_index)

        # Check safety vests
        for safety_vest_index in indices[7]:
            for no_safety_vest_index in indices[4]:
                if self.is_contained(
                    datas[no_safety_vest_index],
                    datas[safety_vest_index],
                ):
                    to_remove.add(no_safety_vest_index)
                elif self.is_contained(
                    datas[safety_vest_index],
                    datas[no_safety_vest_index],
                ):
                    to_remove.add(safety_vest_index)

        return self._without_indices(datas, to_remove)

    @staticmethod
    def _label_indices(datas: list[list[float]]) -> dict[int, list[int]]:
        """Return indices grouped by label id for relevant safety classes."""
        indices: dict[int, list[int]] = {0: [], 2: [], 4: [], 7: []}
        for i, detection in enumerate(datas):
            label = int(detection[5])
            if label in indices:
                indices[label].append(i)
        return indices

    @staticmethod
    def _without_indices(
        datas: list[list[float]],
        to_remove: set[int],
    ) -> list[list[float]]:
        """Return detection rows except removed indices."""
        if not to_remove:
            return datas
        return [
            detection
            for i, detection in enumerate(datas)
            if i not in to_remove
        ]


async def main() -> None:
    """Run the optional local preview CLI for backward-compatible commands."""
    from src.local_yolo_detector import main as local_main

    await local_main()


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
