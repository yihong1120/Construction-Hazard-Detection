# tests/examples/mcp_server/schemas_test.py
from __future__ import annotations

import typing
import unittest

from examples.mcp_server import schemas as S
# Target module under test


class TypeAliasTests(unittest.TestCase):
    """Tests for TypeAlias shapes declared in schemas.py.

    We validate that the public aliases keep their intended nesting and
    atomic element types. Structural checks rely on typing.get_origin/args.
    """

    def test_floatbbox_alias_structure(self) -> None:
        """FloatBBox should be `list[float]`."""
        origin = typing.get_origin(S.FloatBBox)
        args = typing.get_args(S.FloatBBox)
        self.assertIs(origin, list, 'FloatBBox must alias list[...]')
        self.assertEqual(args, (float,), 'FloatBBox elements must be float')

    def test_polygonscoords_alias_structure(self) -> None:
        """PolygonsCoords should be `list[list[list[float]]]`."""
        outer_origin = typing.get_origin(S.PolygonsCoords)
        outer_args = typing.get_args(S.PolygonsCoords)
        self.assertIs(outer_origin, list)
        self.assertEqual(len(outer_args), 1)

        mid = outer_args[0]
        mid_origin = typing.get_origin(mid)
        mid_args = typing.get_args(mid)
        self.assertIs(mid_origin, list)
        self.assertEqual(len(mid_args), 1)

        inner = mid_args[0]
        inner_origin = typing.get_origin(inner)
        inner_args = typing.get_args(inner)
        self.assertIs(inner_origin, list)
        self.assertEqual(inner_args, (float,))


class TypedDictShapeTests(unittest.TestCase):
    """Shape and metadata tests for all TypedDicts in schemas.py."""

    def _resolved_annotations(self, cls: type) -> dict[str, object]:
        """Return fully resolved annotations for a TypedDict class.

        Uses typing.get_type_hints with the module's globals/locals so that
        ForwardRef strings (due to `from __future__ import annotations`)
        are evaluated into real typing objects.
        """
        return typing.get_type_hints(cls, globalns=vars(S), localns=vars(S))

    def _assert_annotations(
        self,
        cls: type,  # a TypedDict class
        expected: typing.Mapping[str, object],
    ) -> None:
        """Assert that a TypedDict class has the resolved keys and types.

        Args:
            cls: The TypedDict type object (class).
            expected: Mapping of key to expected, already-resolved annotation.
        """
        anns = self._resolved_annotations(cls)
        self.assertSetEqual(set(anns.keys()), set(expected.keys()))
        for key, want in expected.items():
            got = anns[key]
            msg = (
                f"Mismatch on key '{key}': "
                f"{got!r} != {want!r}"
            )
            self.assertEqual(got, want, msg)

    def test_detection_like_dict_annotations_and_total(self) -> None:
        """DetectionLikeDict should expose optional keys with correct types."""
        expected = {
            'bbox': list[float],
            'box': list[float],
            'confidence': float,
            'conf': float,
            'class_': int,
            'cls': int,
        }
        self._assert_annotations(S.DetectionLikeDict, expected)

        # total=False means *all* keys are optional at runtime typing level.
        self.assertFalse(
            getattr(S.DetectionLikeDict, '__total__', True),
            'DetectionLikeDict must be declared with total=False',
        )

    def test_inference_meta_annotations_and_total(self) -> None:
        """InferenceMeta must be total and have the expected keys."""
        expected = {
            'model_key': str,
            'engine': str,
            'tracker': str,
            'confidence_threshold': float,
            'track_objects': bool,
            'frame_size': list[int],
        }
        self._assert_annotations(S.InferenceMeta, expected)
        self.assertTrue(
            getattr(S.InferenceMeta, '__total__', True),
            'InferenceMeta should be total (all keys required).',
        )

    def test_inference_response_annotations_and_total(self) -> None:
        """InferenceResponse must reference resolved aliases and sub-metas."""
        # Note: detections uses FloatBBox (list[float]) inside a list,
        # so after resolution it becomes list[list[float]].
        expected = {
            'detections': list[list[float]],
            'tracked': list[list[float]],
            'meta': S.InferenceMeta,
        }
        self._assert_annotations(S.InferenceResponse, expected)
        self.assertTrue(getattr(S.InferenceResponse, '__total__', True))

    def test_hazard_meta_annotations_and_total(self) -> None:
        """HazardMeta includes optional ints and booleans (union with None)."""
        expected = {
            'image_width': int | None,
            'image_height': int | None,
            'working_hour_only': bool | None,
            'site_config_provided': bool,
        }
        self._assert_annotations(S.HazardMeta, expected)
        self.assertTrue(getattr(S.HazardMeta, '__total__', True))

    def test_hazard_response_annotations_and_total(self) -> None:
        """HazardResponse contains warning counts and polygon
        coordinate lists.
        """
        expected = {
            'warnings': dict[str, dict[str, int]],
            'cone_polygons': S.PolygonsCoords,
            'pole_polygons': S.PolygonsCoords,
            'meta': S.HazardMeta,
        }
        self._assert_annotations(S.HazardResponse, expected)
        self.assertTrue(getattr(S.HazardResponse, '__total__', True))


class StructuralValidatorsTests(unittest.TestCase):
    """Runtime structural validation using lightweight checkers.

    TypedDicts are not enforced at runtime by CPython. To guard accidental
    regressions in shape, we validate representative instances with small,
    explicit validators. This keeps tests deterministic and dependency-free.
    """

    def _is_list_of(self, xs: object, tp: type) -> bool:
        return isinstance(xs, list) and all(isinstance(x, tp) for x in xs)

    def _is_floatbbox(self, xs: object) -> bool:
        return self._is_list_of(xs, float)

    def _is_polygonscoords(self, xs: object) -> bool:
        # list[list[list[float]]]
        if not isinstance(xs, list):
            return False
        for poly in xs:
            if not isinstance(poly, list):
                return False
            for ring in poly:
                if not self._is_list_of(ring, float):
                    return False
        return True

    def _validate_inference_meta(self, m: object) -> bool:
        if not isinstance(m, dict):
            return False
        required = {
            'model_key': str,
            'engine': str,
            'tracker': str,
            'confidence_threshold': float,
            'track_objects': bool,
            'frame_size': list,
        }
        for k, t in required.items():
            if k not in m or not isinstance(m[k], t):
                return False
        fs = m['frame_size']
        if not (
            isinstance(fs, list)
            and len(fs) == 2
            and all(isinstance(i, int) for i in fs)
        ):
            return False
        return True

    def _validate_inference_response(self, r: object) -> bool:
        if not isinstance(r, dict):
            return False
        if 'detections' not in r or 'tracked' not in r or 'meta' not in r:
            return False
        if not (
            isinstance(r['detections'], list)
            and all(self._is_floatbbox(bb) for bb in r['detections'])
        ):
            return False
        if not (
            isinstance(r['tracked'], list)
            and all(self._is_list_of(t, float) for t in r['tracked'])
        ):
            return False
        return self._validate_inference_meta(r['meta'])

    def _validate_hazard_meta(self, m: object) -> bool:
        if not isinstance(m, dict):
            return False
        int_or_none = (int, type(None))
        bool_or_none = (bool, type(None))
        for key, allowed in (
            ('image_width', int_or_none),
            ('image_height', int_or_none),
            ('working_hour_only', bool_or_none),
        ):
            if key not in m or not isinstance(m[key], allowed):
                return False
        return (
            'site_config_provided' in m
            and isinstance(m['site_config_provided'], bool)
        )

    def _validate_hazard_response(self, r: object) -> bool:
        if not isinstance(r, dict):
            return False
        if not {
            'warnings',
            'cone_polygons',
            'pole_polygons',
            'meta',
        } <= set(r.keys()):
            return False
        w = r['warnings']
        if not isinstance(w, dict):
            return False
        for k1, v1 in w.items():
            if not isinstance(k1, str) or not isinstance(v1, dict):
                return False
            for k2, v2 in v1.items():
                if not isinstance(k2, str) or not isinstance(v2, int):
                    return False
        if not (
            self._is_polygonscoords(r['cone_polygons'])
            and self._is_polygonscoords(r['pole_polygons'])
        ):
            return False
        return self._validate_hazard_meta(r['meta'])

    def test_detection_like_dict_variants_are_acceptable(self) -> None:
        """Demonstrate permissible variants for DetectionLikeDict keys."""
        a: S.DetectionLikeDict = {
            'bbox': [1.0, 2.0, 3.0, 4.0],
            'confidence': 0.95,
            'class_': 5,
        }
        self.assertIn('bbox', a)
        b: S.DetectionLikeDict = {
            'box': [10.0, 20.0, 30.0, 40.0],
            'conf': 0.88,
            'cls': 7,
        }
        self.assertIn('box', b)

    def test_inference_response_valid_instance(self) -> None:
        """A fully valid InferenceResponse passes the structural validator."""
        ir: S.InferenceResponse = {
            'detections': [[0.0, 1.0, 2.0, 3.0], [10.5, 11.5, 12.5, 13.5]],
            'tracked': [[0.0, 0.1, 0.2, 1.0], [10.0, 10.1, 10.2, 2.0]],
            'meta': {
                'model_key': 'yolo11n',
                'engine': 'onnx',
                'tracker': 'bytetrack',
                'confidence_threshold': 0.5,
                'track_objects': True,
                'frame_size': [1920, 1080],
            },
        }
        self.assertTrue(self._validate_inference_response(ir))

    def test_hazard_response_valid_instance(self) -> None:
        """A fully valid HazardResponse passes the structural validator."""
        hr: S.HazardResponse = {
            'warnings': {'zone_a': {'no_helmet': 3, 'no_vest': 1}},
            'cone_polygons': [[[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]],
            'pole_polygons': [[[10.0, 10.0], [11.0, 10.0], [10.5, 11.0]]],
            'meta': {
                'image_width': 1920,
                'image_height': 1080,
                'working_hour_only': None,
                'site_config_provided': True,
            },
        }
        self.assertTrue(self._validate_hazard_response(hr))

    # ---------- negative cases (guardrails) ----------

    def test_floatbbox_rejects_non_float_elements(self) -> None:
        """FloatBBox validator rejects lists containing non-floats."""
        bad = [1.0, 2, 3.0, 4.0]  # contains an int
        self.assertFalse(self._is_floatbbox(bad))

    def test_polygonscoords_rejects_wrong_nesting(self) -> None:
        """PolygonsCoords validator rejects malformed nestings."""
        bad = [[0.0, 1.0, 2.0]]  # missing two levels of nesting
        self.assertFalse(self._is_polygonscoords(bad))

    def test_inference_response_rejects_bad_meta(self) -> None:
        """InferenceResponse validator fails when frame_size is malformed."""
        ir = {
            'detections': [[0.0, 1.0, 2.0, 3.0]],
            'tracked': [[0.0, 0.1, 0.2, 1.0]],
            'meta': {
                'model_key': 'yolo',
                'engine': 'onnx',
                'tracker': 'bytetrack',
                'confidence_threshold': 0.4,
                'track_objects': True,
                'frame_size': [1920, '1080'],  # wrong type
            },
        }
        self.assertFalse(self._validate_inference_response(ir))

    def test_hazard_response_rejects_wrong_warnings_type(self) -> None:
        """HazardResponse validator fails when warning counts are not ints."""
        hr = {
            'warnings': {'zone_a': {'no_helmet': '3'}},  # wrong type
            'cone_polygons': [[[0.0, 0.0], [1.0, 0.0]]],
            'pole_polygons': [[[1.0, 1.0], [2.0, 2.0]]],
            'meta': {
                'image_width': None,
                'image_height': None,
                'working_hour_only': False,
                'site_config_provided': True,
            },
        }
        self.assertFalse(self._validate_hazard_response(hr))


if __name__ == '__main__':
    unittest.main()

'''
pytest --cov=examples.mcp_server.schemas\
    --cov-report=term-missing\
        tests/examples/mcp_server/schemas_test.py
'''
