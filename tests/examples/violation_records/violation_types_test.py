from __future__ import annotations

import unittest

from pydantic import ValidationError

from examples.violation_records.violation_types import (
    normalise_violation_type,
)
from examples.violation_records.violation_types import (
    violation_type_codes_from_warnings,
)


class TestViolationTypes(unittest.TestCase):
    def test_derives_codes_from_active_structured_warning_keys(self) -> None:
        codes = violation_type_codes_from_warnings(
            '{'
            '"warning_no_hardhat": {"count": 1}, '
            '"warning_close_to_vehicle": {"count": 2}, '
            '"warning_no_safety_vest": {"count": 0}'
            '}',
        )

        self.assertEqual(codes, ['no_safety_helmet', 'near_vehicle'])

    def test_requires_canonical_type_code(self) -> None:
        self.assertIsNone(normalise_violation_type('no_helmet'))
        self.assertEqual(
            normalise_violation_type(
                'near_vehicle',
            ), 'near_vehicle',
        )
        self.assertIsNone(normalise_violation_type('free-text warning'))

    def test_requires_structured_warning_payload(self) -> None:
        """Persisted warnings have one canonical JSON structure."""
        self.assertIsNone(normalise_violation_type('   '))
        with self.assertRaises(ValidationError):
            violation_type_codes_from_warnings('{bad json')
        with self.assertRaises(ValidationError):
            violation_type_codes_from_warnings(
                '{"warning_no_hardhat": {"count": true}}',
            )
        self.assertEqual(
            violation_type_codes_from_warnings(
                '{"warning_no_safety_vest": {"count": 0}}',
            ),
            [],
        )
