from __future__ import annotations

import unittest

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

    def test_legacy_alias_normalises_to_canonical_code(self) -> None:
        self.assertEqual(
            normalise_violation_type(
                'no_helmet',
            ), 'no_safety_helmet',
        )
        self.assertEqual(
            normalise_violation_type(
                'near_vehicle',
            ), 'near_vehicle',
        )
        self.assertIsNone(normalise_violation_type('free-text warning'))

    def test_invalid_and_inactive_warning_values_are_ignored(self) -> None:
        """Malformed, empty, and zero-count warning payloads are harmless."""
        self.assertIsNone(normalise_violation_type('   '))
        self.assertEqual(violation_type_codes_from_warnings('{bad json'), [])
        self.assertEqual(
            violation_type_codes_from_warnings(['not-a-mapping']), [],
        )
        self.assertEqual(
            violation_type_codes_from_warnings(
                {
                    'warning_no_hardhat': {'count': False},
                    'warning_no_safety_vest': {'count': 0},
                    'warning_close_to_vehicle': {'count': 1.5},
                },
            ),
            ['near_vehicle'],
        )
