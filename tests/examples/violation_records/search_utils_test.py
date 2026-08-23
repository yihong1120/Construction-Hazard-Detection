from __future__ import annotations

import unittest

from examples.violation_records.search_utils import SearchUtils


class TestSearchUtils(unittest.TestCase):
    """Verify the PostgreSQL-search synonym behaviour kept in production."""

    def setUp(self) -> None:
        """Create a deterministic segmenter without loading model weights."""
        self.search = SearchUtils(device=-1)
        self.search.ws_driver = lambda _inputs: [
            ['人', '安全帽', '和', '背心'],
        ]

    def test_expand_synonyms_keeps_detector_labels(self) -> None:
        """Expanded terms include original CJK terms and detector labels."""
        expanded = self.search.expand_synonyms('人安全帽和背心')
        self.assertIn('person', expanded)
        self.assertIn('hardhat', expanded)
        self.assertIn('safety_vest', expanded)
        self.assertIn('安全帽', expanded)

    def test_synonym_index_ignores_empty_configuration_keys(self) -> None:
        """Empty configuration entries cannot participate in matching."""
        index = SearchUtils._build_synonym_index(
            [
                ('', ['ignored']),
                ('Helmet', ['hardhat']),
            ],
        )
        self.assertEqual(index, {'h': (('helmet', ('hardhat',)),)})
