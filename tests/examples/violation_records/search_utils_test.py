from __future__ import annotations

import unittest

from examples.violation_records.search_utils import SearchUtils


class TestSearchUtils(unittest.TestCase):
    """
    Test suite for the SearchUtils class.
    """

    def setUp(self) -> None:
        """
        Set up the test environment by creating an instance of SearchUtils
        and mocking the CKIP word segmenter.
        """
        # Create a SearchUtils instance, specifying device=-1 (CPU).
        self.su = SearchUtils(device=-1)

        # Mock the CKIP word segmenter to return a fixed token list.
        self.su.ws_driver = lambda inputs: [
            ['有', '人', '沒', '戴', '安全帽', '和', '未', '穿', '背心'],
        ]

    def test_tokenize(self) -> None:
        """
        Test the tokenisation and stop-word removal functionality.
        """
        user_input = '有人沒戴安全帽和未穿背心'
        tokens = self.su.tokenize(user_input)
        expected = ['有', '人', '沒', '戴', '安全帽', '未', '穿', '背心']
        self.assertEqual(
            tokens, expected,
            'Tokenisation or stop-word removal failed.',
        )

    def test_expand_synonyms(self) -> None:
        """
        Test the synonym expansion functionality.

        The tokens extracted from "有人沒戴安全帽和未穿背心" should be expanded
        to include synonyms such as "hardhat", "no_hardhat", "safety_vest",
        "no_safety_vest", and "vest". Original tokens should also be preserved.
        """
        user_input = '有人沒戴安全帽和未穿背心'
        expanded = self.su.expand_synonyms(user_input)

        # Check if "人" expands to "person".
        self.assertIn(
            'person', expanded,
            "Synonym expansion for '人' is missing 'person'.",
        )

        # Check if "安全帽" expands to "hardhat" and "no_hardhat".
        self.assertIn(
            'hardhat', expanded,
            "Synonym expansion for '安全帽' is missing 'hardhat'.",
        )
        self.assertIn(
            'no_hardhat', expanded,
            "Synonym expansion for '安全帽' is missing 'no_hardhat'.",
        )
        self.assertIn(
            '安全帽', expanded,
            "Original token '安全帽' is missing in the expanded list.",
        )

        # Check if "背心" expands to "safety_vest", "no_safety_vest", and "vest".
        self.assertIn(
            'safety_vest', expanded,
            "Synonym expansion for '背心' is missing 'safety_vest'.",
        )
        self.assertIn(
            'no_safety_vest', expanded,
            "Synonym expansion for '背心' is missing 'no_safety_vest'.",
        )
        self.assertIn(
            'vest', expanded,
            "Synonym expansion for '背心' is missing 'vest'.",
        )
        self.assertIn(
            '背心', expanded,
            "Original token '背心' is missing in the expanded list.",
        )

        # Ensure original (non-stop) tokens are preserved.
        for token in ['有', '人', '沒', '戴', '安全帽', '未', '穿', '背心']:
            self.assertIn(
                token, expanded,
                f"Original token '{token}' is missing from expanded results.",
            )

    def test_build_elasticsearch_query(self) -> None:
        """
        Test the construction of Elasticsearch wildcard queries.

        The expanded synonyms of "有人沒戴安全帽和未穿背心" should generate
        wildcard conditions for both 'stream_name' and 'warnings_json' fields.
        """
        user_input = '有人沒戴安全帽和未穿背心'
        expanded = self.su.expand_synonyms(user_input)
        es_query = self.su.build_elasticsearch_query(user_input)

        # Check if the query structure is correct.
        self.assertIsInstance(
            es_query, dict, 'Elasticsearch query should be a dictionary.',
        )
        self.assertIn(
            'query', es_query,
            "'query' key not found in Elasticsearch query.",
        )
        self.assertIn(
            'bool', es_query['query'],
            "'bool' key not found in Elasticsearch query structure.",
        )
        self.assertIn(
            'should', es_query['query']['bool'],
            "'should' key not found in Elasticsearch bool query.",
        )

        should_list = es_query['query']['bool']['should']

        # We expect 2 wildcard queries per expanded keyword
        # (one for 'stream_name', one for 'warnings_json').
        expected_length = len(expanded) * 2
        self.assertEqual(
            len(should_list),
            expected_length,
            f"Expected {expected_length} wildcard queries "
            'but found {len(should_list)}.',
        )

        # Verify each condition is structured with a wildcard key.
        for condition in should_list:
            self.assertIsInstance(
                condition,
                dict,
                "Each condition in 'should' should be a dictionary.",
            )
            self.assertIn(
                'wildcard', condition,
                "Each condition should contain a 'wildcard' key.",
            )


if __name__ == '__main__':
    unittest.main()

'''
pytest \
    --cov=examples.violation_records.search_utils \
    --cov-report=term-missing \
        tests/examples/violation_records/search_utils_test.py
'''
