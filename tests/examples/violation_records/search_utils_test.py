from __future__ import annotations

import unittest

from examples.violation_records.search_utils import SearchUtils


class TestSearchUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.search = SearchUtils()

    def test_chinese_phrases_expand_without_model_segmentation(self) -> None:
        expanded = self.search.expand_synonyms('人安全帽和背心')
        self.assertIn('person', expanded)
        self.assertIn('helmet', expanded)
        self.assertIn('safety_vest', expanded)
        self.assertIn('人安全帽和背心', expanded)

    def test_specific_warning_phrase_precedes_generic_equipment(self) -> None:
        for query in ('未戴安全帽', 'NO SAFETY VEST', 'controlled area'):
            with self.subTest(query=query):
                expanded = self.search.expand_synonyms(query)
                expected = {
                    '未戴安全帽': 'warning_no_hardhat',
                    'NO SAFETY VEST': 'warning_no_safety_vest',
                    'controlled area': 'warning_people_in_controlled_area',
                }[query]
                self.assertIn(expected, expanded)
        self.assertNotIn('helmet', self.search.expand_synonyms('未戴安全帽'))

    def test_literal_camera_names_and_unknown_languages_remain_searchable(
        self,
    ) -> None:
        for query in ('Camera_A-12', '無人機', 'เครน'):
            with self.subTest(query=query):
                self.assertTrue(self.search.expand_synonyms(query))
        self.assertIn('camera_a', self.search.expand_synonyms('Camera_A-12'))

    def test_equipment_and_canonical_codes_match_persisted_warning_keys(
        self,
    ) -> None:
        for query in ('helmet', '安全帽', 'no_helmet', 'no_safety_helmet'):
            with self.subTest(query=query):
                self.assertIn(
                    'warning_no_hardhat',
                    self.search.expand_synonyms(query),
                )

    def test_english_boundaries_do_not_match_hat_inside_what(self) -> None:
        self.assertEqual(self.search.expand_synonyms('what'), ['what'])

    def test_blank_and_stop_words_return_no_predicates(self) -> None:
        self.assertEqual(self.search.expand_synonyms('  '), [])
        self.assertEqual(self.search.expand_synonyms('the and 的'), [])

    def test_unicode_normalization_and_cache_are_deterministic(self) -> None:
        self.assertEqual(
            self.search.expand_synonyms('ＨＥＬＭＥＴ'),
            self.search.expand_synonyms('helmet'),
        )
        self.assertEqual(
            self.search.expand_synonyms('帽背心'),
            self.search.expand_synonyms('帽背心'),
        )

    def test_synonym_index_ignores_empty_configuration_keys(self) -> None:
        self.assertEqual(
            SearchUtils._build_synonym_index(
                [('', ['ignored']), ('Helmet', ['helmet'])],
            ),
            {'h': (('helmet', ('helmet',)),)},
        )
