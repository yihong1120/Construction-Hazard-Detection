from __future__ import annotations

import unittest

from src.lang_config import LANGUAGES


class TestLangConfig(unittest.TestCase):
    def setUp(self):
        # Define the expected keys for all languages
        self.expected_keys = {
            'warning_people_in_controlled_area',
            'warning_no_hardhat',
            'warning_no_safety_vest',
            'warning_close_to_machinery',
            'label_machinery',
            'label_vehicle',
        }

        # Define all supported languages
        self.supported_languages = {
            'zh-TW', 'zh-CN', 'en', 'fr', 'vi', 'id', 'th',
        }

    def test_all_languages_exist(self):
        """
        Test that all the expected languages exist in the LANGUAGES dictionary.
        """
        for lang in self.supported_languages:
            self.assertIn(
                lang, LANGUAGES,
                f"Language {lang} is missing in LANGUAGES.",
            )

    def test_all_keys_exist_for_each_language(self):
        """
        Test that all the expected keys exist
        for each language in the LANGUAGES dictionary.
        """
        for lang, translations in LANGUAGES.items():
            for key in self.expected_keys:
                self.assertIn(
                    key, translations,
                    f"Key '{key}' is missing in language '{lang}'.",
                )

    def test_translation_values_are_not_empty(self):
        """
        Test that translation values
        for all keys in all languages are not empty.
        """
        for lang, translations in LANGUAGES.items():
            for key in self.expected_keys:
                value = translations.get(key, '')
                self.assertTrue(
                    value,
                    f"Value for key '{key}' in language '{lang}' is empty.",
                )

    def test_translation_format(self):
        """
        Test that all translations containing placeholders have correct format.
        """
        # Placeholders expected in specific translations
        placeholder_keys = {
            'warning_people_in_controlled_area': '{count}',
            'warning_close_to_machinery': '{label}',
        }

        for lang, translations in LANGUAGES.items():
            for key, placeholder in placeholder_keys.items():
                self.assertIn(
                    placeholder,
                    translations[key],
                    (
                        f"Expected placeholder '{placeholder}' not found in "
                        f"key '{key}' for language '{lang}'."
                    ),
                )


if __name__ == '__main__':
    unittest.main()
