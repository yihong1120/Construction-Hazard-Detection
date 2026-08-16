from __future__ import annotations

import unittest
from typing import Any
from typing import cast

from examples.local_notification_server.lang_config import LANGUAGES
from examples.local_notification_server.lang_config import NotificationLanguage
from examples.local_notification_server.lang_config import Translator


class TestLangConfig(unittest.TestCase):
    """
    Test suite for language configuration (LANGUAGES) and the Translator class.
    """

    def setUp(self) -> None:
        """
        Set up the test environment.
        """
        self.original_languages: dict[str, dict[str, str]] = LANGUAGES.copy()

        # Define the supported language codes (matching lang_config.py)
        self.supported_langs: set[str] = {
            'en-GB', 'zh-TW', 'zh-CN', 'fr-FR', 'vi-VN', 'id-ID', 'th-TH',
            'ja-JP',
        }

        # Each language must include these keys
        self.expected_keys: set[str] = {
            'warning_people_in_controlled_area',
            'warning_no_hardhat',
            'warning_no_safety_vest',
            'warning_close_to_machinery',
            'warning_close_to_vehicle',
            'no_warning',
            'machinery',
            'vehicle',
            'helmet',
            'person',
            'no_helmet',
            'vest',
            'no_vest',
            'mask',
            'no_mask',
            'cone',
        }

    def tearDown(self) -> None:
        """
        Restore the original LANGUAGES.
        """
        LANGUAGES.clear()
        LANGUAGES.update(self.original_languages)

    def test_supported_languages_exist(self) -> None:
        """
        Check that all required language codes exist in LANGUAGES.
        """
        for lang in self.supported_langs:
            self.assertIn(
                lang,
                LANGUAGES,
                f"Language '{lang}' is missing from LANGUAGES.",
            )

    def test_all_keys_exist_in_each_language(self) -> None:
        """
        Verify each language has all the expected translation keys.
        """
        for lang in self.supported_langs:
            self.assertIn(lang, LANGUAGES, f"{lang} not found in LANGUAGES.")
            translations: dict[str, str] = LANGUAGES[lang]
            for key in self.expected_keys:
                self.assertIn(
                    key,
                    translations,
                    f"Key '{key}' missing in language '{lang}'.",
                )

    def test_translate_from_dict_basic(self) -> None:
        """
        Test basic behaviour of Translator.translate_from_dict.

        1) Correctly replaces placeholders.
        2) Returns the string if no placeholder is present.
        3) Fails for a warning key that has no translation.
        """
        body_dict: dict[str, dict[str, Any]] = {
            'warning_close_to_vehicle': {'count': '3'},
            'warning_no_safety_vest': {'count': '1'},  # Provide placeholder
        }
        language: NotificationLanguage = 'zh-TW'

        result = Translator.translate_from_dict(body_dict, language)
        self.assertIsInstance(result, list)

        # 1) Check placeholder is replaced: '有3人過於靠近車輛'
        self.assertIn('有3人過於靠近車輛', result[0])

        # 2) Check replaced placeholder for safety vest
        self.assertIn('有1人未穿著安全背心', result[1])

        with self.assertRaises(KeyError):
            Translator.translate_from_dict(
                {'non_existent_key': {'test': '1'}},
                language,
            )

    def test_translate_from_dict_rejects_unsupported_language(self) -> None:
        """Unsupported language codes fail instead of changing locale."""
        body_dict: dict[str, dict[str, Any]] = {
            'warning_close_to_vehicle': {'count': '2'},
        }
        language = 'xx-XX'  # Invalid code

        with self.assertRaises(KeyError):
            Translator.translate_from_dict(
                body_dict,
                cast(NotificationLanguage, language),
            )

    def test_translate_from_dict_placeholder_replacement(self) -> None:
        """
        Check that placeholders like {count} are replaced with actual values.
        """
        body_dict: dict[str, dict[str, Any]] = {
            'warning_people_in_controlled_area': {'count': '5'},
        }
        language: NotificationLanguage = 'en-GB'

        result = Translator.translate_from_dict(body_dict, language)
        # Expect: "Warning: 5 people have entered the controlled area!"
        self.assertIn('5 people have entered the controlled area!', result[0])

    def test_missing_placeholder(self) -> None:
        """
        Test that missing placeholders in the dictionary are
        handled gracefully.
        """
        body_dict: dict[str, dict[str, Any]] = {
            'warning_close_to_machinery': {'x': '1'},
        }
        language: NotificationLanguage = 'en-GB'

        result = Translator.translate_from_dict(body_dict, language)
        self.assertIn('{count}', result[0])
        self.assertIn('Warning:', result[0])


if __name__ == '__main__':
    unittest.main()

"""
pytest \
    --cov=examples.local_notification_server.lang_config \
    --cov-report=term-missing \
    tests/examples/local_notification_server/lang_config_test.py
"""
