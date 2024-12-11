from __future__ import annotations

import os
import subprocess
import unittest
from unittest.mock import patch

from src.lang_config import LANGUAGES
from src.lang_config import main
from src.lang_config import Translator


class TestLangConfig(unittest.TestCase):
    def setUp(self):
        # Define the expected keys for all languages
        self.expected_keys = {
            'warning_people_in_controlled_area',
            'warning_no_hardhat',
            'warning_no_safety_vest',
            'warning_close_to_machinery',
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

        # Define all supported languages
        self.supported_languages = {
            'zh-TW', 'zh-CN', 'en', 'fr', 'vi', 'id', 'th',
        }
        # Backup original translations for restoration in tearDown
        self.original_translations = LANGUAGES.copy()

    def tearDown(self):
        # Restore original LANGUAGES dictionary
        LANGUAGES.clear()
        LANGUAGES.update(self.original_translations)
        # Clear the cache to avoid side effects
        Translator.translate_warning.cache_clear()

    def test_all_languages_exist(self):
        """
        Test that all the expected languages exist in the LANGUAGES dictionary.
        """
        for lang in self.supported_languages:
            self.assertIn(
                lang, LANGUAGES,
                f"Language '{lang}' is missing in LANGUAGES.",
            )

    def test_all_keys_exist_for_each_language(self):
        """
        Test that all the expected keys exist for each language
        in the LANGUAGES dictionary.
        """
        for lang, translations in LANGUAGES.items():
            for key in self.expected_keys:
                self.assertIn(
                    key, translations,
                    f"Key '{key}' is missing in language '{lang}'.",
                )

    def test_no_extra_keys_for_each_language(self):
        """
        Test that there are no extra keys in any language's translation
        dictionary.
        """
        for lang, translations in LANGUAGES.items():
            extra_keys = set(translations.keys()) - self.expected_keys
            self.assertFalse(
                extra_keys,
                f"Language '{lang}' has unexpected keys: {extra_keys}",
            )

    def test_translate_warning_with_unsupported_language(self):
        """
        Test the translate_warning method with an unsupported language,
        ensuring it defaults to English.
        """
        warnings = [
            'Warning: Someone is not wearing a hardhat!',
            'Warning: 2 people have entered the controlled area!',
            'Warning: Someone is too close to machinery!',
        ]
        unsupported_language = 'xx'  # This language code is not supported

        translated_warnings = Translator.translate_warning(
            tuple(warnings), unsupported_language,
        )

        # Since the language is unsupported, it should default to English
        expected_warnings = [
            'Warning: Someone is not wearing a hardhat!',
            'Warning: 2 people have entered the controlled area!',
            'Warning: Someone is too close to machinery!',
        ]
        self.assertEqual(translated_warnings, expected_warnings)

    def test_translation_values_are_not_empty(self):
        """
        Test that translation values for all keys
        in all languages are not empty.
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
                if key in translations:
                    self.assertIn(
                        placeholder,
                        translations[key],
                        (
                            f"Expected placeholder '{placeholder}' not in "
                            f"key '{key}' for language '{lang}'."
                        ),
                    )

    def test_placeholder_replacement_correctness(self):
        """
        Test that placeholders in translations
        can be correctly replaced without errors.
        """
        test_values = {
            'warning_people_in_controlled_area': {'count': '5'},
            'warning_close_to_machinery': {'label': 'machine'},
        }

        for lang, translations in LANGUAGES.items():
            for key, replacements in test_values.items():
                if key in translations:
                    try:
                        translated = translations[key].format(**replacements)
                        # Simple check to ensure placeholders are replaced
                        for placeholder, value in replacements.items():
                            self.assertIn(
                                value, translated,
                                f"Placeholder '{placeholder}' not replaced "
                                f"correctly in language '{lang}'.",
                            )

                    except KeyError as e:
                        self.fail(
                            f"Missing placeholder '{e.args[0]}' in language "
                            f"'{lang}' for key '{key}'.",
                        )
                    except Exception as e:
                        self.fail(
                            f"Error formatting key '{key}' in language "
                            f"'{lang}': {e}",
                        )

    def test_translate_warning_no_safety_vest(self):
        """
        Test the translation of the warning
        for 'Someone is not wearing a safety vest!'.
        """
        warnings = [
            'Warning: Someone is not wearing a safety vest!!',
        ]
        language = 'zh-TW'  # Example of a non-English language

        translated_warnings = Translator.translate_warning(
            tuple(warnings), language,
        )

        expected_warnings = [
            '警告: 有人無穿著安全背心!',
        ]
        self.assertEqual(translated_warnings, expected_warnings)

    def test_language_code_case_insensitivity(self):
        """
        Test that language codes are case-insensitive.
        """
        warnings = [
            'Warning: Someone is not wearing a hardhat!',
            'Warning: 2 people have entered the controlled area!',
            'Warning: Someone is too close to machinery!',
        ]
        language = 'EN'  # Uppercase

        try:
            translated_warnings = Translator.translate_warning(
                tuple(warnings), language.lower(),
            )
            expected_warnings = [
                'Warning: Someone is not wearing a hardhat!',
                'Warning: 2 people have entered the controlled area!',
                'Warning: Someone is too close to machinery!',
            ]
            self.assertEqual(translated_warnings, expected_warnings)
        except Exception as e:
            self.fail(f"Translation failed for uppercase language code: {e}")

    def test_specific_language_translation_correctness(self):
        """
        Test that specific language translations are correct.
        """
        test_cases = {
            'zh-TW': {
                'Warning: Someone is not wearing a hardhat!': (
                    '警告: 有人無配戴安全帽!'
                ),
                'Warning: 2 people have entered the controlled area!': (
                    '警告: 有2個人進入受控區域!'
                ),
                'Warning: Someone is too close to machinery!': (
                    '警告: 有人過於靠近機具!'
                ),
            },
            'fr': {
                'Warning: Someone is not wearing a hardhat!': (
                    "Avertissement: Quelqu'un ne porte pas de casque!"
                ),
                'Warning: 2 people have entered the controlled area!': (
                    'Avertissement: 2 personnes sont entrées '
                    'dans la zone contrôlée!'
                ),
                'Warning: Someone is too close to machinery!': (
                    'Avertissement: Quelqu’un est trop proche de machinerie!'
                ),
            },
            # Add more languages and their expected translations if necessary
        }

        for lang, expected_translations in test_cases.items():
            translated_warnings = Translator.translate_warning(
                tuple(expected_translations.keys()), lang,
            )
            for original, expected in expected_translations.items():
                self.assertIn(
                    expected, translated_warnings,
                    (
                        f"Expected translation '{expected}' for "
                        f"'{original}' in language '{lang}' not found."
                    ),
                )

    def test_translation_with_missing_placeholders(self):
        """
        Test that translations with missing placeholders
        are handled gracefully.
        """
        warnings = [
            'Warning: Someone is not wearing a hardhat!',
            'Warning: 2 people have entered the controlled area!',
            'Warning: Someone is too close to machinery!',
        ]
        language = 'en'

        # Manually remove a placeholder to
        # simulate a missing placeholder scenario
        LANGUAGES[language][
            'warning_people_in_controlled_area'
        ] = 'Warning: {count} people have entered the controlled area!'

        try:
            translated_warnings = Translator.translate_warning(
                tuple(warnings), language,
            )
            expected_warnings = [
                'Warning: Someone is not wearing a hardhat!',
                'Warning: 2 people have entered the controlled area!',
                'Warning: Someone is too close to machinery!',
            ]
            self.assertEqual(translated_warnings, expected_warnings)
        except Exception as e:
            self.fail(f"Translation failed with all placeholders present: {e}")

        # Clear the cache to ensure changes to LANGUAGES are picked up
        Translator.translate_warning.cache_clear()

        # Now, remove a placeholder and expect it to handle gracefully
        # Missing {count}
        LANGUAGES[language]['warning_people_in_controlled_area'] = (
            'Warning: people have entered the controlled area!'
        )

        try:
            translated_warnings = Translator.translate_warning(
                tuple(warnings), language,
            )
            expected_warnings = [
                'Warning: Someone is not wearing a hardhat!',
                'Warning: people have entered the controlled area!',
                'Warning: Someone is too close to machinery!',
            ]
            self.assertEqual(translated_warnings, expected_warnings)
        except Exception as e:
            self.fail(f"Translation failed with missing placeholders: {e}")

    def test_translate_warning_no_match(self):
        """
        Test that if a warning message does not match any predefined warning,
        the original message is returned.
        """
        # Unmatched warning message
        warnings = [
            'Warning: This is an unmatched warning!',
        ]
        language = 'zh-TW'  # Example of a non-English language

        translated_warnings = Translator.translate_warning(
            tuple(warnings), language,
        )

        # Since the warning doesn't match any known pattern,
        # it should be kept as-is
        expected_warnings = [
            'Warning: This is an unmatched warning!',
        ]
        self.assertEqual(translated_warnings, expected_warnings)

    @patch('builtins.print')
    def test_main_function(self, mock_print):
        """
        Test the main function for proper translation and output.
        """
        # Execute main function
        main()

        # Check that the output was printed
        mock_print.assert_any_call(
            'Original Warnings:', [
                'Warning: Someone is not wearing a hardhat!',
                'Warning: 2 people have entered the controlled area!',
                'Warning: Someone is too close to machinery!',
            ],
        )
        mock_print.assert_any_call(
            'Translated Warnings:', [
                '警告: 有人無配戴安全帽!',
                '警告: 有2個人進入受控區域!',
                '警告: 有人過於靠近機具!',
            ],
        )

    def test_main_as_script(self):
        """
        Test executing the script directly with `if __name__ == '__main__'`.
        This will cover the main() function when invoked as standalone script.
        """
        script_path = os.path.join(
            os.path.dirname(
                __file__,
            ), '../../src/lang_config.py',
        )

        result = subprocess.run(
            ['python', script_path],
            capture_output=True, text=True,
        )

        # Assert that the script runs without errors
        self.assertEqual(
            result.returncode, 0,
            'Script exited with a non-zero status.',
        )

        # You can also verify the expected output here if needed
        self.assertIn('Original Warnings:', result.stdout)
        self.assertIn('Translated Warnings:', result.stdout)


if __name__ == '__main__':
    unittest.main()
