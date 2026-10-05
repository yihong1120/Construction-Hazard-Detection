from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from functools import lru_cache

# ---------------------------
# Synonyms mapping
# ---------------------------
SYNONYMS_MAP = {
    # Hardhat-related search terms.
    '帽': ['helmet', 'no_helmet', 'warning_no_hardhat'],
    '帽子': ['helmet', 'no_helmet', 'warning_no_hardhat'],
    'hat': ['helmet', 'no_helmet', 'warning_no_hardhat'],
    'helmet': ['helmet', 'no_helmet', 'warning_no_hardhat'],
    'casque': ['helmet', 'no_helmet', 'warning_no_hardhat'],
    '無安全帽': ['warning_no_hardhat'],
    '未戴帽': ['warning_no_hardhat'],
    '未戴安全帽': ['warning_no_hardhat'],
    '沒有安全帽': ['warning_no_hardhat'],
    '缺安全帽': ['warning_no_hardhat'],
    'no hardhat': ['warning_no_hardhat'],
    'not wearing hardhat': ['warning_no_hardhat'],
    # Safety-vest-related search terms.
    '背': ['safety_vest', 'no_safety_vest', 'vest'],
    '背心': ['safety_vest', 'no_safety_vest', 'vest'],
    'vest': ['safety_vest', 'no_safety_vest', 'vest'],
    'gilet': ['safety_vest', 'no_safety_vest', 'vest'],
    '無背': ['warning_no_safety_vest'],
    '無安全背心': ['warning_no_safety_vest'],
    '未穿背': ['warning_no_safety_vest'],
    '未穿安全背心': ['warning_no_safety_vest'],
    '沒有安全背心': ['warning_no_safety_vest'],
    '缺安全背心': ['warning_no_safety_vest'],
    'no vest': ['warning_no_safety_vest'],
    'no safety vest': ['warning_no_safety_vest'],
    # Person-related search terms.
    '人': ['person'],
    '人員': ['person'],
    'person': ['person'],
    'personne': ['person'],
    'personnes': ['person'],
    # Machinery-related search terms.
    '機具': ['machinery'],
    'machine': ['machinery'],
    'machinery': ['machinery'],
    'machinerie': ['machinery'],
    # Vehicle-related search terms.
    '車': ['vehicle'],
    '車輛': ['vehicle'],
    'vehicle': ['vehicle'],
    'voiture': ['vehicle'],
    # Safety-cone-related search terms.
    '錐': ['safety_cone'],
    '安全錐': ['safety_cone'],
    'cone': ['safety_cone'],
    # Face-mask-related search terms.
    '口罩': ['mask'],
    'mask': ['mask'],
    '無口罩': ['no_mask'],
    'no mask': ['no_mask'],
    'nomask': ['no_mask'],
    # Canonical detector/filter codes map to persisted warning keys.
    'no_helmet': ['warning_no_hardhat'],
    'no_safety_helmet': ['warning_no_hardhat'],
    'no_safety_vest': ['warning_no_safety_vest'],
    'no_mask': ['warning_no_mask'],
    # === Warning: people entering controlled area ===
    '受控': ['warning_people_in_controlled_area'],
    '受控區': ['warning_people_in_controlled_area'],
    '受控區域': ['warning_people_in_controlled_area'],
    '控制區': ['warning_people_in_controlled_area'],
    '控制區域': ['warning_people_in_controlled_area'],
    'controlled': ['warning_people_in_controlled_area'],
    'controlled area': ['warning_people_in_controlled_area'],
    'controlled zone': ['warning_people_in_controlled_area'],
    '進入受控': ['warning_people_in_controlled_area'],
    '進入控制區': ['warning_people_in_controlled_area'],
    '進入控制區域': ['warning_people_in_controlled_area'],
    # === Warning: close to machinery ===
    '靠近機具': ['warning_close_to_machinery'],
    '接近機具': ['warning_close_to_machinery'],
    '機具太近': ['warning_close_to_machinery'],
    'close machinery': ['warning_close_to_machinery'],
    'close to machinery': ['warning_close_to_machinery'],
    # === Warning: close to vehicle ===
    '靠近車輛': ['warning_close_to_vehicle'],
    '接近車輛': ['warning_close_to_vehicle'],
    '車輛太近': ['warning_close_to_vehicle'],
    'close vehicle': ['warning_close_to_vehicle'],
    'close to vehicle': ['warning_close_to_vehicle'],
    # === Warning: entering utility pole restricted area ===
    '電線桿控制區': ['detect_in_utility_pole_restricted_area'],
    '電桿控制區': ['detect_in_utility_pole_restricted_area'],
    'utility pole restricted area': ['detect_in_utility_pole_restricted_area'],
    'pole restricted area': ['detect_in_utility_pole_restricted_area'],
    'enter utility pole area': ['detect_in_utility_pole_restricted_area'],
    '進入電線桿控制區': ['detect_in_utility_pole_restricted_area'],
}

# Stop words are kept separate because Chinese tokenisation needs exact tokens.
ENGLISH_STOP_WORDS = {
    'the',
    'is',
    'at',
    'which',
    'on',
    'a',
    'an',
    'and',
    'or',
    'of',
    'to',
    'in',
}
CHINESE_STOP_WORDS = {'的', '了', '在', '是', '和'}


class SearchUtils:
    """Expand multilingual violation-search terms into detector labels."""

    def __init__(self) -> None:
        self.synonyms_map = SYNONYMS_MAP
        self.english_stop_words = ENGLISH_STOP_WORDS
        self.chinese_stop_words = CHINESE_STOP_WORDS
        self._synonym_index = self._build_synonym_index(
            self.synonyms_map.items(),
        )

    @staticmethod
    def _build_synonym_index(
        synonym_items: Iterable[tuple[str, list[str]]],
    ) -> dict[str, tuple[tuple[str, tuple[str, ...]], ...]]:
        """Group synonym keys by first character to reduce substring checks.

        Args:
            synonym_items: Normalised synonym key and expansion pairs.

        Returns:
            Immutable first-character index for query-time matching.
        """
        grouped: defaultdict[str, list[tuple[str, tuple[str, ...]]]] = (
            defaultdict(list)
        )
        for key, values in synonym_items:
            normalized_key = key.lower()
            if not normalized_key:
                continue
            grouped[normalized_key[0]].append(
                (normalized_key, tuple(values)),
            )
        return {
            first_char: tuple(items) for first_char, items in grouped.items()
        }

    @lru_cache(maxsize=512)
    def _tokenize_cached(self, user_input: str) -> tuple[str, ...]:
        """Keep literal Unicode words and canonical detector codes."""
        return tuple(
            token for token in re.findall(r'\w+', user_input)
            if token not in self.english_stop_words
            and token not in self.chinese_stop_words
        )

    def expand_synonyms(self, user_input: str) -> list[str]:
        """Expand known multilingual phrases while keeping literal searches.

        Match phrases before individual words, so ``no safety vest`` and
        continuous Chinese input do not require a statistical segmenter.
        English keys respect word boundaries to avoid matching ``hat`` inside
        ``what``. Longer phrases take precedence over overlapping short keys.
        """
        normalized = unicodedata.normalize('NFKC', user_input).casefold()
        return list(self._expand_synonyms_cached(normalized))

    @lru_cache(maxsize=512)
    def _expand_synonyms_cached(self, text: str) -> tuple[str, ...]:
        results = set(self._tokenize_cached(text))
        candidates: list[tuple[int, int, tuple[str, ...]]] = []
        for first_char in frozenset(text):
            for key, values in self._synonym_index.get(first_char, ()):
                pattern = re.escape(key)
                if key[0].isascii() and key[0].isalnum():
                    pattern = r'(?<!\w)' + pattern
                if key[-1].isascii() and key[-1].isalnum():
                    pattern += r'(?!\w)'
                candidates.extend(
                    (match.start(), match.end(), values)
                    for match in re.finditer(pattern, text)
                )
        covered: set[int] = set()
        for start, end, values in sorted(
            candidates, key=lambda item: (-(item[1] - item[0]), item[0]),
        ):
            positions = set(range(start, end))
            if positions.isdisjoint(covered):
                results.update(values)
                covered.update(positions)
        return tuple(sorted(results))
