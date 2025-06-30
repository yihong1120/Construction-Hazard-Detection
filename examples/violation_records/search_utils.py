from __future__ import annotations

from ckip_transformers.nlp import CkipWordSegmenter

# ---------------------------
# Synonyms mapping
# ---------------------------
SYNONYMS_MAP = {
    # === Hardhat (安全帽) related ===
    '帽': ['hardhat', 'no_hardhat'],
    '帽子': ['hardhat', 'no_hardhat'],
    'hat': ['hardhat', 'no_hardhat'],
    'helmet': ['hardhat', 'no_hardhat'],
    'casque': ['hardhat', 'no_hardhat'],
    '無安全帽': ['warning_no_hardhat'],
    '未戴帽': ['warning_no_hardhat'],
    '未戴安全帽': ['warning_no_hardhat'],
    '沒有安全帽': ['warning_no_hardhat'],
    '缺安全帽': ['warning_no_hardhat'],
    'no hardhat': ['warning_no_hardhat'],
    'not wearing hardhat': ['warning_no_hardhat'],

    # === Vest (安全背心) related ===
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

    # === Person (人員) related ===
    '人': ['person'],
    '人員': ['person'],
    'person': ['person'],
    'personne': ['person'],
    'personnes': ['person'],

    # === Machinery (機具) related ===
    '機具': ['machinery'],
    'machine': ['machinery'],
    'machinery': ['machinery'],
    'machinerie': ['machinery'],

    # === Vehicle (車輛) related ===
    '車': ['vehicle'],
    '車輛': ['vehicle'],
    'vehicle': ['vehicle'],
    'voiture': ['vehicle'],

    # === Cone (安全錐) related ===
    '錐': ['cone'],
    '安全錐': ['cone'],
    'cone': ['cone'],

    # === Mask (口罩) related ===
    '口罩': ['mask'],
    'mask': ['mask'],
    '無口罩': ['no_mask'],
    'no mask': ['no_mask'],
    'nomask': ['no_mask'],

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

# ---------------------------
# Stop words definition (can be extended as needed)
# ---------------------------
ENGLISH_STOP_WORDS = {
    'the', 'is', 'at', 'which',
    'on', 'a', 'an', 'and', 'or', 'of', 'to', 'in',
}
CHINESE_STOP_WORDS = {'的', '了', '在', '是', '和'}


class SearchUtils:
    """
    A utility class for processing user input, including tokenization,
    synonym expansion, and building Elasticsearch queries.
    """

    def __init__(self, device: int = -1):
        """
        Initialise the CKIP word segmenter.

        Args:
            device (int): If using CPU, set device to -1.
            Otherwise, specify the GPU device number.
        """
        self.ws_driver = CkipWordSegmenter(device=device)
        self.synonyms_map = SYNONYMS_MAP
        self.english_stop_words = ENGLISH_STOP_WORDS
        self.chinese_stop_words = CHINESE_STOP_WORDS

    def tokenize(self, user_input: str) -> list[str]:
        """
        Tokenise the input text and remove stop words.

        Args:
            user_input (str): The raw text from the user.

        Returns:
            list[str]: A list of tokens after stop words are removed.
        """
        tokens = self.ws_driver([user_input])[0]
        filtered_tokens = [
            token.lower().strip()
            for token in tokens
            if token.lower().strip() not in self.english_stop_words
            and token.strip() not in self.chinese_stop_words
        ]
        return filtered_tokens

    def expand_synonyms(self, user_input: str) -> list[str]:
        """
        Expand the input text by replacing tokens with their synonyms.

        Args:
            user_input (str): The raw text from the user.

        Returns:
            list[str]: A list of unique tokens and their associated synonyms.
        """
        tokens = self.tokenize(user_input)
        results = set()
        for token in tokens:
            for key, vlist in self.synonyms_map.items():
                # If the key is a substring of the token, add the synonyms
                if key in token:
                    results.update(vlist)
            # Always add the original token
            results.add(token)
        return list(results)

    def build_elasticsearch_query(self, user_input: str) -> dict:
        """
        Build an Elasticsearch query using the expanded synonyms.

        Args:
            user_input (str): The raw text from the user.

        Returns:
            dict: A dictionary representing the Elasticsearch query.
        """
        keywords = self.expand_synonyms(user_input)
        query = {
            'query': {
                'bool': {
                    'should': (
                        [
                            {'wildcard': {'stream_name': f"*{kw}*"}}
                            for kw in keywords
                        ]
                        + [
                            {'wildcard': {'warnings_json': f"*{kw}*"}}
                            for kw in keywords
                        ]
                    ),
                },
            },
        }
        return query
