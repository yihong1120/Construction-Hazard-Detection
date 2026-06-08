from __future__ import annotations

from src.warning_types import Warnings

LANGUAGES = {
    'en-GB': {
        'warning_people_in_controlled_area':
            'Warning: {count} people have entered the controlled area!',
        'warning_no_hardhat':
            'Warning: {count} people are not wearing a hardhat!',
        'warning_no_safety_vest':
            'Warning: {count} people are not wearing a safety vest!',
        'warning_no_mask':
            'Warning: {count} people are not wearing a mask!',
        'warning_close_to_machinery':
            'Warning: {count} people are too close to machinery!',
        'warning_close_to_vehicle':
            'Warning: {count} people are too close to vehicles!',
        'warning_people_in_utility_pole_controlled_area':
            'Warning: {count} people have entered the utility pole '
            'restricted area!',
        'detect_machinery_close_to_pole':
            'Warning: {count} machineries are too close to the utility pole!',

        'no_warning': 'No warning',
        'machinery': 'machinery',
        'vehicle': 'vehicles',
        'helmet': 'helmet',
        'person': 'person',
        'no_helmet': 'no helmet',
        'vest': 'safety vest',
        'no_vest': 'no safety vest',
        'mask': 'mask',
        'no_mask': 'no mask',
        'cone': 'cone',

        'warning_notification': '[Warning Notification]',
    },

    'zh-TW': {
        'warning_people_in_controlled_area':
            '警告: 有{count}個人進入受控區域!',
        'warning_no_hardhat':
            '警告: 有{count}人未佩戴安全帽!',
        'warning_no_safety_vest':
            '警告: 有{count}人未穿著安全背心!',
        'warning_no_mask':
            '警告: 有{count}人未佩戴口罩!',
        'warning_close_to_machinery':
            '警告: 有{count}人過於靠近機具!',
        'warning_close_to_vehicle':
            '警告: 有{count}人過於靠近車輛!',
        'warning_people_in_utility_pole_controlled_area':
            '警告: 有{count}人進入電桿管制區!',
        'detect_machinery_close_to_pole':
            '警告: 有{count}個機具過於靠近電桿!',

        'no_warning': '無警告',
        'machinery': '機具',
        'vehicle': '車輛',
        'helmet': '安全帽',
        'person': '人員',
        'no_helmet': '無安全帽',
        'vest': '安全背心',
        'no_vest': '無安全背心',
        'mask': '口罩',
        'no_mask': '無口罩',
        'cone': '安全錐',

        'warning_notification': '[警示通知]',
    },

    'zh-CN': {
        'warning_people_in_controlled_area':
            '警告: 有{count}个人进入受控区域!',
        'warning_no_hardhat':
            '警告: 有{count}人未佩戴安全帽!',
        'warning_no_safety_vest':
            '警告: 有{count}人未穿着安全背心!',
        'warning_no_mask':
            '警告: 有{count}人未佩戴口罩!',
        'warning_close_to_machinery':
            '警告: 有{count}人过于靠近机械!',
        'warning_close_to_vehicle':
            '警告: 有{count}人过于靠近车辆!',
        'warning_people_in_utility_pole_controlled_area':
            '警告: 有{count}人进入电桿管制区!',
        'detect_machinery_close_to_pole':
            '警告: 有{count}个机械过于靠近电桿!',

        'no_warning': '无警告',
        'machinery': '机械',
        'vehicle': '车辆',
        'helmet': '安全帽',
        'person': '人员',
        'no_helmet': '无安全帽',
        'vest': '安全背心',
        'no_vest': '无安全背心',
        'mask': '口罩',
        'no_mask': '无口罩',
        'cone': '安全锥',

        'warning_notification': '[警示通知]',
    },

    'fr-FR': {
        'warning_people_in_controlled_area':
            'Avertissement: {count} personnes sont entrées dans la '
            'zone contrôlée!',
        'warning_no_hardhat':
            'Avertissement: {count} personnes ne portent pas de casque!',
        'warning_no_safety_vest':
            'Avertissement: {count} personnes ne portent pas de '
            'gilet de sécurité!',
        'warning_no_mask':
            'Avertissement: {count} personnes ne portent pas de masque!',
        'warning_close_to_machinery':
            'Avertissement: Il y a {count} personnes trop proches de '
            'la machinerie!',
        'warning_close_to_vehicle':
            'Avertissement: Il y a {count} personnes trop proches de '
            'véhicules!',
        'warning_people_in_utility_pole_controlled_area':
            'Avertissement: {count} personnes sont entrées dans la zone '
            'de poteau électrique!',
        'detect_machinery_close_to_pole':
            'Avertissement: {count} machines sont trop proches du poteau '
            'électrique!',

        'no_warning': "Pas d'avertissement",
        'machinery': 'machinerie',
        'vehicle': 'véhicules',
        'helmet': 'casque',
        'person': 'personne',
        'no_helmet': 'pas de casque',
        'vest': 'gilet de sécurité',
        'no_vest': 'pas de gilet de sécurité',
        'mask': 'masque',
        'no_mask': 'pas de masque',
        'cone': 'cône',

        'warning_notification': '[Avertissement Notification]',
    },

    'vi-VN': {
        'warning_people_in_controlled_area':
            'Cảnh báo: Có {count} người đã vào khu vực kiểm soát!',
        'warning_no_hardhat':
            'Cảnh báo: Có {count} người không đội mũ bảo hộ!',
        'warning_no_safety_vest':
            'Cảnh báo: Có {count} người không mặc áo gi-lê an toàn!',
        'warning_no_mask':
            'Cảnh báo: Có {count} người không đeo khẩu trang!',
        'warning_close_to_machinery':
            'Cảnh báo: Có {count} người quá gần máy móc!',
        'warning_close_to_vehicle':
            'Cảnh báo: Có {count} người quá gần phương tiện!',
        'warning_people_in_utility_pole_controlled_area':
            'Cảnh báo: Có {count} người đã vào khu vực cột điện!',
        'detect_machinery_close_to_pole':
            'Cảnh báo: Có {count} máy móc quá gần cột điện!',

        'no_warning': 'Không có cảnh báo',
        'machinery': 'máy móc',
        'vehicle': 'phương tiện',
        'helmet': 'mũ bảo hộ',
        'person': 'người',
        'no_helmet': 'không mũ bảo hộ',
        'vest': 'áo gi-lê an toàn',
        'no_vest': 'không áo gi-lê an toàn',
        'mask': 'khẩu trang',
        'no_mask': 'không khẩu trang',
        'cone': 'cọc tiêu',

        'warning_notification': '[Thông báo cảnh báo]',
    },

    'id-ID': {
        'warning_people_in_controlled_area':
            'Peringatan: Ada {count} orang memasuki area terkontrol!',
        'warning_no_hardhat':
            'Peringatan: Ada {count} orang tidak mengenakan helm!',
        'warning_no_safety_vest':
            'Peringatan: Ada {count} orang tidak mengenakan '
            'rompi keselamatan!',
        'warning_no_mask':
            'Peringatan: Ada {count} orang tidak mengenakan masker!',
        'warning_close_to_machinery':
            'Peringatan: Ada {count} orang terlalu dekat dengan mesin!',
        'warning_close_to_vehicle':
            'Peringatan: Ada {count} orang terlalu dekat dengan kendaraan!',
        'warning_people_in_utility_pole_controlled_area':
            'Peringatan: Ada {count} orang memasuki area tiang listrik!',
        'detect_machinery_close_to_pole':
            'Peringatan: Ada {count} mesin terlalu dekat dengan '
            'tiang listrik!',

        'no_warning': 'Tidak ada peringatan',
        'machinery': 'mesin',
        'vehicle': 'kendaraan',
        'helmet': 'helm',
        'person': 'orang',
        'no_helmet': 'tanpa helm',
        'vest': 'rompi keselamatan',
        'no_vest': 'tanpa rompi keselamatan',
        'mask': 'masker',
        'no_mask': 'tanpa masker',
        'cone': 'kerucut pengaman',

        'warning_notification': '[Pemberitahuan Peringatan]',
    },

    'th-TH': {
        'warning_people_in_controlled_area':
            'คำเตือน: มี {count} คนเข้ามาในพื้นที่ควบคุม!',
        'warning_no_hardhat':
            'คำเตือน: มี {count} คนไม่สวมหมวกนิรภัย!',
        'warning_no_safety_vest':
            'คำเตือน: มี {count} คนไม่สวมเสื้อกั๊กนิรภัย!',
        'warning_no_mask':
            'คำเตือน: มี {count} คนไม่สวมหน้ากากอนามัย!',
        'warning_close_to_machinery':
            'คำเตือน: มี {count} คนอยู่ใกล้เครื่องจักรมากเกินไป!',
        'warning_close_to_vehicle':
            'คำเตือน: มี {count} คนอยู่ใกล้ยานพาหนะมากเกินไป!',
        'warning_people_in_utility_pole_controlled_area':
            'คำเตือน: มี {count} คนได้เข้ามาในพื้นที่เสาไฟฟ้า!',
        'detect_machinery_close_to_pole':
            'คำเตือน: มี {count} เครื่องจักรอยู่ใกล้เสาไฟฟ้ามากเกินไป!',

        'no_warning': 'ไม่มีคำเตือน',
        'machinery': 'เครื่องจักร',
        'vehicle': 'ยานพาหนะ',
        'helmet': 'หมวกนิรภัย',
        'person': 'บุคคล',
        'no_helmet': 'ไม่สวมหมวกนิรภัย',
        'vest': 'เสื้อกั๊กนิรภัย',
        'no_vest': 'ไม่สวมเสื้อกั๊กนิรภัย',
        'mask': 'หน้ากากอนามัย',
        'no_mask': 'ไม่สวมหน้ากากอนามัย',
        'cone': 'กรวยนิรภัย',

        'warning_notification': '[การแจ้งเตือนความปลอดภัย]',
    },

    'ja-JP': {
        'warning_people_in_controlled_area':
            '警告: {count}人が管理区域に入りました!',
        'warning_no_hardhat':
            '警告: {count}人がヘルメットを着用していません!',
        'warning_no_safety_vest':
            '警告: {count}人が安全ベストを着用していません!',
        'warning_no_mask':
            '警告: {count}人がマスクを着用していません!',
        'warning_close_to_machinery':
            '警告: {count}人が重機に近づきすぎています!',
        'warning_close_to_vehicle':
            '警告: {count}人が車両に近づきすぎています!',
        'warning_people_in_utility_pole_controlled_area':
            '警告: {count}人が電柱制限区域に入りました!',
        'detect_machinery_close_to_pole':
            '警告: {count}台の重機が電柱に近づきすぎています!',

        'no_warning': '警告なし',
        'machinery': '重機',
        'vehicle': '車両',
        'helmet': 'ヘルメット',
        'person': '作業員',
        'no_helmet': 'ヘルメットなし',
        'vest': '安全ベスト',
        'no_vest': '安全ベストなし',
        'mask': 'マスク',
        'no_mask': 'マスクなし',
        'cone': 'カラーコーン',

        'warning_notification': '[警告通知]',
    },
}


LANGUAGE_ALIASES: dict[str, str] = {
    'en': 'en-GB',
    'en-gb': 'en-GB',
    'en-us': 'en-GB',
    'zh': 'zh-TW',
    'zh-tw': 'zh-TW',
    'zh-hant': 'zh-TW',
    'zh-hk': 'zh-TW',
    'zh-mo': 'zh-TW',
    'zh-cn': 'zh-CN',
    'zh-hans': 'zh-CN',
    'fr': 'fr-FR',
    'fr-fr': 'fr-FR',
    'vi': 'vi-VN',
    'vi-vn': 'vi-VN',
    'id': 'id-ID',
    'id-id': 'id-ID',
    'th': 'th-TH',
    'th-th': 'th-TH',
    'ja': 'ja-JP',
    'jp': 'ja-JP',
    'ja-jp': 'ja-JP',
}


def normalize_language(language: str | None) -> str | None:
    """Return the supported notification language code for a device value.

    Args:
        language: Raw language value from a device or client request.

    Returns:
        Normalised BCP 47 language code, or None when unsupported.
    """
    if not language:
        return None
    candidate = language.strip().replace('_', '-')
    if candidate in LANGUAGES:
        return candidate
    return LANGUAGE_ALIASES.get(candidate.lower())


class Translator:
    """Translate warning payloads into supported notification languages."""

    @staticmethod
    def translate_from_dict(
        body_dict: Warnings,
        language: str,
    ) -> list[str]:
        """Translate warning payload entries into the requested language.

        Example:
            ```python
            Translator.translate_from_dict(
                {'warning_close_to_vehicle': {'count': 3}},
                'zh-TW',
            )
            ```

        Args:
            body_dict: Warning-key mapping whose values contain template
                placeholders such as `count`.
            language: Requested locale code, such as `en-GB` or `zh-TW`.

        Returns:
            Translated warning messages. Unsupported languages return an empty
            list.
        """
        normalized_language = normalize_language(language)
        if normalized_language is None:
            return []

        translations: list[str] = []
        lang_map = LANGUAGES[normalized_language]

        for key, placeholders in body_dict.items():
            template = lang_map.get(key)
            if not template:
                # If the key has no template, use the key itself as the message
                translations.append(key)
                continue

            msg = template
            # Replace placeholders like {count} in the template
            for ph_key, ph_value in placeholders.items():
                msg = msg.replace(f"{{{ph_key}}}", str(ph_value))

            translations.append(msg)

        return translations


def main() -> None:
    """Print a small translation demonstration for local manual testing."""
    body_dict = {
        'warning_no_hardhat': {'count': 2},
        'warning_no_safety_vest': {'count': 1},
        'warning_people_in_utility_pole_controlled_area': {'count': 3},
    }
    language = 'en-GB'

    translations = Translator.translate_from_dict(body_dict, language)
    print(translations)
    # Possible output:
    # [
    #   "Warning: 2 people are not wearing a hardhat!",
    #   "Warning: 1 people are not wearing a safety vest!",
    #   "Warning: 3 people have entered the utility pole restricted area!"
    # ]


if __name__ == '__main__':
    main()
