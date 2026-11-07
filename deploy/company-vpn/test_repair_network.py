from __future__ import annotations

import copy
import unittest

import repair_network as r

import deploy as d


class NetworkRepair(unittest.TestCase):
    def setUp(self):
        self.config = d.compose_config(
            {'bridge': '172.29.0.0/24', 'gateway': '172.29.0.1'},
            False,
        )
        self.config['services']['server']['ports'] = [
            '127.0.0.1:19081:80',
            r.OLD_IP + ':3478:3478/udp',
        ]

    def test_only_stun_address_changes(self):
        original = copy.deepcopy(self.config)
        result = r.repaired_config(self.config)
        self.assertEqual(self.config, original)
        self.assertIn(
            r.NEW_IP + ':3478:3478/udp',
            result['services']['server']['ports'],
        )
        result['services']['server']['ports'] = original['services']['server'][
            'ports'
        ]
        self.assertEqual(result, original)

    def test_repeat_repair_is_idempotent(self):
        result = r.repaired_config(self.config)
        self.assertEqual(r.repaired_config(result), result)

    def test_unexpected_image_mode_mounts_or_public_http_are_refused(self):
        for key, value in [
            ('image', 'other/image'),
            ('network_mode', 'host'),
            ('volumes', ['new-data:/var/lib/netbird']),
            ('ports', ['0.0.0.0:19081:80', r.OLD_IP + ':3478:3478/udp']),
        ]:
            config = copy.deepcopy(self.config)
            config['services']['server'][key] = value
            with self.subTest(field=key), self.assertRaises(RuntimeError):
                r.repaired_config(config)
