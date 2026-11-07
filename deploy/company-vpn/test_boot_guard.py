# Startup recovery boundaries and healthy-state behavior.
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import patch

import boot_guard as b

import deploy as d


def _healthy_check_never_restarts_or_command(
    calls,
    /,
    *args,
    **kwargs,
):
    calls.append(args)
    if args[:3] == ('docker', 'volume', 'inspect'):
        return json.dumps(
            [{'Labels': {'com.docker.compose.project': 'company-vpn'}}],
        )
    if args[:2] == ('docker', 'ps'):
        return 'dashboard-id\n'
    raise AssertionError('Unexpected mutation: ' + repr(args))


def managed_container():
    return {
        'State': {'Running': True},
        'NetworkSettings': {
            'Networks': {'company-vpn_default': {}},
            'Ports': {
                '80/tcp': [{'HostIp': '127.0.0.1', 'HostPort': '19081'}],
                '3478/udp': [{'HostIp': '192.168.1.88', 'HostPort': '3478'}],
            },
        },
        'Mounts': [
            {
                'Destination': '/var/lib/netbird',
                'Type': 'volume',
                'Name': 'company-vpn_data',
            },
        ],
    }


class BootGuard(unittest.TestCase):
    def setUp(self):
        self.config = d.compose_config(
            {'bridge': '172.29.0.0/24', 'gateway': '172.29.0.1'},
            False,
        )
        self.config['services']['server']['ports'] = [
            '127.0.0.1:19081:80',
            '192.168.1.86:3478:3478/udp',
        ]

    def test_lan_change_only_updates_stun_binding(self):
        original = copy.deepcopy(self.config)
        result = b.config_for_address(self.config, '192.168.1.88')
        self.assertEqual(self.config, original)
        self.assertIn(
            '192.168.1.88:3478:3478/udp',
            result['services']['server']['ports'],
        )
        self.assertEqual(b.config_for_address(result, '192.168.1.88'), result)
        result['services']['server']['ports'] = original['services']['server'][
            'ports'
        ]
        self.assertEqual(result, original)

    def test_does_not_follow_b_to_an_unapproved_network(self):
        for value in (
            '192.168.2.88',
            '100.93.178.78',
            '0.0.0.0',
            '192.168.1.1',
            '192.168.1.255',
            '::1',
        ):
            with self.subTest(address=value), self.assertRaises(RuntimeError):
                b.config_for_address(self.config, value)

    def test_changed_volume_definition_cannot_initialize_another_database(
        self,
    ):
        for definition in ({'external': True}, {'name': 'another-volume'}):
            config = copy.deepcopy(self.config)
            config['volumes']['data'] = definition
            with self.assertRaises(RuntimeError):
                b.config_for_address(config, '192.168.1.88')

    def test_running_container_with_detached_network_is_unhealthy(self):
        value = managed_container()
        self.assertTrue(b.network_ok(value, '192.168.1.88'))
        value['NetworkSettings'] = {'Networks': {}, 'Ports': {}}
        self.assertFalse(b.network_ok(value, '192.168.1.88'))

    def test_cooldown_limits_repairs_but_allows_later_recovery(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            b,
            'RUNTIME',
            Path(tmp),
        ):
            with patch.object(b.time, 'monotonic', return_value=1000):
                b.repair_slot()
            with patch.object(
                b.time,
                'monotonic',
                return_value=1299,
            ), self.assertRaises(RuntimeError):
                b.repair_slot()
            with patch.object(b.time, 'monotonic', return_value=1300):
                b.repair_slot()

    def test_healthy_check_never_restarts_or_recreates_containers(self):
        config = b.config_for_address(self.config, '192.168.1.88')
        calls = []
        command = partial(_healthy_check_never_restarts_or_command, calls)
        with patch.object(
            b,
            'load',
            side_effect=[{'data_volume': 'company-vpn_data'}, config],
        ), patch.object(
            b,
            'lan_address',
            return_value='192.168.1.88',
        ), patch.object(
            b,
            'container',
            return_value=managed_container(),
        ), patch.object(
            b,
            'api_ready',
            return_value=True,
        ), patch.object(
            b,
            'call',
            side_effect=command,
        ):
            b.ensure()
        self.assertTrue(calls)

    def test_missing_recorded_volume_stops_before_recreation(self):
        config = b.config_for_address(self.config, '192.168.1.88')
        with patch.object(
            b,
            'load',
            side_effect=[{'data_volume': 'company-vpn_data'}, config],
        ), patch.object(
            b,
            'lan_address',
            return_value='192.168.1.88',
        ), patch.object(
            b,
            'container',
            return_value=managed_container(),
        ), patch.object(
            b,
            'call',
            side_effect=RuntimeError('volume missing'),
        ) as command:
            with self.assertRaises(RuntimeError):
                b.ensure()
            self.assertEqual(
                command.call_args.args,
                ('docker', 'volume', 'inspect', 'company-vpn_data'),
            )
