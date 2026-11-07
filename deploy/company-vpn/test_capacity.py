# Capacity-test guards, measurement boundaries and resource cleanup.
from __future__ import annotations

import json
import tempfile
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import capacity_test as c
import configure_sso as s

import deploy as d


def _wiki_login_redirect_does_not_curl(
    testcase,
    urls,
    /,
    command,
    **kwargs,
):
    url = command[-1]
    urls.append(url)
    status = (
        '302' if url == 'https://wiki.changdar-server.mooo.com/' else '200'
    )
    testcase.assertNotIn('--insecure', command)
    testcase.assertNotIn('-k', command)
    return SimpleNamespace(returncode=0, stdout=status + ' 0.02')


def _cleanup_deletes_only_named_test_api(
    deleted,
    name,
    /,
    path,
    method='GET',
    **kwargs,
):
    if method == 'DELETE':
        deleted.append(path)
        return
    return {
        'policies': [
            {'id': 'test-policy', 'name': name},
            {'id': 'production', 'name': 'company-internet-b'},
        ],
        'groups': [
            {'id': 'test-client', 'name': name + '-client'},
            {'id': 'test-server', 'name': name + '-server'},
            {'id': 'all', 'name': 'All'},
        ],
    }[path]


class _FailedLoadStepDoesNotResult:
    returncode = 1
    stdout = '{"error":"server bitrate limit exceeded"}'


class CapacityTests(unittest.TestCase):
    def test_address_rejects_public_lan_and_ipv6(self):
        for value in (
            '60.249.248.158',
            '192.168.1.86',
            '::1',
            '100.93.178.78',
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                c.vpn_address(value)
        self.assertEqual(c.vpn_address('10.203.252.213'), '10.203.252.213')

    def test_health_guard_reacts_to_resource_exhaustion_and_website_failure(
        self,
    ):
        baseline = {'site': {'ok': True, 'seconds': 0.03}}
        row = {
            'cpu_percent': 24,
            'available_ram_mib': 14000,
            'websites': baseline,
        }
        self.assertFalse(c.guard_reasons(row, baseline))
        self.assertTrue(c.guard_reasons({**row, 'cpu_percent': 95}, baseline))
        self.assertTrue(
            c.guard_reasons({**row, 'available_ram_mib': 1000}, baseline),
        )
        for website in (
            {'ok': False, 'seconds': 0.1},
            {'ok': True, 'seconds': 3},
        ):
            self.assertTrue(
                c.guard_reasons(
                    {**row, 'websites': {'site': website}},
                    baseline,
                ),
            )

    def test_wiki_login_redirect_does_not_prevent_test_startup(self):
        urls = []
        curl = partial(_wiki_login_redirect_does_not_curl, self, urls)
        with patch.object(c.subprocess, 'run', side_effect=curl):
            values = c.health()
        self.assertTrue(all(value['ok'] for value in values.values()))
        self.assertIn('https://wiki.changdar-server.mooo.com/healthz', urls)

    def test_real_http_failure_and_tls_failure_still_fail_health_check(self):
        for code, status in ((0, '500'), (60, '000')):
            with self.subTest(code=code), patch.object(
                c.subprocess,
                'run',
                return_value=SimpleNamespace(
                    returncode=code,
                    stdout=status + ' 0.02',
                ),
            ):
                values = c.health()
            self.assertTrue(all(not value['ok'] for value in values.values()))

    def test_cleanup_deletes_only_named_test_resources_in_dependency_order(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / 'receipt.json'
            identifier = '123456abcdef'
            name = 'company-vpn-capacity-' + identifier
            receipt.write_text(
                json.dumps(
                    {
                        'identifier': identifier,
                        'policies': [name],
                        'groups': [name + '-client', name + '-server'],
                    },
                ),
            )
            deleted = []
            api = partial(_cleanup_deletes_only_named_test_api, deleted, name)
            with patch.object(d, 'api', side_effect=api), patch.object(
                s,
                'get_token',
                return_value='fixture',
            ), patch.object(c.subprocess, 'run'):
                c.cleanup(receipt)
            self.assertEqual(
                deleted,
                [
                    'policies/test-policy',
                    'groups/test-client',
                    'groups/test-server',
                ],
            )
            self.assertTrue(json.loads(receipt.read_text())['cleaned'])

    def test_status_report_omits_keys_and_other_employee_details(self):
        report = c.connection_summary(
            {
                'usesKernelInterface': True,
                'peers': {
                    'connected': 2,
                    'details': [
                        {
                            'netbirdIp': '10.203.252.213',
                            'status': 'Connected',
                            'connectionType': 'P2P',
                            'publicKey': 'private-fixture-key',
                        },
                        {'netbirdIp': '10.203.0.7', 'fqdn': 'other-employee'},
                    ],
                },
            },
            '10.203.252.213',
        )
        self.assertEqual(report['test_connection'][0]['type'], 'P2P')
        self.assertNotIn('private-fixture-key', json.dumps(report))
        self.assertNotIn('other-employee', json.dumps(report))

    def test_failed_load_step_does_not_report_success(self):
        Result = _FailedLoadStepDoesNotResult
        with patch.object(
            c.subprocess,
            'run',
            return_value=Result(),
        ), self.assertRaises(RuntimeError):
            c.run_case('iperf3', '10.203.0.1', '10.203.252.213', 8, True)

    def test_connection_failure_is_preserved_in_client_report(self):
        iperf_error = 'unable to connect to server: Connection refused'
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                server='10.203.0.1',
                client_ip='10.203.252.213',
                output=Path(tmp) / 'report.json',
            )
            responses = [
                SimpleNamespace(stdout='60.249.248.158', returncode=0),
                SimpleNamespace(
                    stdout=json.dumps({'error': iperf_error}),
                    stderr='',
                    returncode=1,
                ),
            ]
            with patch.object(
                c,
                'iperf',
                return_value='fixture-iperf3',
            ), patch.object(
                c.subprocess,
                'run',
                side_effect=responses,
            ), self.assertRaisesRegex(
                c.LoadStepError,
                'Connection refused',
            ):
                c.client(args)
            report = json.loads(args.output.read_text())
            self.assertFalse(report['completed'])
            self.assertEqual(report['steps'], [])
            self.assertEqual(report['failure']['error'], iperf_error)
            self.assertEqual(report['failure']['target_mbps'], 5)

    def test_empty_json_output_reports_stderr_instead_of_json_decode_error(
        self,
    ):
        response = SimpleNamespace(
            stdout='',
            stderr='iperf3: bind failed',
            returncode=1,
        )
        with patch.object(
            c.subprocess,
            'run',
            return_value=response,
        ), self.assertRaisesRegex(c.LoadStepError, 'bind failed'):
            c.run_case('iperf3', '10.203.0.1', '10.203.252.213', 1, True)

    def test_server_must_bind_before_it_is_ready(self):
        responses = [
            SimpleNamespace(stdout='', returncode=0),
            SimpleNamespace(stdout='failed', returncode=3),
        ]
        with patch.object(
            c.subprocess,
            'run',
            side_effect=responses,
        ), self.assertRaisesRegex(RuntimeError, 'failed to start'):
            c.wait_listener('fixture-unit', '10.203.0.1')

    def test_report_summary_preserves_failure_and_resource_peaks(self):
        rows = [
            {
                'cpu_percent': cpu,
                'busiest_core_percent': cpu + 10,
                'available_ram_mib': memory,
                'guard_reasons': reasons,
                'interfaces': {'wt0': {'rx_mbps': cpu, 'tx_mbps': cpu / 2}},
                'websites': {c.HOSTS[0]: {'ok': ok, 'seconds': seconds}},
            }
            for cpu, memory, reasons, ok, seconds in (
                (20, 14000, [], True, 0.01),
                (40, 13000, ['website failure'], False, 5.0),
            )
        ]
        value = c.summarize_report(
            {'kind': 'one peer', 'samples': rows, 'stop_reason': 'fixture'},
        )
        self.assertEqual(value['cpu_average_percent'], 30)
        self.assertEqual(value['cpu_peak_percent'], 40)
        self.assertEqual(value['minimum_available_ram_mib'], 13000)
        self.assertEqual(value['guard_trigger_samples'], 1)
        self.assertEqual(value['websites'][c.HOSTS[0]]['failed_checks'], 1)
        self.assertEqual(value['websites'][c.HOSTS[0]]['p95_ms'], 5000)
        self.assertEqual(value['tunnel_interval_peak_mbps']['rx'], 40)


if __name__ == '__main__':
    unittest.main()
