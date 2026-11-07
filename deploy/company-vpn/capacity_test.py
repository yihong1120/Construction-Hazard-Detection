# Bounded VPN throughput test: temporary B endpoint, external client and health
# guard.
#
# Parallel TCP streams are NOT distinct VPN peers. This tests the employee-to-B
# VPN leg, not an Internet origin behind B or an 80-device workload.
from __future__ import annotations

import argparse
import concurrent.futures
import ipaddress
import json
import os
import re
import shutil
import signal
import socket
import statistics
import subprocess
import time
import uuid
from pathlib import Path

PORT = 18520
HOSTS = (
    'changdar-server.mooo.com',
    'ai.changdar-server.mooo.com',
    'wiki.changdar-server.mooo.com',
)
HEALTH_PATHS = {
    host: '/healthz' if host == 'wiki.changdar-server.mooo.com' else '/'
    for host in HOSTS
}
VPN = ipaddress.ip_network('10.203.0.0/16')


class LoadStepError(RuntimeError):
    def __init__(self, details):
        self.details = details
        super().__init__('VPN load step failed: ' + details['error'])


def iperf():
    binary = shutil.which('iperf3')
    if not binary:
        raise RuntimeError(
            (
                'Install iperf3 first: B: sudo apt-get install '
                'iperf3; Mac: brew install iperf3'
            ),
        )
    result = subprocess.run(
        [binary, '--version'],
        capture_output=True,
        text=True,
        check=True,
    )
    version = re.search(r'iperf\s+(\d+)\.(\d+)', result.stdout)
    if not version or tuple(map(int, version.groups())) < (3, 16):
        raise RuntimeError('iperf3 3.16 or newer is required on both ends.')
    return binary


def install_tools():
    import deploy as d

    addresses = json.loads(d.run('ip', '-j', '-4', 'address', 'show'))
    if not any(a['local'] == d.LAN for i in addresses for a in i['addr_info']):
        raise RuntimeError('Run install-tools on B (192.168.1.86).')
    if shutil.which('iperf3'):
        iperf()
        print(
            (
                'Existing iperf3 version is suitable; no package or '
                'service was changed.'
            ),
        )
        return
    # Never install an unattended public iperf daemon for this temporary test.
    subprocess.run(
        ['debconf-set-selections'],
        input='iperf3 iperf3/start_daemon boolean false\n',
        text=True,
        check=True,
    )
    subprocess.run(
        ['apt-get', 'install', '-y', '--no-install-recommends', 'iperf3'],
        env={**os.environ, 'DEBIAN_FRONTEND': 'noninteractive'},
        check=True,
    )
    unit = subprocess.run(
        ['systemctl', 'show', 'iperf3.service', '-p', 'LoadState', '--value'],
        capture_output=True,
        text=True,
    )
    if unit.stdout.strip() == 'loaded':
        d.run('systemctl', 'disable', '--now', 'iperf3.service')
    iperf()
    print(
        (
            'iperf3 ready; the package background service is '
            'disabled. No test port was opened.'
        ),
    )


def vpn_address(value):
    address = ipaddress.ip_address(value)
    if address not in VPN:
        raise ValueError(
            'Use a company VPN IPv4 address, not a LAN or public address.',
        )
    return str(address)


def snapshot():
    cpu = {
        line.split()[0]: list(map(int, line.split()[1:9]))
        for line in Path('/proc/stat').read_text().splitlines()
        if line.startswith('cpu')
    }
    mem = {
        line.split(':')[0]: int(line.split()[1])
        for line in Path('/proc/meminfo').read_text().splitlines()
        if line.startswith(('MemTotal:', 'MemAvailable:'))
    }
    net = {}
    for interface in Path('/sys/class/net').iterdir():
        if (
            interface.name == 'wt0'
            or (interface / 'operstate').read_text().strip() == 'up'
        ):
            net[interface.name] = {
                key: int((interface / 'statistics' / key).read_text())
                for key in ('rx_bytes', 'tx_bytes', 'rx_dropped', 'tx_dropped')
            }
    return {'time': time.monotonic(), 'cpu': cpu, 'memory': mem, 'net': net}


def delta(before, after):
    seconds = after['time'] - before['time']
    busy = {}
    for name, current in after['cpu'].items():
        if name in before['cpu']:
            values = [b - a for a, b in zip(before['cpu'][name], current)]
            total = sum(values)
            busy[name] = round(
                100 * (total - values[3] - values[4]) / max(1, total),
                1,
            )
    interfaces = {}
    for name, current in after['net'].items():
        if name in before['net']:
            old = before['net'][name]
            interfaces[name] = {
                direction
                + '_mbps': round(
                    (current[direction + '_bytes'] - old[direction + '_bytes'])
                    * 8
                    / seconds
                    / 1e6,
                    3,
                )
                for direction in ('rx', 'tx')
            }
            interfaces[name].update(
                {
                    key: current[key] - old[key]
                    for key in ('rx_dropped', 'tx_dropped')
                },
            )
    return {
        'at': time.time(),
        'cpu_percent': busy['cpu'],
        'busiest_core_percent': max(v for k, v in busy.items() if k != 'cpu'),
        'available_ram_mib': round(after['memory']['MemAvailable'] / 1024),
        'interfaces': interfaces,
    }


def health():
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        return dict(pool.map(_gateway_health, HOSTS))


def guard_reasons(row, baseline):
    reasons = []
    if row['cpu_percent'] >= 90:
        reasons.append('CPU >= 90%')
    if row['available_ram_mib'] < 2048:
        reasons.append('available RAM < 2 GiB')
    for host, value in row['websites'].items():
        if not value['ok'] or value['seconds'] > max(
            2.0,
            baseline[host]['seconds'] * 4,
        ):
            reasons.append('production website health/latency: ' + host)
    return reasons


def policy_payload(name, source, target):
    return {
        'name': name,
        'description': 'Temporary bounded VPN capacity test',
        'enabled': True,
        'rules': [
            {
                'name': 'Test device to B benchmark port only',
                'enabled': True,
                'action': 'accept',
                'bidirectional': False,
                'protocol': 'tcp',
                'ports': [str(PORT)],
                'sources': [source],
                'destinations': [target],
            },
        ],
    }


def connection_summary(status, client_ip):
    details = status.get('peers', {}).get('details', []) or []
    selected = [
        peer
        for peer in details
        if str(peer.get('netbirdIp', '')).split('/')[0] == client_ip
    ]
    return {
        'kernel_wireguard': status.get('usesKernelInterface'),
        'connected_peer_count': status.get('peers', {}).get('connected'),
        'test_connection': [
            {
                'status': peer.get('status'),
                'type': peer.get('connectionType'),
                'latency': peer.get('latency'),
            }
            for peer in selected
        ],
    }


def summarize_report(report):
    samples = report.get('samples', [])
    summary = {
        'kind': report['kind'],
        'sample_count': len(samples),
        'stop_reason': report.get('stop_reason'),
        'connection_before': report.get('connection_before'),
        'connection_after': report.get('connection_after'),
    }
    if not samples:
        return summary
    summary.update(
        cpu_average_percent=round(
            statistics.mean(s['cpu_percent'] for s in samples),
            1,
        ),
        cpu_peak_percent=max(s['cpu_percent'] for s in samples),
        busiest_core_peak_percent=max(
            s['busiest_core_percent'] for s in samples
        ),
        minimum_available_ram_mib=min(s['available_ram_mib'] for s in samples),
        guard_trigger_samples=sum(
            bool(s.get('guard_reasons')) for s in samples
        ),
    )
    summary['websites'] = {}
    for host in HOSTS:
        values = [
            s['websites'][host]
            for s in samples
            if host in s.get('websites', {})
        ]
        if values:
            times = sorted(value['seconds'] for value in values)
            # Nearest-rank p95, meaningful also for small collections.
            rank = (95 * len(times) + 99) // 100
            summary['websites'][host] = {
                'checks': len(values),
                'failed_checks': sum(not value['ok'] for value in values),
                'p95_ms': round(times[rank - 1] * 1000, 1),
                'max_ms': round(times[-1] * 1000, 1),
            }
    tunnel = [
        s['interfaces']['wt0']
        for s in samples
        if 'wt0' in s.get('interfaces', {})
    ]
    if tunnel:
        summary['tunnel_interval_peak_mbps'] = {
            direction: max(s[direction + '_mbps'] for s in tunnel)
            for direction in ('rx', 'tx')
        }
    return summary


def report_latest():
    import deploy as d
    import enroll_b as e

    paths = [
        path
        for path in d.ROOT.glob('company-vpn-capacity-*-report.json')
        if re.fullmatch(
            r'company-vpn-capacity-[0-9a-f]{12}-report\.json',
            path.name,
        )
    ]
    if not paths:
        raise RuntimeError('No B capacity report is available yet.')
    path = max(paths, key=_mtime_ns)
    report = json.loads(path.read_text())
    summary = summarize_report(report)
    summary['report_file'] = str(path)
    summary['connection_now'] = connection_summary(
        e.status(),
        report['client_vpn_ip'],
    )
    receipt = path.with_name(path.name.replace('-report.json', '.json'))
    if receipt.exists():
        summary['temporary_access_cleaned'] = json.loads(
            receipt.read_text(),
        ).get('cleaned', False)
    print(json.dumps(summary, indent=2))


def cleanup(state_path):
    import deploy as d
    from configure_sso import get_token

    state = json.loads(state_path.read_text())
    # State is written in a root-only directory before any access is granted.
    identifier = state['identifier']
    if not re.fullmatch(r'[0-9a-f]{12}', identifier):
        raise RuntimeError('Invalid test receipt.')
    unit = 'company-vpn-capacity-' + identifier
    subprocess.run(
        ['systemctl', 'stop', unit + '.service'],
        capture_output=True,
    )
    token = get_token()
    # Match the unique name too, including a POST that succeeded before its ID
    # could be recorded. Delete only resources from this individual test.
    for kind in ('policies', 'groups'):
        expected = set(state.get(kind, []))
        for obj in d.api(kind, token=token):
            if obj.get('name') in expected:
                d.api(kind + '/' + obj['id'], 'DELETE', token=token)
    state['cleaned'] = True
    d.save(state_path, state)


def serve(args):
    import deploy as d
    import enroll_b as e
    import configure_exit as x
    from configure_sso import get_token

    binary = iperf()
    help_text = subprocess.run(
        [binary, '--help'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if '--server-bitrate-limit' not in help_text:
        raise RuntimeError(
            'This iperf3 build lacks the server bitrate safety limit.',
        )
    token = get_token()
    if any(
        obj.get('name', '').startswith('company-vpn-capacity-')
        for obj in d.api('policies', token=token)
    ):
        raise RuntimeError(
            (
                'A previous capacity test is still active; let its '
                'cleanup finish first.'
            ),
        )
    state = json.loads(x.STATE.read_text()) if x.STATE.exists() else {}
    if not state.get('complete'):
        raise RuntimeError(
            (
                'Configure the company exit first: sudo python3 '
                '~/company-vpn/configure_exit.py apply --client-ip '
            )
            + args.client_ip,
        )
    x.verify(token, state)
    employee = x.active_client(token, args.client_ip)
    if employee['id'] not in state['client_peer_ids']:
        raise RuntimeError(
            'This test device is not in the configured full-tunnel group.',
        )
    status = e.status()
    address = vpn_address(str(ipaddress.ip_interface(status['netbirdIp']).ip))
    with socket.socket() as sock:
        sock.bind((address, PORT))
    baseline = health()
    if not all(item['ok'] for item in baseline.values()):
        failures = [
            host
            + value['path']
            + ': HTTP '
            + str(value['status'])
            + ', curl exit '
            + str(value['curl_exit_code'])
            for host, value in baseline.items()
            if not value['ok']
        ]
        raise RuntimeError(
            (
                'Production health checks failed before testing; no '
                'endpoint created. '
            )
            + '; '.join(failures),
        )
    identifier = uuid.uuid4().hex[:12]
    name = 'company-vpn-capacity-' + identifier
    receipt = d.ROOT / (name + '.json')
    report_path = d.ROOT / (name + '-report.json')
    record = {
        'identifier': identifier,
        'groups': [name + '-client', name + '-server'],
        'policies': [name],
        'report': str(report_path),
        'cleaned': False,
    }
    d.save(receipt, record)
    report = {
        'kind': 'single-peer VPN-leg test; not a measured user limit',
        'server_vpn_ip': address,
        'client_vpn_ip': args.client_ip,
        'baseline': baseline,
        'samples': [],
        'stop_reason': None,
        'connection_before': connection_summary(status, args.client_ip),
    }
    # A separate timer survives terminal disconnects and kills of the monitor.
    # The server itself also has RuntimeMaxSec, so the listener cannot persist.
    d.run(
        'systemd-run',
        '--unit=' + name + '-cleanup',
        '--on-active=660s',
        '/usr/bin/python3',
        str(Path(__file__).resolve()),
        'cleanup',
        '--state',
        str(receipt),
    )
    try:
        source = d.api(
            'groups',
            'POST',
            {'name': record['groups'][0], 'peers': [employee['id']]},
            token,
        )
        target = d.api(
            'groups',
            'POST',
            {'name': record['groups'][1], 'peers': [state['server_peer_id']]},
            token,
        )
        d.api(
            'policies',
            'POST',
            policy_payload(name, source['id'], target['id']),
            token,
        )
        d.run(
            'systemd-run',
            '--unit=' + name,
            '--collect',
            '--property=RuntimeMaxSec=600',
            '--property=User=nobody',
            '--property=Group=nogroup',
            '--property=NoNewPrivileges=yes',
            '--property=PrivateTmp=yes',
            '--property=ProtectSystem=strict',
            '--property=ProtectHome=yes',
            '--property=MemoryMax=256M',
            '--property=CPUQuota=200%',
            binary,
            '-s',
            '-B',
            address,
            '-p',
            str(PORT),
            '--server-bitrate-limit',
            '50M/5',
        )
        wait_listener(name, address)
        print(
            ('Temporary test endpoint: ') + address + (':') + str(PORT),
            flush=True,
        )
        print(
            ('On the external Mac, with company exit selected, ' 'run:'),
            flush=True,
        )
        print(
            'python3 ~/Downloads/company-vpn-capacity-test.py client --server '
            + address
            + ' --client-ip '
            + args.client_ip,
            flush=True,
        )
        print(
            (
                'Endpoint expires after 10 minutes. Ctrl+C also '
                'removes the test access policy.'
            ),
            flush=True,
        )
        print('Report: ' + str(report_path), flush=True)
        previous = snapshot()
        failures = 0
        deadline = time.monotonic() + 590
        while time.monotonic() < deadline:
            time.sleep(5)
            current = snapshot()
            row = delta(previous, current)
            previous = current
            row['websites'] = health()
            row['guard_reasons'] = guard_reasons(row, baseline)
            report['samples'].append(row)
            failures = failures + 1 if row['guard_reasons'] else 0
            if failures >= 2:
                report['stop_reason'] = row['guard_reasons']
                print(
                    'Stopped load test: ' + '; '.join(row['guard_reasons']),
                    flush=True,
                )
                break
            active = subprocess.run(
                ['systemctl', 'is-active', name],
                capture_output=True,
                text=True,
            )
            if active.returncode:
                report['stop_reason'] = (
                    'Test server stopped; inspect its local systemd journal.'
                )
                break
            d.save(report_path, report)
    except KeyboardInterrupt:
        report['stop_reason'] = 'Operator stopped the test.'
    except Exception as error:
        report['stop_reason'] = str(error)[:500]
        raise
    finally:
        subprocess.run(
            ['systemctl', 'stop', name + '.service'],
            capture_output=True,
        )
        try:
            report['connection_after'] = connection_summary(
                e.status(),
                args.client_ip,
            )
            d.save(report_path, report)
        finally:
            cleanup(receipt)
        subprocess.run(
            [('systemctl'), ('stop'), name + ('-cleanup.timer')],
            capture_output=True,
        )
        print(
            'Temporary listener and access policy removed; report: '
            + str(report_path),
            flush=True,
        )


def wait_listener(unit, address):
    # systemd-run acknowledges unit creation before the process actually binds.
    # Inspect the listening socket without opening an iperf control connection.
    for _ in range(20):
        result = subprocess.run(
            ['ss', '-H', '-ltn', 'sport = :' + str(PORT)],
            capture_output=True,
            text=True,
            check=True,
        )
        if any(
            address + ':' + str(PORT) in line.split()
            for line in result.stdout.splitlines()
        ):
            return
        active = subprocess.run(
            ['systemctl', 'is-active', unit],
            capture_output=True,
            text=True,
        )
        if active.stdout.strip() not in ('active', 'activating'):
            raise RuntimeError(
                'Temporary test server failed to start; inspect journalctl -u '
                + unit,
            )
        time.sleep(0.5)
    raise RuntimeError(
        'Temporary test server has not opened its VPN listening port.',
    )


def run_case(binary, server, address, streams, reverse):
    command = [
        binary,
        '-c',
        server,
        '-B',
        address,
        '-p',
        str(PORT),
        '-P',
        str(streams),
        '-b',
        '5M',
        '-t',
        '20',
        '-J',
        '--connect-timeout',
        '15000',
    ]
    if reverse:
        command.append('-R')
    details = {
        'direction': 'employee_download' if reverse else 'employee_upload',
        'streams': streams,
        'target_mbps': streams * 5,
        'server': server,
        'port': PORT,
    }
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=50,
        )
    except subprocess.TimeoutExpired:
        raise LoadStepError(
            {**details, 'error': 'iperf3 exceeded the 50-second timeout.'},
        ) from None
    try:
        value = json.loads(result.stdout)
    except ValueError:
        value = {}
    if (
        result.returncode
        or value.get('error')
        or not value.get('end', {}).get('sum_received')
    ):
        error = (
            value.get('error')
            or result.stderr.strip()
            or 'iperf3 returned no throughput result.'
        )
        error = ' '.join(str(error).split())[:500]
        raise LoadStepError(
            {**details, 'returncode': result.returncode, 'error': error},
        )
    received = value['end']['sum_received']
    mbps = received['bits_per_second'] / 1e6
    target = streams * 5
    return {
        'direction': 'employee_download' if reverse else 'employee_upload',
        'streams': streams,
        'target_mbps': target,
        'received_mbps': round(mbps, 2),
        'delivered_fraction': round(mbps / target, 3),
        'retransmits': value['end'].get('sum_sent', {}).get('retransmits'),
    }


def client(args):
    binary = iperf()
    # This check is separate from iperf: an endpoint on B tests the VPN leg,
    # while the public-IP check confirms that the chosen Internet exit is B.
    result = subprocess.run(
        [
            'curl',
            '-q',
            '--noproxy',
            '*',
            '-4',
            '--fail',
            '--silent',
            '--show-error',
            '--max-time',
            '15',
            'https://api.ipify.org',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    if result.stdout.strip() != '60.249.248.158':
        raise RuntimeError(
            (
                'Select the company exit first; public IPv4 is not '
                '60.249.248.158.'
            ),
        )
    report = {
        ('kind'): (
            'one VPN peer with parallel TCP streams; not separate ' 'employees'
        ),
        'started_at': time.time(),
        'steps': [],
        'completed': False,
        'note': '40 Mbps cap is a bounded first test, not B maximum capacity.',
    }
    try:
        for streams in (1, 2, 4, 8):
            for reverse in (True, False):
                row = run_case(
                    binary,
                    args.server,
                    args.client_ip,
                    streams,
                    reverse,
                )
                report['steps'].append(row)
                print(json.dumps(row), flush=True)
                args.output.write_text(json.dumps(report, indent=2) + '\n')
                if row['delivered_fraction'] < 0.85:
                    print(
                        (
                            'Throughput below 85% of target; '
                            'stopping further load increases.'
                        ),
                        flush=True,
                    )
                    return
                time.sleep(5)
        report['completed'] = True
        print(
            (
                'Bounded test passed through 40 Mbps in each '
                'direction; this is a lower bound, not the maximum.'
            ),
        )
    except LoadStepError as error:
        report['failure'] = {**error.details, 'at': time.time()}
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print('Client report: ' + str(args.output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    commands.add_parser('install-tools')
    commands.add_parser('health-check')
    commands.add_parser('report')
    commands.add_parser('serve').add_argument(
        '--client-ip',
        required=True,
        type=vpn_address,
    )
    remote = commands.add_parser('client')
    remote.add_argument('--server', required=True, type=vpn_address)
    remote.add_argument('--client-ip', required=True, type=vpn_address)
    remote.add_argument(
        ('--output'),
        type=Path,
        default=Path.home() / ('vpn-capacity-client.json'),
    )
    commands.add_parser('cleanup').add_argument(
        '--state',
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    if args.action == 'health-check':
        values = health()
        print(json.dumps(values, indent=2))
        if not all(value['ok'] for value in values.values()):
            raise RuntimeError(
                'Production health check failed; see the statuses above.',
            )
        return
    if args.action == 'client':
        client(args)
        return
    if os.geteuid() != 0:
        raise RuntimeError(
            (
                'Run this B command using sudo; no production '
                'settings were changed.'
            ),
        )
    os.umask(0o077)
    signal.signal(signal.SIGTERM, _interrupted)
    if args.action == 'install-tools':
        install_tools()
    elif args.action == 'report':
        report_latest()
    elif args.action == 'cleanup':
        cleanup(args.state)
    else:
        import fcntl
        import deploy as d

        with (d.ROOT / 'capacity-test.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            serve(args)


def _gateway_health(host):
    # Test the local production gateway with strict certificate validation.
    # This guard does not depend on router hairpin or count as a WAN test.
    # Wiki redirects anonymous page requests through company SSO. Its
    # health endpoint returns 200 directly, without a login/session flow.
    path = HEALTH_PATHS[host]
    result = subprocess.run(
        [
            'curl',
            '-q',
            '--noproxy',
            '*',
            '--silent',
            '--show-error',
            '--max-time',
            '5',
            '--resolve',
            host + ':443:127.0.0.1',
            '--output',
            '/dev/null',
            '--write-out',
            '%{http_code} %{time_total}',
            'https://' + host + path,
        ],
        capture_output=True,
        text=True,
    )
    fields = result.stdout.split()
    return host, {
        'ok': result.returncode == 0 and bool(fields) and fields[0] == '200',
        'path': path,
        'status': fields[0] if fields else None,
        'curl_exit_code': result.returncode,
        'seconds': float(fields[1]) if len(fields) == 2 else 5.0,
    }


def _interrupted(*_):
    raise KeyboardInterrupt()


def _mtime_ns(path):
    return path.stat().st_mtime_ns


if __name__ == '__main__':
    try:
        main()
    except (
        RuntimeError,
        OSError,
        ValueError,
        subprocess.SubprocessError,
    ) as error:
        raise SystemExit(str(error)) from None
