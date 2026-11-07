from __future__ import annotations

import ipaddress
import json
import shutil
import socket
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

DOMAIN = 'vpn.changdar-server.mooo.com'
MAIN = 'changdar-server.mooo.com'
B_ADDRESS = '192.168.1.86'
VPN_CANDIDATE = ipaddress.ip_network('10.203.0.0/16')


def run(*args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise RuntimeError(args[0] + ' unavailable (may require sudo)')
    return result.stdout


def addresses(host):
    try:
        return sorted(
            {
                r[4][0]
                for r in socket.getaddrinfo(
                    host,
                    None,
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                )
            },
        )
    except socket.gaierror:
        return []


def main():
    interfaces = json.loads(run('ip', '-j', '-4', 'address', 'show'))
    local = {
        a['local']
        for interface in interfaces
        for a in interface.get('addr_info', [])
    }
    if B_ADDRESS not in local:
        raise SystemExit(
            'Run this check on B (expected LAN address ' + B_ADDRESS + ').',
        )
    pending = []
    main_ip, vpn_ip = addresses(MAIN), addresses(DOMAIN)
    print('Main public IPv4:', ', '.join(main_ip) or 'unresolved')
    print('VPN public IPv4:', ', '.join(vpn_ip) or 'unresolved')
    if not main_ip or vpn_ip != main_ip:
        pending.append('VPN A record must match the current main public IPv4.')

    listeners = run('ss', '-H', '-lntu').splitlines()
    for protocol, port in (('tcp', 19080), ('tcp', 19081), ('udp', 3478)):
        occupied = any(
            row.split()[0] == protocol
            and row.split()[4].rsplit(':', 1)[-1] == str(port)
            for row in listeners
        )
        print(
            f"{protocol.upper()} {port}:",
            'IN USE' if occupied else 'available locally',
        )
        if occupied:
            pending.append(
                (
                    f"{protocol.upper()}"
                    ' '
                    f"{port}"
                    ' needs a different mapping or '
                    'investigation.'
                ),
            )
    for tool in ('docker', 'nginx', 'certbot'):
        print(tool + ':', shutil.which(tool) or 'missing')
        if not shutil.which(tool):
            pending.append(tool + ' is required.')

    networks = []
    for route in json.loads(
        run('ip', '-j', '-4', 'route', 'show', 'table', 'all'),
    ):
        destination = route.get('dst', '')
        if destination and destination != 'default':
            networks.append(ipaddress.ip_network(destination, strict=False))
    try:
        print(
            'Docker server:',
            run(
                'docker',
                'version',
                '--format',
                '{{.Server.Version}}',
            ).strip(),
        )
        ids = run('docker', 'network', 'ls', '-q').split()
        if ids:
            for network in json.loads(
                run('docker', 'network', 'inspect', *ids),
            ):
                for config in network.get('IPAM', {}).get('Config') or []:
                    if config.get('Subnet'):
                        subnet = ipaddress.ip_network(
                            config['Subnet'],
                            strict=False,
                        )
                        if subnet.version == 4:
                            networks.append(subnet)
    except (RuntimeError, FileNotFoundError):
        pending.append('Repeat with sudo to inspect Docker network ranges.')
    overlaps = sorted({str(n) for n in networks if n.overlaps(VPN_CANDIDATE)})
    print('Candidate VPN range:', str(VPN_CANDIDATE), '(not configured)')
    print(
        'Overlap with local routes/Docker:',
        ', '.join(overlaps) or 'none observed',
    )
    if overlaps:
        pending.append('Choose another VPN range before enrolling devices.')
    available = next(
        (
            int(line.split()[1]) // 1024
            for line in Path('/proc/meminfo').read_text().splitlines()
            if line.startswith('MemAvailable:')
        ),
        0,
    )
    print('Available RAM:', available, 'MiB')
    print(
        'Free disk:',
        shutil.disk_usage(
            Path.home(),
        ).free
        // (1024**3),
        'GiB',
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for host in (MAIN, 'ai.' + MAIN, 'wiki.' + MAIN):
        try:
            with opener.open('https://' + host + '/', timeout=12) as response:
                print('Existing HTTPS:', host, response.status)
        except (urllib.error.URLError, TimeoutError):
            pending.append('Investigate existing HTTPS service: ' + host)
    print('Router UDP forwarding is not tested by this local check.')
    print(
        (
            'Employee home networks must also be checked before choosing the '
            'final VPN '
            'range.'
        ),
    )
    print(
        (
            'No services, firewall, routes, DNS or Keycloak settings were '
            'changed.'
        ),
    )
    if pending:
        print('Pending prerequisites:')
        for item in pending:
            print('- ' + item)
    else:
        print(
            (
                'Local prerequisites passed; deployment and external '
                'connectivity testing are separate '
                'steps.'
            ),
        )


if __name__ == '__main__':
    main()
