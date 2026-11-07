from __future__ import annotations

import argparse
import fcntl
import ipaddress
import json
import os
import time

import configure_sso as s
import enroll_b as e

import deploy as d

STATE = d.ROOT / 'exit-node.json'
PUBLIC_IP = '60.249.248.158'
CLIENT_GROUP = 'company-vpn-full-tunnel'
EXIT_GROUP = 'company-vpn-exit-servers'
EXIT_NAME = 'company-internet-b'
DNS_NAME = 'company-vpn-internet-dns'


def public_ip():
    value = d.run(
        'curl',
        '-q',
        '--noproxy',
        '*',
        '--ipv4',
        '--fail',
        '--silent',
        '--show-error',
        '--max-time',
        '20',
        'https://api.ipify.org',
    ).strip()
    if value != PUBLIC_IP:
        raise RuntimeError(
            'B internet egress is not the expected company public IPv4.',
        )
    return value


def active_client(token, address):
    ipaddress.IPv4Address(address)
    peers = [p for p in d.api('peers', token=token) if p['ip'] == address]
    if (
        len(peers) != 1
        or not peers[0]['user_id']
        or peers[0].get('approval_required')
    ):
        raise RuntimeError(
            (
                'Choose an existing employee device registered through '
                'company '
                'SSO.'
            ),
        )
    peer = peers[0]
    users = [
        u for u in d.api('users', token=token) if u['id'] == peer['user_id']
    ]
    if (
        len(users) != 1
        or users[0]['is_blocked']
        or users[0]['pending_approval']
        or users[0].get('idp_id') != s.provider_for(token)['id']
    ):
        raise RuntimeError(
            'The employee device owner is not an approved company SSO user.',
        )
    return peer


def payloads(state, enabled):
    clients = (
        list(state['oidc_distribution_groups'].values())
        if state.get(
            'distribution_mode',
        )
        == 'oidc'
        else [state['client_group_id']]
    )
    exits = state['exit_group_id']
    return {
        'policies': {
            'name': EXIT_NAME,
            'description': 'Company full tunnel: employee to B exit only',
            'enabled': enabled,
            'rules': [
                {
                    'name': 'Discover B internet exit',
                    'enabled': True,
                    'action': 'accept',
                    'bidirectional': False,
                    'protocol': 'icmp',
                    'sources': clients,
                    'destinations': [exits],
                },
            ],
        },
        'routes': {
            'network_id': EXIT_NAME,
            'description': 'Company internet via B',
            'network': '0.0.0.0/0',
            'peer': state['server_peer_id'],
            'groups': clients,
            'access_control_groups': [],
            'enabled': enabled,
            'masquerade': True,
            'metric': 9999,
            'keep_route': False,
            'skip_auto_apply': False,
        },
        'dns/nameservers': {
            'name': DNS_NAME,
            'description': 'Internet DNS through company exit',
            'nameservers': [
                {'ip': '1.1.1.1', 'ns_type': 'udp', 'port': 53},
                {'ip': '1.0.0.1', 'ns_type': 'udp', 'port': 53},
            ],
            'groups': clients,
            'domains': [],
            'primary': True,
            'enabled': enabled,
            'search_domains_enabled': False,
        },
    }


def comparable(value, fields):
    result = {key: value.get(key) for key in fields}
    for field in ('peers', 'access_control_groups'):
        if field in result:
            result[field] = [
                item['id'] if isinstance(item, dict) else item
                for item in (result[field] or [])
            ]
    if 'rules' in result:
        result['rules'] = [
            {
                key: rule.get(key)
                for key in (
                    'name',
                    'enabled',
                    'action',
                    'bidirectional',
                    'protocol',
                    'sources',
                    'destinations',
                )
            }
            for rule in result['rules']
        ]
        for rule in result['rules']:
            for side in ('sources', 'destinations'):
                rule[side] = [
                    (
                        g['id']
                        if isinstance(
                            g,
                            dict,
                        )
                        else g
                    )
                    for g in rule[side]
                ]
    return result


def ensure(kind, key, payload, state, token):
    if state.get(key):
        old = d.api(kind + '/' + state[key], token=token)
        # During a resumed apply, only the enabled switch may differ.
        expected = {
            **payload,
            **({'enabled': old['enabled']} if 'enabled' in payload else {}),
        }
        if comparable(old, expected) != comparable(expected, expected):
            raise RuntimeError(
                'Managed VPN object changed; review before overwriting: '
                + kind,
            )
        if old.get('enabled') != payload.get('enabled'):
            d.api(kind + '/' + state[key], 'PUT', payload, token)
        return
    marker = 'network_id' if kind == 'routes' else 'name'
    if any(obj[marker] == payload[marker] for obj in d.api(kind, token=token)):
        raise RuntimeError(
            'An unmanaged VPN object already has this name: '
            + payload[marker],
        )
    obj = d.api(kind, 'POST', payload, token)
    state[key] = obj['id']
    d.save(STATE, state)


def resources(state, enabled, token):
    values = payloads(state, enabled)
    # The route is enabled last; rollback withdraws it first.
    order = [
        ('policies', 'policy_id'),
        ('dns/nameservers', 'dns_id'),
        ('routes', 'route_id'),
    ]
    if not enabled:
        order.reverse()
    for kind, key in order:
        ensure(kind, key, values[kind], state, token)


def verify(token, state):
    if not state.get('complete'):
        raise RuntimeError('Exit node apply has not completed.')
    if e.digest(e.BINARY) != e.BINARY_SHA:
        raise RuntimeError('B client binary differs from the pinned version.')
    value = e.status()
    if not e.connected(value) or not value.get('signal', {}).get('connected'):
        raise RuntimeError(
            'B VPN management/signal connection is not healthy.',
        )
    peer = e.peer_for(value, token)
    if peer['id'] != state['server_peer_id']:
        raise RuntimeError('B VPN identity changed.')
    config = json.loads(e.CONFIG.read_text())
    for name, expected in {
        'DisableDNS': True,
        'DisableClientRoutes': True,
        'DisableServerRoutes': False,
        'BlockLANAccess': True,
        'DisableIPv6': True,
        'DisableAutoConnect': False,
    }.items():
        if config.get(name) != expected:
            raise RuntimeError('Unexpected B exit client setting: ' + name)
    if config.get('ServerSSHAllowed') or config.get('RemoteJobsAllowed'):
        raise RuntimeError('Unexpected B client SSH or remote job setting.')
    for kind, key in [
        ('policies', 'policy_id'),
        ('routes', 'route_id'),
        ('dns/nameservers', 'dns_id'),
    ]:
        expected = payloads(state, True)[kind]
        current = d.api(kind + '/' + state[key], token=token)
        if comparable(current, expected) != comparable(expected, expected):
            raise RuntimeError('Exit configuration differs: ' + kind)
    checked_groups = [('exit_group_id', EXIT_GROUP, [state['server_peer_id']])]
    if state.get('distribution_mode') == 'oidc':
        import auto_exit

        auto_exit.validate_distribution(token, state)
    else:
        checked_groups.append(
            ('client_group_id', CLIENT_GROUP, state['client_peer_ids']),
        )
    for key, name, peers in checked_groups:
        current = d.api('groups/' + state[key], token=token)
        actual = [
            (
                p['id']
                if isinstance(
                    p,
                    dict,
                )
                else p
            )
            for p in current.get('peers', [])
        ]
        if current['name'] != name or set(actual) != set(peers):
            raise RuntimeError(
                'Exit group membership differs; review the recorded scope.',
            )
    if d.run('sysctl', '-n', 'net.ipv4.ip_forward').strip() != '1':
        raise RuntimeError('B IPv4 forwarding is not enabled.')
    for service in ('netbird', 'tailscaled', 'nginx', 'company-vpn'):
        d.run('systemctl', 'is-active', service)
    if d.run('systemctl', 'is-enabled', 'netbird').strip() != 'enabled':
        raise RuntimeError('B VPN client autostart is disabled.')
    for host in (
        'changdar-server.mooo.com',
        'ai.changdar-server.mooo.com',
        'wiki.changdar-server.mooo.com',
    ):
        d.https_check(host)
    public_ip()
    print(
        'B exit ready: '
        + EXIT_NAME
        + '; company public IPv4 '
        + PUBLIC_IP
        + '.',
    )
    print(
        'Scoped employee group, masquerade, auto-apply and VPN DNS verified.',
    )
    print(
        'B autostart, Tailscale and the existing three websites are healthy.',
    )
    print(
        (
            'Employee device verification is still required: select this exit '
            'and check public '
            'IPv4/DNS/IPv6.'
        ),
    )


def disable(token, state):
    if not state:
        raise RuntimeError('No managed exit configuration exists.')
    if state.get('distribution_mode') == 'oidc':
        d.run('systemctl', 'disable', '--now', 'company-vpn-oidc-exit.timer')
    # Do not modify any other VPN objects or the default All policy.
    if state.get('client_group_id') and state.get('exit_group_id'):
        for kind, key in [
            ('routes', 'route_id'),
            ('dns/nameservers', 'dns_id'),
            ('policies', 'policy_id'),
        ]:
            if state.get(key):
                d.api(
                    kind + '/' + state[key],
                    'PUT',
                    payloads(state, False)[kind],
                    token,
                )
    original = state['original_client_flags']
    e.client('down')
    e.client(
        'up',
        '--disable-server-routes='
        + str(original['DisableServerRoutes']).lower(),
        '--block-lan-access=' + str(original['BlockLANAccess']).lower(),
    )
    state['complete'] = False
    state['disabled'] = True
    d.save(STATE, state)
    print(
        (
            'Managed exit route, DNS and access policy disabled; B previous '
            'routing flags '
            'restored.'
        ),
    )


def add_device(token, state, address):
    if not state or not state.get('complete'):
        raise RuntimeError(
            (
                'Apply and verify the B exit before adding another employee '
                'device.'
            ),
        )
    verify(token, state)
    peer = active_client(token, address)
    if state.get('distribution_mode') == 'oidc':
        print(
            (
                'Company SSO devices receive the exit automatically; no '
                'manual group assignment is '
                'needed.'
            ),
        )
        return
    if peer['id'] in state['client_peer_ids']:
        print(
            (
                'Employee device already belongs to the company full-tunnel '
                'group.'
            ),
        )
        return
    updated = {
        **state,
        'client_peer_ids': [*state['client_peer_ids'], peer['id']],
        'client_ips': [*state['client_ips'], address],
    }
    d.api(
        'groups/' + state['client_group_id'],
        'PUT',
        {'name': CLIENT_GROUP, 'peers': updated['client_peer_ids']},
        token,
    )
    d.save(STATE, updated)
    verify(token, updated)
    print('Approved employee device added: ' + address)


def apply(token, address):
    state = json.loads(STATE.read_text()) if STATE.exists() else None
    if state and state.get('complete'):
        active_client(token, address)
        if address not in state['client_ips']:
            raise RuntimeError(
                (
                    'This device is not in the recorded exit scope; add it '
                    'deliberately in a later '
                    'update.'
                ),
            )
        verify(token, state)
        return
    public_ip()
    if not state:
        if not e.STATE.exists() or not json.loads(e.STATE.read_text()).get(
            'complete',
        ):
            e.install(token)
        e.verify(token, json.loads(e.STATE.read_text()))
        client = active_client(token, address)
        server = e.peer_for(e.status(), token)
        settings = d.api('dns/settings', token=token)
        disabled = set(settings['disabled_management_groups'])
        if disabled.intersection(g['id'] for g in client['groups']):
            raise RuntimeError(
                (
                    'DNS management is disabled for this employee device '
                    'group; review it '
                    'first.'
                ),
            )
        nameservers = d.api('dns/nameservers', token=token)
        if any(ns['enabled'] and ns['primary'] for ns in nameservers):
            raise RuntimeError(
                'An existing primary VPN DNS setting needs review first.',
            )
        backup = d.ROOT / ('exit-backup-' + str(time.time_ns()) + '.json')
        config = json.loads(e.CONFIG.read_text())
        d.save(
            backup,
            {
                'client_config': config,
                **{
                    kind: d.api(kind, token=token)
                    for kind in (
                        'groups',
                        'policies',
                        'routes',
                        'dns/nameservers',
                    )
                },
                'dns/settings': settings,
            },
        )
        state = {
            'complete': False,
            'server_peer_id': server['id'],
            'client_peer_ids': [client['id']],
            'client_ips': [address],
            'backup': str(backup),
            'original_client_flags': {
                name: config.get(name, False)
                for name in ('DisableServerRoutes', 'BlockLANAccess')
            },
        }
        d.save(STATE, state)
    elif address not in state['client_ips']:
        raise RuntimeError('Resume with the same employee device IP.')
    active_client(token, address)
    ensure(
        'groups',
        'client_group_id',
        {
            'name': CLIENT_GROUP,
            'peers': state['client_peer_ids'],
        },
        state,
        token,
    )
    ensure(
        'groups',
        'exit_group_id',
        {
            'name': EXIT_GROUP,
            'peers': [
                state['server_peer_id'],
            ],
        },
        state,
        token,
    )
    try:
        resources(state, False, token)
        print(
            'Exit route and VPN DNS staged for the selected employee device.',
            flush=True,
        )
        # 0.80.0's `up` returns early when already connected, before SetConfig.
        # Disconnect this client before applying routing flags; the control
        # plane, Nginx and Tailscale services keep running.
        e.client('down')
        e.client(
            'up',
            '--disable-server-routes=false',
            '--block-lan-access=true',
            '--disable-client-routes=true',
            '--disable-dns=true',
            '--disable-ipv6=true',
            '--allow-server-ssh=false',
            '--allow-remote-jobs=false',
        )
        resources(state, True, token)
        for _ in range(30):
            if (
                e.connected(e.status())
                and d.run('sysctl', '-n', 'net.ipv4.ip_forward').strip() == '1'
            ):
                break
            time.sleep(2)
        state['complete'] = True
        state.pop('disabled', None)
        d.save(STATE, state)
        verify(token, state)
    except Exception:
        # Withdraw only this managed route before returning a deployment error.
        try:
            disable(token, state)
        except Exception:
            print(
                (
                    'Automatic rollback could not finish; run '
                    'configure_exit.py disable '
                    'locally.'
                ),
            )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'action',
        choices=[
            'apply',
            'verify',
            'disable',
            'add-device',
        ],
    )
    parser.add_argument('--client-ip', default='10.203.252.213')
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise RuntimeError('Run on B using sudo.')
    os.umask(0o077)
    if not (d.ROOT / 'sso-ready.json').exists():
        raise RuntimeError('Company VPN SSO must be configured first.')
    with (d.ROOT / 'client-enrollment.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        token = s.get_token()
        if args.action == 'apply':
            apply(token, args.client_ip)
        else:
            state = json.loads(STATE.read_text()) if STATE.exists() else None
            if args.action == 'verify':
                verify(token, state or {})
            elif args.action == 'add-device':
                add_device(token, state, args.client_ip)
            else:
                disable(token, state)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from None
