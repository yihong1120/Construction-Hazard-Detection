"""Link already-provisioned Keycloak users to Visionnaire accounts safely.

This tool intentionally does not read or migrate password hashes. Keycloak is
the password authority; the script only records the immutable Keycloak user
UUID (the OIDC ``sub`` claim) in ``user_identities`` after administrators have
provisioned the corresponding Keycloak accounts.
"""
from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Sequence
from dataclasses import dataclass

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.database import AsyncSessionLocal
from examples.auth.models import User
from examples.auth.models import UserIdentity


@dataclass(frozen=True, slots=True)
class KeycloakUser:
    """Minimal, stable Keycloak account fields used for identity linking."""

    subject: str
    username: str
    email: str | None
    email_verified: bool
    display_name: str | None


def _arguments() -> argparse.Namespace:
    """Parse explicit Keycloak administration credentials and dry-run mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-url', required=True)
    parser.add_argument('--realm', required=True)
    parser.add_argument('--client-id', required=True)
    parser.add_argument('--client-secret', required=True)
    parser.add_argument('--admin-realm', default='master')
    parser.add_argument('--provider', default='keycloak')
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Persist links. The default only reports the proposed changes.',
    )
    return parser.parse_args()


def _normalise_server_url(value: str) -> str:
    """Return a Keycloak root URL without a trailing slash."""
    server_url = value.strip().rstrip('/')
    if not server_url.startswith(('https://', 'http://')):
        raise ValueError('--server-url must be an absolute HTTP(S) URL')
    return server_url


async def _admin_token(
    client: httpx.AsyncClient,
    *,
    server_url: str,
    admin_realm: str,
    client_id: str,
    client_secret: str,
) -> str:
    """Obtain a Keycloak service-account administration access token."""
    response = await client.post(
        f'{server_url}/realms/{admin_realm}/protocol/openid-connect/token',
        data={'grant_type': 'client_credentials'},
        auth=(client_id, client_secret),
    )
    response.raise_for_status()
    token = response.json().get('access_token')
    if not isinstance(token, str) or not token:
        raise RuntimeError(
            'Keycloak token response did not include access_token',
        )
    return token


async def _keycloak_users(
    client: httpx.AsyncClient,
    *,
    server_url: str,
    realm: str,
    access_token: str,
) -> list[KeycloakUser]:
    """Retrieve every Keycloak user with pagination and validate its ID."""
    users: list[KeycloakUser] = []
    first = 0
    headers = {'Authorization': f'Bearer {access_token}'}
    while True:
        response = await client.get(
            f'{server_url}/admin/realms/{realm}/users',
            headers=headers,
            params={'briefRepresentation': 'true', 'first': first, 'max': 100},
        )
        response.raise_for_status()
        page = response.json()
        if not isinstance(page, list):
            raise RuntimeError('Keycloak users response is not a list')
        for item in page:
            if not isinstance(item, dict):
                continue
            if item.get('serviceAccountClientLink'):
                continue
            subject = item.get('id')
            username = item.get('username')
            if not isinstance(subject, str) or not isinstance(username, str):
                continue
            email = item.get('email')
            first_name = item.get('firstName')
            last_name = item.get('lastName')
            display_name = ' '.join(
                str(part).strip()
                for part in (first_name, last_name)
                if isinstance(part, str) and part.strip()
            ) or None
            users.append(
                KeycloakUser(
                    subject=subject,
                    username=username,
                    email=email if isinstance(email, str) else None,
                    email_verified=bool(item.get('emailVerified')),
                    display_name=display_name,
                ),
            )
        if len(page) < 100:
            return users
        first += len(page)


async def _link_users(
    db: AsyncSession,
    remote_users: Sequence[KeycloakUser],
    *,
    provider: str,
    apply: bool,
) -> tuple[int, list[str]]:
    """Create only unambiguous local-user to stable-Keycloak-ID links."""
    local_users = (
        (await db.execute(select(User))).unique().scalars().all()
    )
    local_by_username = {user.username: user for user in local_users}
    local_by_normalized_username: dict[str, list[User]] = {}
    for user in local_users:
        local_by_normalized_username.setdefault(
            user.username.casefold(),
            [],
        ).append(user)
    identities = (
        (
            await db.execute(
                select(UserIdentity).where(UserIdentity.provider == provider),
            )
        )
        .scalars()
        .all()
    )
    by_subject = {
        identity.provider_user_id: identity for identity in identities
    }
    by_user_id = {identity.user_id: identity for identity in identities}
    linked = 0
    warnings: list[str] = []
    for remote in remote_users:
        user = local_by_username.get(remote.username)
        if user is None:
            candidates = local_by_normalized_username.get(
                remote.username.casefold(),
                [],
            )
            if len(candidates) == 1:
                user = candidates[0]
            elif len(candidates) > 1:
                warnings.append(
                    f'skip {remote.username}: local username match is '
                    'ambiguous when case is ignored',
                )
                continue
        if user is None:
            warnings.append(
                f'skip {remote.username}: local username not found',
            )
            continue
        existing_subject = by_subject.get(remote.subject)
        existing_user = by_user_id.get(user.id)
        if (
            existing_subject is not None
            and existing_subject.user_id != user.id
        ):
            warnings.append(
                f'skip {remote.username}: Keycloak ID is linked to another '
                'user',
            )
            continue
        if existing_user is not None and (
            existing_user.provider_user_id != remote.subject
        ):
            warnings.append(
                f'skip {remote.username}: local user has another Keycloak ID',
            )
            continue
        if existing_subject is not None:
            continue
        linked += 1
        if not apply:
            continue
        identity = UserIdentity(
            user_id=user.id,
            provider=provider,
            provider_user_id=remote.subject,
            email=remote.email.strip().lower() if remote.email else None,
            email_verified=remote.email_verified,
            display_name=remote.display_name,
            raw_profile={'keycloak_username': remote.username},
        )
        db.add(identity)
        by_subject[remote.subject] = identity
        by_user_id[user.id] = identity
    if apply:
        await db.commit()
    return linked, warnings


async def _run(args: argparse.Namespace) -> int:
    """Fetch Keycloak users and create safe links in one database session."""
    if len(args.provider.strip()) > 20 or not args.provider.strip():
        raise ValueError('--provider must contain 1 to 20 characters')
    if not os.getenv('DATABASE_URL'):
        raise RuntimeError('DATABASE_URL is required')
    server_url = _normalise_server_url(args.server_url)
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(20.0, connect=5.0),
    ) as client:
        token = await _admin_token(
            client,
            server_url=server_url,
            admin_realm=args.admin_realm,
            client_id=args.client_id,
            client_secret=args.client_secret,
        )
        remote_users = await _keycloak_users(
            client,
            server_url=server_url,
            realm=args.realm,
            access_token=token,
        )
    async with AsyncSessionLocal() as db:
        linked, warnings = await _link_users(
            db,
            remote_users,
            provider=args.provider.strip(),
            apply=args.apply,
        )
    action = 'linked' if args.apply else 'would link'
    print(f'{action} {linked} of {len(remote_users)} Keycloak users')
    for warning in warnings:
        print(f'warning: {warning}')
    return 0 if not warnings else 2


if __name__ == '__main__':
    try:
        raise SystemExit(asyncio.run(_run(_arguments())))
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        raise SystemExit(f'keycloak link failed: {exc}') from exc
