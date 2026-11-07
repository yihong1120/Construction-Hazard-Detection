from __future__ import annotations

import ipaddress
import json
from datetime import datetime
from datetime import timezone
from operator import attrgetter
from uuid import UUID
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.models import utc_now
from examples.db_management.policy_settings import DEFAULT_WEBSITE_HOSTS
from examples.db_management.policy_settings import WebsiteAccessSettings
from examples.db_management.schemas.website_access import canonical_ip
from examples.db_management.schemas.website_access import PolicyRead
from examples.db_management.schemas.website_access import RuleInput
from examples.db_management.schemas.website_access import RuleRead
from examples.db_management.schemas.website_access import SiteKey
from examples.db_management.website_access_models import WebsiteAccessAudit
from examples.db_management.website_access_models import WebsiteAccessPolicy
from examples.db_management.website_access_models import WebsiteAccessRule

DEFAULT_HOSTS = DEFAULT_WEBSITE_HOSTS
MAX_RULES = 256


def gateway_status() -> dict:
    """Report only a fresh receipt for the configured production hosts."""
    try:
        config = WebsiteAccessSettings()
        receipt = json.loads(config.status_file.read_text())
        age = (
            datetime.now(timezone.utc)
            - datetime.fromisoformat(receipt['checked_at'])
        ).total_seconds()
        hosts_match = receipt['hosts'] == config.hosts
        versions = receipt['versions']
        if set(versions) != set(DEFAULT_HOSTS) or any(
            type(v) is not int or v < 0 for v in versions.values()
        ):
            raise ValueError('Invalid receipt')
        return {
            'online': hosts_match and 0 <= age < 30,
            'checked_at': receipt['checked_at'],
            'versions': versions,
        }
    except (OSError, ValueError, KeyError, TypeError):
        return {'online': False, 'checked_at': None, 'versions': {}}


def website_hosts() -> dict[str, str]:
    """Use deployment-owned hostnames, never administrator-submitted URLs."""
    return WebsiteAccessSettings().hosts


def snapshot(policy: WebsiteAccessPolicy) -> dict:
    """Return an audit/agent snapshot with validated, non-secret fields."""
    return {
        'site_key': policy.site_key,
        'restricted': policy.restricted,
        'version': policy.version,
        'rules': [
            {
                'id': str(r.id),
                'network': r.network,
                'label': r.label,
                'enabled': r.enabled,
            }
            for r in sorted(policy.rules, key=attrgetter('network'))
        ],
    }


def ip_allowed(policy: dict, address: str) -> bool:
    """Match IPv4/IPv6 without trusting forwarded headers or DNS names."""
    ip = ipaddress.ip_address(canonical_ip(address))
    if not policy['restricted']:
        return True
    return any(
        rule['enabled'] and ip in ipaddress.ip_network(rule['network'])
        for rule in policy['rules']
    )


def projection(policy: WebsiteAccessPolicy) -> PolicyRead:
    """Present the immutable website hostname alongside editable rules."""
    return PolicyRead(
        site_key=policy.site_key,
        hostname=website_hosts()[policy.site_key],
        restricted=policy.restricted,
        version=policy.version,
        updated_at=policy.updated_at,
        rules=[RuleRead.model_validate(r) for r in policy.rules],
    )


async def policies(db: AsyncSession) -> list[WebsiteAccessPolicy]:
    """Require the complete migrated catalog before returning policies."""
    rows = list(
        (
            await db.execute(
                select(WebsiteAccessPolicy).order_by(
                    WebsiteAccessPolicy.site_key,
                ),
            )
        )
        .scalars()
        .all(),
    )
    if {r.site_key for r in rows} != DEFAULT_HOSTS.keys():
        raise HTTPException(
            503,
            detail={'code': 'website_access_schema_not_ready'},
        )
    return rows


async def locked_policy(
    db: AsyncSession,
    site: str,
    version: int,
) -> WebsiteAccessPolicy:
    """Serialize changes to each policy and reject stale management forms."""
    row = (
        await db.execute(
            select(WebsiteAccessPolicy)
            .where(
                WebsiteAccessPolicy.site_key == site,
            )
            .with_for_update(),
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(404, detail={'code': 'website_not_found'})
    if row.version != version:
        raise HTTPException(
            409,
            detail={'code': 'website_access_version_conflict'},
        )
    return row


async def save_change(
    db: AsyncSession,
    policy: WebsiteAccessPolicy,
    before: dict,
    actor_id: int,
    actor_ip: str,
    action: str,
) -> PolicyRead:
    """Commit audit and policy together, rejecting administrator lockout."""
    if not ip_allowed(snapshot(policy), actor_ip):
        await db.rollback()
        raise HTTPException(
            409,
            detail={'code': 'website_access_would_lock_you_out'},
        )
    policy.version += 1
    policy.updated_at = utc_now()
    db.add(
        WebsiteAccessAudit(
            site_key=policy.site_key,
            actor_user_id=actor_id,
            actor_ip=canonical_ip(actor_ip),
            action=action,
            before=before,
            after=snapshot(policy),
        ),
    )
    result = projection(policy)
    try:
        await db.commit()
    except Exception:
        await db.rollback()
        raise
    return result


async def change_rule(
    db: AsyncSession,
    policy: WebsiteAccessPolicy,
    payload: RuleInput,
    rule_id: UUID | None,
) -> None:
    """Replace a rule or append a uniquely addressed, bounded new entry."""
    target = next((r for r in policy.rules if r.id == rule_id), None)
    if rule_id is not None and target is None:
        raise HTTPException(
            404,
            detail={'code': 'website_access_rule_not_found'},
        )
    if any(
        r.network == payload.network and r.id != rule_id for r in policy.rules
    ):
        raise HTTPException(
            409,
            detail={'code': 'website_access_duplicate_network'},
        )
    if target is None:
        if len(policy.rules) >= MAX_RULES:
            raise HTTPException(
                409,
                detail={'code': 'website_access_rule_limit'},
            )
        target = WebsiteAccessRule(id=uuid4(), created_at=utc_now())
        policy.rules.append(target)
    target.network, target.label, target.enabled = (
        payload.network,
        payload.label,
        payload.enabled,
    )


async def list_audit(
    db: AsyncSession,
    limit: int,
    site_key: SiteKey | None = None,
) -> list[WebsiteAccessAudit]:
    """Read the newest bounded change records for all or one website."""
    stmt = (
        select(WebsiteAccessAudit)
        .order_by(WebsiteAccessAudit.created_at.desc())
        .limit(limit)
    )
    if site_key is not None:
        stmt = stmt.where(WebsiteAccessAudit.site_key == site_key)
    return list((await db.execute(stmt)).scalars().all())


async def update_policy(
    db: AsyncSession,
    site_key: SiteKey,
    restricted: bool,
    expected_version: int,
    actor_id: int,
    actor_ip: str,
) -> PolicyRead:
    """Change enforcement mode in the same transaction as its audit row."""
    row = await locked_policy(db, site_key, expected_version)
    before = snapshot(row)
    row.restricted = restricted
    return await save_change(
        db,
        row,
        before,
        actor_id,
        actor_ip,
        'policy_updated',
    )


async def create_rule(
    db: AsyncSession,
    site_key: SiteKey,
    payload: RuleInput,
    actor_id: int,
    actor_ip: str,
) -> PolicyRead:
    """Add one validated allowlist entry and audit the resulting policy."""
    row = await locked_policy(db, site_key, payload.expected_version)
    before = snapshot(row)
    await change_rule(db, row, payload, None)
    return await save_change(
        db,
        row,
        before,
        actor_id,
        actor_ip,
        'rule_created',
    )


async def update_rule(
    db: AsyncSession,
    site_key: SiteKey,
    rule_id: UUID,
    payload: RuleInput,
    actor_id: int,
    actor_ip: str,
) -> PolicyRead:
    """Replace a rule belonging to the named website and record the change."""
    row = await locked_policy(db, site_key, payload.expected_version)
    before = snapshot(row)
    await change_rule(db, row, payload, rule_id)
    return await save_change(
        db,
        row,
        before,
        actor_id,
        actor_ip,
        'rule_updated',
    )


async def delete_rule(
    db: AsyncSession,
    site_key: SiteKey,
    rule_id: UUID,
    expected_version: int,
    actor_id: int,
    actor_ip: str,
) -> PolicyRead:
    """Remove a rule from its own policy, enforcing lockout protection."""
    row = await locked_policy(db, site_key, expected_version)
    before = snapshot(row)
    target = next((r for r in row.rules if r.id == rule_id), None)
    if target is None:
        raise HTTPException(
            404,
            detail={'code': 'website_access_rule_not_found'},
        )
    row.rules.remove(target)
    return await save_change(
        db,
        row,
        before,
        actor_id,
        actor_ip,
        'rule_deleted',
    )
