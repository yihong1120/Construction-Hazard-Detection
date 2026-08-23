from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import timedelta
from uuid import UUID

from examples.auth.config import Settings
from examples.auth.database import AsyncSessionLocal
from examples.auth.models import utc_now
from examples.deployment_registry.enrollments import provision_enrollment_code


def _arguments() -> argparse.Namespace:
    """Parse the minimal audited parameters needed to provision a code."""
    parser = argparse.ArgumentParser(
        description='Create one one-time Deployment Registry enrollment code.',
    )
    parser.add_argument('--deployment-id', required=True, type=UUID)
    parser.add_argument(
        '--created-by',
        required=True,
        help='Operator or ticket reference recorded with the verifier.',
    )
    parser.add_argument(
        '--expires-in-hours',
        type=int,
        default=24,
        help='Positive validity period; defaults to 24 hours.',
    )
    return parser.parse_args()


async def _create(args: argparse.Namespace) -> str:
    """Create and commit one verifier, returning its one-time raw code."""
    if args.expires_in_hours <= 0 or args.expires_in_hours > 24 * 365:
        raise ValueError('--expires-in-hours must be between 1 and 8760')
    settings = Settings()
    expires_at = utc_now() + timedelta(hours=args.expires_in_hours)
    async with AsyncSessionLocal() as db:
        provisioned = await provision_enrollment_code(
            db,
            deployment_id=args.deployment_id,
            expires_at=expires_at,
            created_by=args.created_by,
            pepper=settings.deployment_enrollment_code_pepper,
        )
        await db.commit()
    return provisioned.raw_code


def main() -> int:
    """Run trusted provisioning and print the raw code only after commit."""
    try:
        code = asyncio.run(_create(_arguments()))
    except ValueError as exc:
        # These messages never contain a raw enrollment code.
        print(f'Enrollment code was not created: {exc}', file=sys.stderr)
        return 2
    print(code)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
