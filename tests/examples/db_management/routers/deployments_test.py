from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.models import Deployment
from examples.auth.models import Tenant
from examples.db_management.routers import deployments
from examples.db_management.schemas.deployment import DeploymentCreate
from examples.db_management.schemas.deployment import DeploymentUpdate
from examples.db_management.schemas.deployment import TenantCreate
from examples.db_management.schemas.deployment import TenantUpdate


_TENANT_ID = UUID('11111111-1111-1111-1111-111111111111')
_DEPLOYMENT_ID = UUID('22222222-2222-2222-2222-222222222222')


def _tenant() -> Tenant:
    """Build a complete tenant model suitable for read-schema validation.

    Returns:
        Active tenant with deterministic identifiers and metadata.
    """
    tenant = Tenant(name='Tenant A', description='Construction tenant')
    tenant.id = _TENANT_ID
    tenant.status = 'active'
    return tenant


def _deployment() -> Deployment:
    """Build a complete deployment model for management route tests.

    Returns:
        Active deployment with a stable public API base URL.
    """
    deployment = Deployment(
        tenant_id=_TENANT_ID,
        api_base_url='https://api.example.test/hazard/api',
    )
    deployment.id = _DEPLOYMENT_ID
    deployment.config_revision = 1
    deployment.status = 'active'
    return deployment


def _refresh_tenant(tenant: Tenant) -> None:
    """Populate database-generated tenant fields in a unit-test refresh.

    Args:
        tenant: Newly created tenant model refreshed by the session double.
    """
    tenant.id = _TENANT_ID
    tenant.status = 'active'


class TestDeploymentManagementRoutes(unittest.IsolatedAsyncioTestCase):
    """Verify privileged tenant and deployment state-management handlers."""

    async def test_list_and_create_tenant_use_management_schema(self) -> None:
        """Tenant list and creation return only the declared public fields."""
        tenant = _tenant()
        deployment = _deployment()
        db = SimpleNamespace(
            execute=AsyncMock(
                side_effect=[
                    SimpleNamespace(scalars=lambda: [tenant]),
                    SimpleNamespace(scalars=lambda: [deployment]),
                ],
            ),
            add=MagicMock(),
            commit=AsyncMock(),
            refresh=AsyncMock(side_effect=_refresh_tenant),
            rollback=AsyncMock(),
        )

        listed = await deployments.list_tenants(
            object(),
            cast(AsyncSession, db),
        )
        listed_deployments = await deployments.list_deployments(
            object(),
            cast(AsyncSession, db),
        )
        created = await deployments.create_tenant(
            TenantCreate(name='  Tenant B  ', description='New tenant'),
            object(),
            cast(AsyncSession, db),
        )

        self.assertEqual(listed[0].id, _TENANT_ID)
        self.assertEqual(listed_deployments[0].id, _DEPLOYMENT_ID)
        self.assertEqual(created.name, 'Tenant B')
        self.assertEqual(created.status, 'active')
        db.commit.assert_awaited_once()
        db.refresh.assert_awaited_once()

    async def test_update_tenant_revisions_all_deployments_on_lifecycle_change(
        self,
    ) -> None:
        """Disabling a tenant advances every dependent deployment revision."""
        tenant = _tenant()
        first = _deployment()
        second = _deployment()
        second.id = UUID('33333333-3333-3333-3333-333333333333')
        second.config_revision = 4
        db = SimpleNamespace(
            get=AsyncMock(return_value=tenant),
            execute=AsyncMock(
                return_value=SimpleNamespace(scalars=lambda: [first, second]),
            ),
            commit=AsyncMock(),
            refresh=AsyncMock(),
            rollback=AsyncMock(),
        )

        updated = await deployments.update_tenant(
            _TENANT_ID,
            TenantUpdate(
                name='Tenant A Updated',
                description='Disabled for maintenance',
                status='disabled',
            ),
            object(),
            cast(AsyncSession, db),
        )

        self.assertEqual(updated.status, 'disabled')
        self.assertEqual(tenant.name, 'Tenant A Updated')
        self.assertEqual(first.config_revision, 2)
        self.assertEqual(second.config_revision, 5)
        db.commit.assert_awaited_once()

    async def test_create_and_update_deployment_canonicalise_url(self) -> None:
        """Deployment create and update validate a canonical API URL once."""
        tenant = _tenant()
        deployment = _deployment()

        def refresh(item: Deployment) -> None:
            """Populate only the newly created deployment's default fields.

            Args:
                item: Deployment being refreshed by the session double.
            """
            if item is deployment:
                return
            item.id = _DEPLOYMENT_ID
            item.config_revision = 1
            item.status = 'active'

        db = SimpleNamespace(
            get=AsyncMock(side_effect=[tenant, deployment, tenant]),
            add=MagicMock(),
            commit=AsyncMock(),
            refresh=AsyncMock(side_effect=refresh),
            rollback=AsyncMock(),
        )

        created = await deployments.create_deployment(
            DeploymentCreate(
                tenant_id=_TENANT_ID,
                api_base_url='https://api.example.test/hazard/api/',
            ),
            object(),
            cast(AsyncSession, db),
        )
        updated = await deployments.update_deployment(
            _DEPLOYMENT_ID,
            DeploymentUpdate(
                tenant_id=_TENANT_ID,
                api_base_url='https://api.example.test/hazard/api/',
                status='revoked',
            ),
            object(),
            cast(AsyncSession, db),
        )

        self.assertEqual(
            created.api_base_url,
            'https://api.example.test/hazard/api',
        )
        self.assertEqual(updated.status, 'revoked')
        self.assertEqual(
            deployment.api_base_url,
            'https://api.example.test/hazard/api',
        )
        db.commit.assert_awaited()

    async def test_missing_records_and_invalid_status_return_client_errors(
        self,
    ) -> None:
        """Management routes distinguish unknown resources and invalid states.
        """
        db = SimpleNamespace(get=AsyncMock(return_value=None))

        with self.assertRaises(HTTPException) as missing:
            await deployments.update_tenant(
                _TENANT_ID,
                TenantUpdate(status='active'),
                object(),
                cast(AsyncSession, db),
            )
        self.assertEqual(missing.exception.status_code, 404)

        with self.assertRaises(HTTPException) as invalid:
            raise deployments._invalid_status('tenant', 'deleted')
        self.assertEqual(invalid.exception.status_code, 422)

    async def test_deployment_create_and_update_validate_resources(
        self,
    ) -> None:
        """Deployment routes reject unknown tenants, IDs, URLs, and states."""
        creation = DeploymentCreate(
            tenant_id=_TENANT_ID,
            api_base_url='https://api.example.test/hazard/api',
        )

        with self.assertRaises(HTTPException) as missing_tenant:
            await deployments.create_deployment(
                creation,
                object(),
                cast(
                    AsyncSession,
                    SimpleNamespace(get=AsyncMock(return_value=None)),
                ),
            )
        self.assertEqual(missing_tenant.exception.status_code, 404)

        with self.assertRaises(HTTPException) as invalid_url:
            await deployments.create_deployment(
                DeploymentCreate(
                    tenant_id=_TENANT_ID,
                    api_base_url='http://api.example.test/hazard/api',
                ),
                object(),
                cast(
                    AsyncSession,
                    SimpleNamespace(get=AsyncMock(return_value=_tenant())),
                ),
            )
        self.assertEqual(invalid_url.exception.status_code, 422)

        with self.assertRaises(HTTPException) as missing_deployment:
            await deployments.update_deployment(
                _DEPLOYMENT_ID,
                DeploymentUpdate(),
                object(),
                cast(
                    AsyncSession,
                    SimpleNamespace(get=AsyncMock(return_value=None)),
                ),
            )
        self.assertEqual(missing_deployment.exception.status_code, 404)

        with self.assertRaises(HTTPException) as invalid_status:
            await deployments.update_deployment(
                _DEPLOYMENT_ID,
                DeploymentUpdate(status='deleted'),
                object(),
                cast(
                    AsyncSession,
                    SimpleNamespace(get=AsyncMock(return_value=_deployment())),
                ),
            )
        self.assertEqual(invalid_status.exception.status_code, 422)

    async def test_conflicts_and_nested_update_resources_return_client_errors(
        self,
    ) -> None:
        """Management writes translate unique conflicts and missing tenants."""
        conflict = IntegrityError('insert', {}, RuntimeError('duplicate'))
        write_db = SimpleNamespace(
            add=MagicMock(),
            commit=AsyncMock(side_effect=conflict),
            rollback=AsyncMock(),
            refresh=AsyncMock(),
        )
        with self.assertRaises(HTTPException) as tenant_conflict:
            await deployments.create_tenant(
                TenantCreate(name='Existing'),
                object(),
                cast(AsyncSession, write_db),
            )
        self.assertEqual(tenant_conflict.exception.status_code, 409)

        update_db = SimpleNamespace(
            get=AsyncMock(side_effect=[_deployment(), None]),
        )
        with self.assertRaises(HTTPException) as missing_update_tenant:
            await deployments.update_deployment(
                _DEPLOYMENT_ID,
                DeploymentUpdate(tenant_id=_TENANT_ID),
                object(),
                cast(AsyncSession, update_db),
            )
        self.assertEqual(missing_update_tenant.exception.status_code, 404)

        with self.assertRaises(HTTPException) as invalid_update_url:
            await deployments.update_deployment(
                _DEPLOYMENT_ID,
                DeploymentUpdate(api_base_url='http://invalid.example.test'),
                object(),
                cast(
                    AsyncSession,
                    SimpleNamespace(get=AsyncMock(return_value=_deployment())),
                ),
            )
        self.assertEqual(invalid_update_url.exception.status_code, 422)

        deployment_db = SimpleNamespace(
            get=AsyncMock(return_value=_tenant()),
            add=MagicMock(),
            commit=AsyncMock(side_effect=conflict),
            rollback=AsyncMock(),
            refresh=AsyncMock(),
        )
        with self.assertRaises(HTTPException) as deployment_conflict:
            await deployments.create_deployment(
                DeploymentCreate(
                    tenant_id=_TENANT_ID,
                    api_base_url='https://api.example.test/hazard/api',
                ),
                object(),
                cast(AsyncSession, deployment_db),
            )
        self.assertEqual(deployment_conflict.exception.status_code, 409)

        update_conflict_db = SimpleNamespace(
            get=AsyncMock(return_value=_deployment()),
            commit=AsyncMock(side_effect=conflict),
            rollback=AsyncMock(),
            refresh=AsyncMock(),
        )
        with self.assertRaises(HTTPException) as update_conflict:
            await deployments.update_deployment(
                _DEPLOYMENT_ID,
                DeploymentUpdate(),
                object(),
                cast(AsyncSession, update_conflict_db),
            )
        self.assertEqual(update_conflict.exception.status_code, 409)


if __name__ == '__main__':
    unittest.main()
