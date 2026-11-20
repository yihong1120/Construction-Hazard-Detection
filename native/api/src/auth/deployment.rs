use std::{collections::HashMap, sync::Arc};

use anyhow::{Result, ensure};
use axum::http::{HeaderMap, StatusCode, header, uri::Authority};
use serde::Deserialize;
use sqlx::PgPool;
use url::{Position, Url};

use crate::{
    ApiError,
    tenancy::{DeploymentScope, database_unavailable},
};

use super::{Grant, PermissionSet, VerifiedAccess, invalid_token, permission_denied};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatabaseAccessConfig {
    pub public_api_base_url: String,
    #[serde(default)]
    pub loopback_deployment_access: bool,
}

impl DatabaseAccessConfig {
    fn authority(&self) -> Result<String> {
        let url = Url::parse(&self.public_api_base_url)?;
        ensure!(
            url.scheme() == "https"
                && url.has_host()
                && url.username().is_empty()
                && url.password().is_none()
                && url.query().is_none()
                && url.fragment().is_none()
                && url.path().len() > 1
                && !url.path().ends_with('/')
                && !url.path().contains('%')
                && url.as_str() == self.public_api_base_url,
            "Configure the canonical HTTPS deployment API URL"
        );
        Ok(url[Position::BeforeHost..Position::AfterPort].to_owned())
    }

    pub fn validate(&self) -> Result<()> {
        self.authority().map(drop)
    }
}

pub(crate) struct DatabaseAccess {
    db: PgPool,
    scope: DeploymentScope,
    api_url: String,
    authority: String,
    loopback_deployment_access: bool,
}

pub(super) struct LocalAccount {
    pub id: i32,
    pub username: String,
    pub role: String,
}

type Sites = Arc<HashMap<i64, String>>;

impl DatabaseAccess {
    pub(crate) async fn new(
        config: DatabaseAccessConfig,
        scope: DeploymentScope,
        db: PgPool,
    ) -> Result<Self> {
        let authority = config.authority()?;
        let access = Self {
            db,
            scope,
            api_url: config.public_api_base_url,
            authority,
            loopback_deployment_access: config.loopback_deployment_access,
        };
        ensure!(
            access.active().await?,
            "Database access origin does not match the active configured deployment"
        );
        Ok(access)
    }

    async fn active(&self) -> Result<bool, sqlx::Error> {
        sqlx::query_scalar(
            "SELECT EXISTS (
                SELECT 1 FROM public.deployments d JOIN public.tenants t ON t.id = d.tenant_id
                WHERE d.id = $1 AND t.id = $2 AND d.api_base_url = $3
                  AND d.status = 'active' AND t.status = 'active'
            )",
        )
        .bind(self.scope.deployment_id)
        .bind(self.scope.tenant_id)
        .bind(&self.api_url)
        .fetch_one(&self.db)
        .await
    }

    pub(super) async fn authorize(
        &self,
        headers: &HeaderMap,
        access: &VerifiedAccess,
        grant: Grant,
        configured: Option<&PermissionSet>,
    ) -> Result<(Sites, Option<LocalAccount>), ApiError> {
        let mut hosts = headers.get_all(header::HOST).iter();
        let host = hosts.next().and_then(|value| value.to_str().ok());
        // This option matches the configured deployment on the loopback-only
        // listener. A public reverse proxy replaces Host with its own origin.
        let origin_allowed = host == Some(&self.authority)
            || self.loopback_deployment_access && host.is_some_and(loopback);
        if !origin_allowed || hosts.next().is_some() {
            return Err(ApiError::new(
                StatusCode::CONFLICT,
                "deployment_origin_mismatch",
            ));
        }
        if let Some(entry) = configured {
            if entry.permissions.contains(&grant) {
                if !self.active().await.map_err(database_unavailable)? {
                    return Err(ApiError::new(
                        StatusCode::CONFLICT,
                        "deployment_configuration_changed",
                    ));
                }
                return Ok((entry.sites.clone(), None));
            }
            // A dispatch grant does not replace the linked account's inbox,
            // device registration, or notification preferences permissions.
            // Their site access comes from the database below.
            if grant != Grant::NotificationRead
                || !entry.permissions.contains(&Grant::NotificationDispatch)
            {
                return Err(permission_denied());
            }
        }
        let (id, username, role, site_ids, site_names): (i32, String, String, Vec<i32>, Vec<String>) =
            sqlx::query_as(
                "SELECT u.id, u.username, u.role,
                        COALESCE(ARRAY_AGG(s.id ORDER BY s.id) FILTER (WHERE s.id IS NOT NULL), '{}'),
                        COALESCE(ARRAY_AGG(s.name::text ORDER BY s.id) FILTER (WHERE s.id IS NOT NULL), '{}')
                 FROM public.user_identities i
                 JOIN users u ON u.id = i.user_id
                 JOIN public.tenants t ON t.id = u.tenant_id
                 JOIN public.deployments d ON d.tenant_id = t.id
                 LEFT JOIN sites s ON (
                     EXISTS (SELECT 1 FROM user_sites us
                             JOIN site_groups sg ON sg.site_id = us.site_id
                             WHERE us.user_id = u.id AND us.site_id = s.id
                               AND sg.group_id = u.group_id)
                     OR (u.role = 'super_admin' AND EXISTS (
                         SELECT 1 FROM site_groups sg JOIN users member ON member.group_id = sg.group_id
                         WHERE sg.site_id = s.id AND member.tenant_id = t.id
                     ))
                 )
                 WHERE i.provider = 'keycloak' AND i.provider_user_id = $1
                   AND d.id = $2 AND t.id = $3 AND d.api_base_url = $4
                   AND d.status = 'active' AND t.status = 'active'
                   AND u.status = 'active' AND NOT u.deletion_pending
                 GROUP BY u.id",
            )
            .bind(&access.subject)
            .bind(self.scope.deployment_id)
            .bind(self.scope.tenant_id)
            .bind(&self.api_url)
            .fetch_optional(&self.db)
            .await
            .map_err(database_unavailable)?
            .ok_or_else(invalid_token)?;
        if !account_permits(&role, grant) {
            return Err(permission_denied());
        }
        let sites = site_ids
            .into_iter()
            .map(i64::from)
            .zip(site_names)
            .collect();
        Ok((Arc::new(sites), Some(LocalAccount { id, username, role })))
    }
}

fn loopback(host: &str) -> bool {
    host.parse::<Authority>().is_ok_and(|authority| {
        let host = authority.host();
        host.eq_ignore_ascii_case("localhost")
            || host
                .trim_matches(['[', ']'])
                .parse::<std::net::IpAddr>()
                .is_ok_and(|ip| ip.is_loopback())
    })
}

fn account_permits(role: &str, grant: Grant) -> bool {
    match grant {
        Grant::NotificationDispatch | Grant::EvidenceWrite => false,
        Grant::AccountManage | Grant::ViolationReview => matches!(role, "admin" | "super_admin"),
        Grant::ModelManage => matches!(role, "admin" | "model_manage"),
        Grant::ModelDownload => role != "guest",
        Grant::AccountRead
        | Grant::EvidenceRead
        | Grant::ImageDetection
        | Grant::NotificationRead
        | Grant::StreamRead
        | Grant::ViolationFeedback
        | Grant::ViolationRead => true,
    }
}
