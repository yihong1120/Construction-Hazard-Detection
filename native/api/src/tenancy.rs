use anyhow::{Context, Result, ensure};
use axum::{
    http::{HeaderValue, StatusCode, header},
    response::Response,
};
use serde::Deserialize;
use sqlx::{PgConnection, PgPool, Postgres};
use uuid::Uuid;

use crate::ApiError;

const SCOPED_ACCOUNT: &str = "SELECT u.id, u.username, u.group_id, u.role FROM user_identities i
    JOIN users u ON i.user_id = u.id
    JOIN tenants t ON t.id = u.tenant_id
    JOIN deployments d ON d.tenant_id = t.id
    WHERE i.provider = 'keycloak' AND i.provider_user_id = $1
      AND t.id = $2 AND d.id = $3 AND u.status = 'active' AND NOT u.deletion_pending
      AND t.status = 'active' AND d.status = 'active'";

#[derive(Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentScope {
    pub tenant_id: Uuid,
    pub deployment_id: Uuid,
}

impl DeploymentScope {
    pub async fn validate(self, db: &PgPool) -> Result<()> {
        ensure!(
            !self.tenant_id.is_nil() && !self.deployment_id.is_nil(),
            "Configure a tenant and deployment"
        );
        ensure!(
            self.active(db)
                .await
                .context("Check the configured deployment")?,
            "Configured deployment is inactive or outside its tenant"
        );
        Ok(())
    }

    pub async fn active(self, db: &PgPool) -> Result<bool, sqlx::Error> {
        sqlx::query_scalar(
            "SELECT EXISTS (SELECT 1 FROM deployments d JOIN tenants t ON t.id = d.tenant_id
             WHERE d.id = $1 AND t.id = $2 AND d.status = 'active' AND t.status = 'active')",
        )
        .bind(self.deployment_id)
        .bind(self.tenant_id)
        .fetch_one(db)
        .await
    }

    pub async fn account<'e>(
        self,
        db: impl sqlx::Executor<'e, Database = Postgres>,
        subject: &str,
    ) -> Result<Account, ApiError> {
        self.load_account(db, subject, SCOPED_ACCOUNT).await
    }

    pub async fn account_for_update(
        self,
        db: &mut PgConnection,
        subject: &str,
    ) -> Result<Account, ApiError> {
        let query = format!("{SCOPED_ACCOUNT} FOR SHARE OF u, i, t, d");
        self.load_account(db, subject, &query).await
    }

    async fn load_account<'e>(
        self,
        db: impl sqlx::Executor<'e, Database = Postgres>,
        subject: &str,
        query: &str,
    ) -> Result<Account, ApiError> {
        let row = sqlx::query_as(query)
            .bind(subject)
            .bind(self.tenant_id)
            .bind(self.deployment_id)
            .fetch_optional(db)
            .await
            .map_err(database_unavailable)?;
        row.ok_or_else(|| ApiError::new(StatusCode::FORBIDDEN, "Account or deployment unavailable"))
    }
}

pub struct Account {
    pub id: i32,
    pub username: String,
    pub group_id: Option<i32>,
    pub role: String,
}

impl<'r> sqlx::FromRow<'r, sqlx::postgres::PgRow> for Account {
    fn from_row(row: &'r sqlx::postgres::PgRow) -> Result<Self, sqlx::Error> {
        use sqlx::Row;
        Ok(Self {
            id: row.try_get("id")?,
            username: row.try_get("username")?,
            group_id: row.try_get("group_id")?,
            role: row.try_get("role")?,
        })
    }
}

pub fn database_unavailable(error: sqlx::Error) -> ApiError {
    tracing::warn!(code = ?error.as_database_error().and_then(|error| error.code()), "domain storage operation failed");
    ApiError::new(
        StatusCode::SERVICE_UNAVAILABLE,
        "Database operation unavailable",
    )
}

pub async fn private_response(mut response: Response) -> Response {
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("private, no-store"),
    );
    response
}
