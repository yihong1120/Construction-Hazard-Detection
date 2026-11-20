//! Site-scoped history and review against the existing PostgreSQL schema.

mod analytics;
mod evidence_upload;
mod feedback;
mod media;
mod projection;
mod query;
mod review;
mod search;

use std::{path::PathBuf, sync::Arc};

use anyhow::{Context, ensure};
use axum::{
    Extension, Router,
    http::{HeaderMap, StatusCode},
    routing::{get, patch, post},
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{PgConnection, PgPool, Postgres, pool::PoolConnection};
use uuid::Uuid;

use crate::{
    ApiError, AppState,
    auth::{Actor, Grant},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub tenant_id: Uuid,
    pub media_root: PathBuf,
    #[serde(default)]
    pub write_uploads: bool,
}

#[derive(Clone)]
struct Settings {
    tenant: Uuid,
    media: Arc<media::Store>,
}

pub(crate) async fn routes(
    config: Config,
    db: &PgPool,
    shutdown: tokio_util::sync::CancellationToken,
) -> anyhow::Result<Router<AppState>> {
    ensure!(
        !config.tenant_id.is_nil(),
        "Set a violation history tenant_id"
    );
    ensure!(
        config.media_root.is_absolute(),
        "Violation media_root must be absolute"
    );
    let media_root = config
        .media_root
        .canonicalize()
        .context("Could not resolve violation media_root")?;
    ensure!(
        media_root.is_dir(),
        "Violation media_root must be a directory"
    );

    // The history API relies on the installed application schema. It never
    // creates or upgrades the production violation tables.
    let missing = sqlx::query_scalar::<_, Option<String>>(
        "SELECT table_name FROM unnest(ARRAY[
             'public.tenants', 'users', 'public.user_identities', 'sites', 'user_sites', 'site_groups',
             'stream_configs', 'violations', 'public.violation_feedback', 'public.violation_review_audit_logs'
         ]) AS tables(table_name)
         WHERE to_regclass(table_name) IS NULL LIMIT 1",
    )
    .fetch_optional(db)
    .await
    .context("Check violation history schema")?
    .flatten();
    ensure!(
        missing.is_none(),
        "Violation history requires the existing application schema"
    );

    let active_tenant = sqlx::query_scalar::<_, bool>(
        "SELECT EXISTS (SELECT 1 FROM public.tenants WHERE id = $1 AND status = 'active')",
    )
    .bind(config.tenant_id)
    .fetch_one(db)
    .await
    .context("Check violation history tenant")?;
    ensure!(active_tenant, "Violation history tenant is unavailable");

    let settings = Settings {
        tenant: config.tenant_id,
        media: Arc::new(media::Store::new(media_root)),
    };
    let router = Router::new()
        .route("/my_sites", get(query::sites))
        .route("/violations", get(query::history))
        .route("/violations/filter-options", get(query::filters))
        .route("/violations/analytics", get(analytics::get))
        .route("/violations/next", get(review::next))
        .route("/violations/{id}", get(query::detail))
        .route("/violations/{id}/audit-log", get(review::audit_log))
        .route("/violations/{id}/review-options", get(review::options))
        .route(
            "/violations/{id}/feedback-options",
            get(review::feedback_options),
        )
        .route("/violations/{id}/feedback", post(feedback::submit))
        .route("/violations/{id}/review", patch(review::transition))
        .route("/get_violation_image", get(media::image))
        .route("/get_violation_thumbnail", get(media::thumbnail));
    let router = if config.write_uploads {
        crate::accounts::start_media_cleanup(db, settings.media.root(), shutdown.clone()).await?;
        evidence_upload::start(db, settings.media.clone(), shutdown).await?;
        router.route(
            "/upload",
            post(evidence_upload::upload).layer(axum::extract::DefaultBodyLimit::max(
                crate::evidence::PRODUCTION_UPLOAD_BODY_LIMIT,
            )),
        )
    } else {
        router
    };
    Ok(router.layer(Extension(settings)))
}

#[derive(Serialize)]
struct Site {
    id: i32,
    name: String,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

struct Scope {
    user: i32,
    role: String,
    group: Option<i32>,
    sites: Vec<Site>,
}

const USER_SQL: &str = "SELECT u.id, u.role, u.group_id FROM public.user_identities AS i
     JOIN users AS u ON u.id = i.user_id
     JOIN public.tenants AS t ON t.id = u.tenant_id
     WHERE i.provider = 'keycloak' AND i.provider_user_id = $1
       AND t.id = $2 AND t.status = 'active' AND u.status = 'active'
       AND NOT u.deletion_pending";

impl Scope {
    async fn load(
        db: &mut PgConnection,
        actor: &Actor,
        tenant: Uuid,
        require_reviewer: bool,
        lock_user: bool,
    ) -> Result<Self, ApiError> {
        let mut statement = USER_SQL.to_owned();
        if lock_user {
            statement.push_str(" FOR SHARE OF i, u, t");
        }
        let (user, role, group) = sqlx::query_as::<_, (i32, String, Option<i32>)>(&statement)
            .bind(&actor.subject)
            .bind(tenant)
            .fetch_optional(&mut *db)
            .await
            .map_err(database_error)?
            .ok_or_else(|| ApiError::new(StatusCode::UNAUTHORIZED, "Account unavailable"))?;
        if require_reviewer && !matches!(role.as_str(), "admin" | "super_admin") {
            return Err(ApiError::new(
                StatusCode::FORBIDDEN,
                "Only admin or super_admin can review violations",
            ));
        }

        // Intersect static API grants with live membership and group policy.
        // Platform operators can read explicitly granted sites without a group.
        let allowed_ids: Vec<i32> = actor
            .sites()
            .iter()
            .filter_map(|(id, _)| i32::try_from(*id).ok())
            .collect();
        let sites = sqlx::query_as::<_, (i32, String, DateTime<Utc>, DateTime<Utc>)>(
            "SELECT s.id, s.name, s.created_at, s.updated_at FROM sites AS s
             WHERE s.id = ANY($1)
             AND ($2 OR (
               EXISTS (SELECT 1 FROM user_sites WHERE user_id = $3 AND site_id = s.id)
               AND EXISTS (SELECT 1 FROM site_groups WHERE group_id = $4 AND site_id = s.id)
             )) ORDER BY s.id",
        )
        .bind(allowed_ids)
        .bind(role == "super_admin")
        .bind(user)
        .bind(group)
        .fetch_all(&mut *db)
        .await
        .map_err(database_error)?
        .into_iter()
        .filter(|(id, name, _, _)| actor.allows_site(i64::from(*id), name))
        .map(|(id, name, created_at, updated_at)| Site {
            id,
            name,
            created_at,
            updated_at,
        })
        .collect();
        Ok(Self {
            user,
            role,
            group,
            sites,
        })
    }

    fn select_sites(&self, site_id: Option<i32>) -> Result<(Vec<i32>, Vec<&str>), ApiError> {
        let sites: Vec<_> = self
            .sites
            .iter()
            .filter(|site| site_id.is_none_or(|id| site.id == id))
            .map(|site| (site.id, site.name.as_str()))
            .collect();
        if site_id.is_some() && sites.is_empty() {
            return Err(ApiError::new(StatusCode::FORBIDDEN, "No access to site_id"));
        }
        Ok(sites.into_iter().unzip())
    }
}

async fn scope(
    state: &AppState,
    settings: &Settings,
    headers: &HeaderMap,
    grant: Grant,
) -> Result<(PoolConnection<Postgres>, Scope), ApiError> {
    let actor = state.auth.authorize(headers, grant).await?;
    let mut db = state.db.acquire().await.map_err(database_error)?;
    let scope = Scope::load(
        &mut db,
        &actor,
        settings.tenant,
        grant == Grant::ViolationReview,
        false,
    )
    .await?;
    Ok((db, scope))
}

fn inaccessible() -> ApiError {
    ApiError::new(StatusCode::FORBIDDEN, "No access to this violation")
}

fn database_error(error: sqlx::Error) -> ApiError {
    tracing::warn!(code = ?error.as_database_error().and_then(|error| error.code()), "violation history database operation failed");
    ApiError::new(
        StatusCode::SERVICE_UNAVAILABLE,
        "Violation history unavailable",
    )
}

fn invalid(detail: &'static str) -> ApiError {
    ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, detail)
}

fn private(headers: axum::response::Response) -> axum::response::Response {
    let mut response = headers;
    response.headers_mut().insert(
        axum::http::header::CACHE_CONTROL,
        "private, no-store".parse().unwrap(),
    );
    response
}
