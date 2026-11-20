use std::time::Duration;

use anyhow::{Result, ensure};
use futures_util::{StreamExt, stream};
use serde_json::{Value, json};
use sqlx::{PgConnection, PgPool};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    ApiError,
    tenancy::{DeploymentScope, database_unavailable},
};

use super::{deletion, keycloak::Keycloak, linked_subject, provision};
use crate::accounts::users_admin::Target;

const SUPPORTED: &[&str] = &["status", "profile", "rename", "provision", "delete"];

#[derive(Clone)]
pub(super) struct Reconciler {
    pub(super) provider: Keycloak,
    pub(super) receipt_days: i32,
}

impl Reconciler {
    pub(super) fn new(provider: Keycloak) -> Result<Self> {
        let receipt_days = quantity(
            "ACCOUNT_OPERATIONS_DELETION_RECEIPT_DAYS",
            365,
            i32::MAX.into(),
        )? as i32;
        Ok(Self {
            provider,
            receipt_days,
        })
    }
}

pub(super) async fn enqueue(
    db: &mut PgConnection,
    operation: &str,
    kind: &str,
    user: i32,
    target: Target,
    subject: Option<&str>,
    revision: i32,
) -> Result<(), ApiError> {
    sqlx::query(
        "INSERT INTO public.account_operations(id,kind,phase,user_id,tenant_id,username,group_id,role,subject,target_revision)
         VALUES ($1,$2,'pending',$3,$4,$5,$6,$7,$8,$9)",
    ).bind(operation).bind(kind).bind(user).bind(target.tenant_id).bind(target.username)
        .bind(target.group_id).bind(target.role).bind(subject).bind(revision)
        .execute(db).await.map_err(database_unavailable)?;
    Ok(())
}

pub(super) async fn start(
    db: &PgPool,
    scope: DeploymentScope,
    reconciler: Reconciler,
    shutdown: CancellationToken,
) -> Result<()> {
    let schema_ready: bool = sqlx::query_scalar(
        "SELECT to_regclass('public.account_operations') IS NOT NULL
            AND to_regclass('public.account_deletion_receipts') IS NOT NULL
            AND to_regclass('public.user_identities') IS NOT NULL
            AND EXISTS (SELECT 1 FROM pg_attribute WHERE attrelid='users'::regclass
                         AND attname='state_revision' AND NOT attisdropped)",
    )
    .fetch_one(db)
    .await?;
    ensure!(
        schema_ready,
        "Account lifecycle requires the installed account operations schema"
    );
    let sync = SyncSettings::from_env()?;
    let cleanup = CleanupSettings::from_env()?;
    tokio::spawn(synchronize(
        db.clone(),
        scope,
        reconciler,
        sync,
        shutdown.clone(),
    ));
    tokio::spawn(cleanup_completed(db.clone(), scope, cleanup, shutdown));
    Ok(())
}

struct SyncSettings {
    interval: Duration,
    budget: Duration,
    batch: i64,
    concurrency: usize,
}

impl SyncSettings {
    fn from_env() -> Result<Self> {
        Ok(Self {
            interval: seconds("ACCOUNT_OPERATIONS_SYNC_INTERVAL_SECONDS", 5.0)?,
            budget: seconds("ACCOUNT_OPERATIONS_SYNC_BUDGET_SECONDS", 30.0)?,
            batch: quantity("ACCOUNT_OPERATIONS_SYNC_BATCH_SIZE", 16, 256)?,
            concurrency: quantity("ACCOUNT_OPERATIONS_SYNC_CONCURRENCY", 2, 8)? as usize,
        })
    }
}

struct CleanupSettings {
    interval: Duration,
    budget: Duration,
    batch: i64,
    completed_days: i32,
}

impl CleanupSettings {
    fn from_env() -> Result<Self> {
        Ok(Self {
            interval: seconds("ACCOUNT_OPERATIONS_CLEANUP_INTERVAL_SECONDS", 60.0)?,
            budget: seconds("ACCOUNT_OPERATIONS_CLEANUP_BUDGET_SECONDS", 2.0)?,
            batch: quantity("ACCOUNT_OPERATIONS_CLEANUP_BATCH_SIZE", 128, 4096)?,
            completed_days: quantity("ACCOUNT_OPERATIONS_COMPLETED_DAYS", 30, i32::MAX.into())?
                as i32,
        })
    }
}

fn seconds(key: &str, fallback: f64) -> Result<Duration> {
    let raw = std::env::var(key)
        .ok()
        .map(|value| value.parse::<f64>())
        .transpose()?
        .unwrap_or(fallback);
    let interval = Duration::try_from_secs_f64(raw)?;
    ensure!(!interval.is_zero(), "Set a positive duration for {key}");
    Ok(interval)
}

fn quantity(key: &str, fallback: i64, maximum: i64) -> Result<i64> {
    let raw = std::env::var(key)
        .ok()
        .map(|value| value.parse::<i64>())
        .transpose()?
        .unwrap_or(fallback);
    ensure!((1..=maximum).contains(&raw), "Invalid quantity for {key}");
    Ok(raw)
}

async fn synchronize(
    db: PgPool,
    scope: DeploymentScope,
    reconciler: Reconciler,
    settings: SyncSettings,
    shutdown: CancellationToken,
) {
    loop {
        tokio::select! {
            _ = shutdown.cancelled() => break,
            _ = tokio::time::sleep(settings.interval) => {}
        }
        tokio::select! {
            _ = shutdown.cancelled() => break,
            result = tokio::time::timeout(settings.budget, sweep(&db, scope, &reconciler, &settings)) => {
                if !matches!(result, Ok(Ok(()))) { tracing::warn!("account operation sweep deferred"); }
            }
        }
    }
}

async fn sweep(
    db: &PgPool,
    scope: DeploymentScope,
    reconciler: &Reconciler,
    settings: &SyncSettings,
) -> Result<(), sqlx::Error> {
    let ids: Vec<String> = sqlx::query_scalar(
        "SELECT id FROM public.account_operations WHERE tenant_id=$1 AND kind=ANY($2)
           AND phase NOT IN ('completed','uncertain') AND next_attempt<=now()
         ORDER BY next_attempt,created_at LIMIT $3",
    )
    .bind(scope.tenant_id)
    .bind(SUPPORTED)
    .bind(settings.batch)
    .fetch_all(db)
    .await?;
    stream::iter(ids)
        .map(|operation| async move {
            process(db, scope, reconciler, &operation, false).await;
        })
        .buffer_unordered(settings.concurrency)
        .for_each(|()| async {})
        .await;
    Ok(())
}

pub(super) async fn process(
    db: &PgPool,
    scope: DeploymentScope,
    reconciler: &Reconciler,
    operation: &str,
    wait: bool,
) -> bool {
    match finish(db, scope, reconciler, operation, wait).await {
        Ok(completed) => completed,
        Err(error) => {
            let reason = match error {
                Failure::Storage(error) => {
                    tracing::warn!(code = ?error.as_database_error().and_then(|error| error.code()), "account operation database step deferred");
                    "AccountDatabaseUnavailable"
                }
                Failure::Provider(error) => {
                    tracing::warn!(status = %error.status(), "account identity update deferred");
                    "KeycloakSyncUnavailable"
                }
                Failure::Scope => "AccountIdentityScopeChanged",
            };
            let updated = sqlx::query(
                "UPDATE public.account_operations SET attempts=attempts+1,last_error=$2,
                 next_attempt=now()+make_interval(secs=>power(2::float8,LEAST(attempts+1,8)))
                 WHERE id=$1 AND tenant_id=$3 AND phase NOT IN ('completed','uncertain')",
            )
            .bind(operation)
            .bind(reason)
            .bind(scope.tenant_id)
            .execute(db)
            .await;
            if updated.is_err() {
                tracing::warn!("account operation retry could not be recorded");
            }
            false
        }
    }
}

pub(super) enum Failure {
    Storage(sqlx::Error),
    Provider(ApiError),
    Scope,
}

impl From<sqlx::Error> for Failure {
    fn from(error: sqlx::Error) -> Self {
        Self::Storage(error)
    }
}

type OperationRow = (
    String,
    String,
    Option<i32>,
    String,
    Option<i32>,
    String,
    Option<String>,
);

pub(super) struct Operation {
    pub(super) user: Option<i32>,
    pub(super) phase: String,
    pub(super) username: String,
    pub(super) group_id: Option<i32>,
    pub(super) role: String,
    pub(super) subject: Option<String>,
}

pub(super) async fn lock_operation(
    db: &mut PgConnection,
    id: &str,
    wait: bool,
) -> Result<bool, sqlx::Error> {
    if wait {
        sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended($1,4102))")
            .bind(id)
            .execute(db)
            .await?;
        Ok(true)
    } else {
        sqlx::query_scalar("SELECT pg_try_advisory_xact_lock(hashtextextended($1,4102))")
            .bind(id)
            .fetch_one(db)
            .await
    }
}

async fn finish(
    db: &PgPool,
    scope: DeploymentScope,
    reconciler: &Reconciler,
    id: &str,
    wait: bool,
) -> Result<bool, Failure> {
    let mut tx = db.begin().await?;
    if !lock_operation(&mut tx, id, wait).await? {
        return Ok(false);
    }
    let row: Option<OperationRow> = sqlx::query_as(
        "SELECT kind,phase,user_id,username,group_id,role,subject FROM public.account_operations
         WHERE id=$1 AND tenant_id=$2 AND ($3 OR next_attempt<=now()) FOR UPDATE",
    )
    .bind(id)
    .bind(scope.tenant_id)
    .bind(wait)
    .fetch_optional(&mut *tx)
    .await?;
    let Some((kind, phase, user, username, group_id, role, subject)) = row else {
        return Ok(true);
    };
    if phase == "completed" {
        return Ok(true);
    }
    if phase == "uncertain" || !SUPPORTED.contains(&kind.as_str()) {
        return Ok(false);
    }
    let operation = Operation {
        phase,
        user,
        username,
        group_id,
        role,
        subject,
    };
    let provider = &reconciler.provider;
    let revision = match kind.as_str() {
        "provision" if operation.subject.is_none() => {
            sqlx::query("UPDATE public.account_operations SET phase='uncertain',last_error='remote_subject_unknown' WHERE id=$1")
                .bind(id).execute(&mut *tx).await?;
            tx.commit().await?;
            return Ok(false);
        }
        "provision" => provision::recover(&mut tx, scope, provider, &operation).await?,
        "delete" => deletion::recover(&mut tx, scope, reconciler, &operation).await?,
        _ => sync_user(&mut tx, scope, provider, &kind, &operation).await?,
    };
    sqlx::query(
        "UPDATE public.account_operations SET phase='completed',target_revision=COALESCE($2,target_revision),
         last_error=NULL,completed_at=clock_timestamp() WHERE id=$1",
    ).bind(id).bind(revision).execute(&mut *tx).await?;
    tx.commit().await?;
    Ok(true)
}

async fn sync_user(
    db: &mut PgConnection,
    scope: DeploymentScope,
    provider: &Keycloak,
    kind: &str,
    operation: &Operation,
) -> Result<Option<i32>, Failure> {
    let target = operation.user.ok_or(Failure::Scope)?;
    let subject = operation.subject.as_deref().ok_or(Failure::Scope)?;
    let user: Option<(Uuid, String, String, bool, i32)> = sqlx::query_as(
        "SELECT tenant_id,username,status,deletion_pending,state_revision FROM users WHERE id=$1 FOR UPDATE NOWAIT",
    ).bind(target).fetch_optional(&mut *db).await?;
    let Some((tenant, name, status, deleting, revision)) = user else {
        return Ok(None);
    };
    if tenant != scope.tenant_id {
        return Err(Failure::Scope);
    }
    if deleting {
        return Ok(Some(revision));
    }
    if linked_subject(db, target)
        .await
        .map_err(Failure::Provider)?
        .as_deref()
        != Some(subject)
    {
        return Err(Failure::Scope);
    }
    let body = match kind {
        "status" => Some(json!({"enabled":status == "active"})),
        "rename" => Some(json!({"username":name})),
        "profile" => profile_payload(db, target).await?,
        _ => return Err(Failure::Scope),
    };
    if let Some(body) = body {
        provider
            .update(subject, body)
            .await
            .map_err(Failure::Provider)?;
    }
    Ok(Some(revision))
}

async fn profile_payload(db: &mut PgConnection, id: i32) -> Result<Option<Value>, sqlx::Error> {
    sqlx::query_scalar(
        "SELECT jsonb_build_object('email',email,'firstName',given_name,'lastName',family_name)
        FROM user_profiles WHERE user_id=$1 FOR SHARE",
    )
    .bind(id)
    .fetch_optional(db)
    .await
}

async fn cleanup_completed(
    db: PgPool,
    scope: DeploymentScope,
    settings: CleanupSettings,
    shutdown: CancellationToken,
) {
    loop {
        tokio::select! {
            _ = shutdown.cancelled() => break,
            _ = tokio::time::sleep(settings.interval) => {}
        }
        tokio::select! {
            _ = shutdown.cancelled() => break,
            result = tokio::time::timeout(settings.budget, prune(&db, scope, &settings)) => {
                if !matches!(result, Ok(Ok(()))) { tracing::warn!("account operation retention deferred"); }
            }
        }
    }
}

async fn prune(
    db: &PgPool,
    scope: DeploymentScope,
    settings: &CleanupSettings,
) -> Result<(), sqlx::Error> {
    let mut tx = db.begin().await?;
    sqlx::query("SET LOCAL statement_timeout='1000ms'")
        .execute(&mut *tx)
        .await?;
    sqlx::query(
        "DELETE FROM public.account_operations WHERE id IN (
           SELECT id FROM public.account_operations WHERE tenant_id=$1 AND phase='completed'
              AND completed_at<now()-make_interval(days=>$2)
           ORDER BY completed_at LIMIT $3 FOR UPDATE SKIP LOCKED)",
    )
    .bind(scope.tenant_id)
    .bind(settings.completed_days)
    .bind(settings.batch)
    .execute(&mut *tx)
    .await?;
    sqlx::query(
        "DELETE FROM public.account_deletion_receipts WHERE user_id IN (
           SELECT user_id FROM public.account_deletion_receipts WHERE tenant_id=$1 AND expires_at<=now()
           ORDER BY expires_at LIMIT $2 FOR UPDATE SKIP LOCKED)",
    ).bind(scope.tenant_id).bind(settings.batch).execute(&mut *tx).await?;
    tx.commit().await?;
    Ok(())
}
