use axum::{
    Extension, Json,
    extract::State,
    http::{HeaderMap, StatusCode},
};
use serde::Deserialize;
use serde_json::json;
use sqlx::PgConnection;
use uuid::Uuid;

use crate::{
    ApiError, AppState,
    accounts::{
        SUPER_ADMIN_USERNAME, admin_group,
        users_admin::{Target, can_manage_target_scope, find_user_target},
    },
    auth::Grant,
    tenancy::{DeploymentScope, database_unavailable},
};

use super::{
    Lifecycle, RECONCILE, Reply,
    journal::{self, Failure, Operation, Reconciler},
    linked_subject,
};
use crate::accounts::models::ResourceError;

const NOT_FOUND: ApiError = ApiError::new(StatusCode::NOT_FOUND, "User not found.");
const DELETION_PENDING: ResourceError =
    ResourceError::Code(StatusCode::SERVICE_UNAVAILABLE, "account_deletion_pending");

#[derive(Deserialize)]
pub(super) struct DeleteUser {
    user_id: i32,
}

pub(super) async fn delete_user(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    Extension(runtime): Extension<Lifecycle>,
    headers: HeaderMap,
    Json(body): Json<DeleteUser>,
) -> Reply {
    let actor = state.auth.authorize(&headers, Grant::AccountManage).await?;
    let reconciler = runtime.reconciler()?;
    let mut tx = state.db.begin().await.map_err(database_unavailable)?;
    let operator = scope.account_for_update(&mut tx, &actor.subject).await?;
    let group = admin_group(&operator)?;
    let user = body.user_id;
    let operation = if let Some(target) = find_user_target(&mut tx, user, true).await? {
        can_manage_target_scope(group, scope, &target)?;
        if target.deletion_pending {
            let existing = history(&mut tx, user)
                .await?
                .filter(|row| row.phase != "completed")
                .ok_or(RECONCILE)?;
            can_manage_target_scope(group, scope, &existing.scope)?;
            existing.operation
        } else {
            let subject = linked_subject(&mut tx, user).await?;
            let revision = sqlx::query_scalar(
                "UPDATE users SET status='suspended',deletion_pending=true,state_revision=state_revision+1,updated_at=now()
                 WHERE id=$1 RETURNING state_revision",
            ).bind(user).fetch_one(&mut *tx).await.map_err(database_unavailable)?;
            let operation = Uuid::new_v4().simple().to_string();
            journal::enqueue(
                &mut tx,
                &operation,
                "delete",
                user,
                target,
                subject.as_deref(),
                revision,
            )
            .await?;
            operation
        }
    } else if let Some(existing) = history(&mut tx, user).await? {
        can_manage_target_scope(group, scope, &existing.scope)?;
        if existing.phase == "completed" {
            return deleted();
        }
        existing.operation
    } else {
        let receipt = receipt_scope(&mut tx, user).await?.ok_or(NOT_FOUND)?;
        can_manage_target_scope(group, scope, &receipt)?;
        return deleted();
    };
    tx.commit().await.map_err(database_unavailable)?;
    if !journal::process(&state.db, scope, reconciler, &operation, true).await {
        return Err(DELETION_PENDING);
    }
    deleted()
}

fn deleted() -> Reply {
    Ok(Json(json!({"message":"User deleted successfully."})))
}

struct History {
    operation: String,
    phase: String,
    scope: Target,
}

async fn history(db: &mut PgConnection, user: i32) -> Result<Option<History>, ApiError> {
    type HistoryRow = (String, String, String, String, Option<i32>, Uuid);
    let row: Option<HistoryRow> = sqlx::query_as(
        "SELECT id,phase,username,role,group_id,tenant_id FROM public.account_operations
         WHERE user_id=$1 AND kind='delete' ORDER BY created_at DESC,id DESC LIMIT 1",
    )
    .bind(user)
    .fetch_optional(db)
    .await
    .map_err(database_unavailable)?;
    Ok(row.map(
        |(operation, phase, username, role, group_id, tenant_id)| History {
            operation,
            phase,
            scope: Target {
                username,
                role,
                group_id,
                tenant_id,
                deletion_pending: false,
            },
        },
    ))
}

async fn receipt_scope(db: &mut PgConnection, user: i32) -> Result<Option<Target>, ApiError> {
    let row = sqlx::query_as(
        "SELECT username,role,group_id,tenant_id FROM public.account_deletion_receipts WHERE user_id=$1 AND expires_at>now()",
    ).bind(user).fetch_optional(db).await.map_err(database_unavailable)?;
    Ok(row.map(|(username, role, group_id, tenant_id)| Target {
        username,
        role,
        group_id,
        tenant_id,
        deletion_pending: false,
    }))
}

pub(super) async fn recover(
    db: &mut PgConnection,
    scope: DeploymentScope,
    reconciler: &Reconciler,
    operation: &Operation,
) -> Result<Option<i32>, Failure> {
    let user = operation.user.ok_or(Failure::Scope)?;
    if operation
        .username
        .eq_ignore_ascii_case(SUPER_ADMIN_USERNAME)
    {
        return Err(Failure::Scope);
    }
    let target: Option<(Uuid, String, bool, i32)> = sqlx::query_as(
        "SELECT tenant_id,username,deletion_pending,state_revision FROM users WHERE id=$1 FOR UPDATE NOWAIT",
    ).bind(user).fetch_optional(&mut *db).await?;
    if let Some((tenant, username, pending, _)) = &target {
        if *tenant != scope.tenant_id
            || !pending
            || username.eq_ignore_ascii_case(SUPER_ADMIN_USERNAME)
            || linked_subject(db, user).await.map_err(Failure::Provider)? != operation.subject
        {
            return Err(Failure::Scope);
        }
    }
    if let Some(subject) = &operation.subject {
        let owner: Option<i32> = sqlx::query_scalar(
            "SELECT user_id FROM public.user_identities WHERE provider='keycloak' AND provider_user_id=$1 FOR SHARE NOWAIT",
        ).bind(subject).fetch_optional(&mut *db).await?;
        if owner.is_some_and(|owner| owner != user) {
            return Err(Failure::Scope);
        }
        reconciler
            .provider
            .delete(subject)
            .await
            .map_err(Failure::Provider)?;
    }
    if target.is_some() {
        // Existing releases used ORM cascades for these associations. Some
        // installed schemas lack equivalent database cascades.
        sqlx::query("DELETE FROM user_sites WHERE user_id=$1")
            .bind(user)
            .execute(&mut *db)
            .await?;
        sqlx::query("DELETE FROM user_profiles WHERE user_id=$1")
            .bind(user)
            .execute(&mut *db)
            .await?;
        sqlx::query("DELETE FROM users WHERE id=$1")
            .bind(user)
            .execute(&mut *db)
            .await?;
    }
    sqlx::query(
        "INSERT INTO public.account_deletion_receipts(user_id,tenant_id,group_id,username,role,completed_at,expires_at)
         VALUES ($1,$2,$3,$4,$5,clock_timestamp(),clock_timestamp()+make_interval(days=>$6))
         ON CONFLICT(user_id) DO UPDATE SET tenant_id=EXCLUDED.tenant_id,group_id=EXCLUDED.group_id,
           username=EXCLUDED.username,role=EXCLUDED.role,completed_at=EXCLUDED.completed_at,expires_at=EXCLUDED.expires_at",
    ).bind(user).bind(scope.tenant_id).bind(operation.group_id).bind(&operation.username).bind(&operation.role)
        .bind(reconciler.receipt_days).execute(db).await?;
    Ok(target.map(|(_, _, _, revision)| revision))
}
