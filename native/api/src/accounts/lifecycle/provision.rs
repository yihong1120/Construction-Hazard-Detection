use axum::{
    Extension, Json,
    extract::State,
    http::{HeaderMap, StatusCode},
};
use serde::Deserialize;
use serde_json::{Value, json};
use sqlx::{PgConnection, PgPool};
use uuid::Uuid;

use crate::{
    ApiError, AppState,
    accounts::{
        SUPER_ADMIN_USERNAME, USER_JSON, admin_group, sites::require_group, users_admin::roles_for,
    },
    auth::Grant,
    tenancy::{DeploymentScope, database_unavailable},
};

use super::{
    Lifecycle, RECONCILE, Reply,
    journal::{self, Failure, Operation, Reconciler},
    keycloak::Keycloak,
    normalise_email, password_policy, validate_username,
};

const PROFILE_REQUIRED: ApiError = ApiError::new(
    StatusCode::BAD_REQUEST,
    "A profile with email and name is required for Keycloak.",
);
const PERSISTENCE: ApiError = ApiError::new(
    StatusCode::INTERNAL_SERVER_ERROR,
    "user_identity_persistence_failed",
);
const DUPLICATE: ApiError = ApiError::new(StatusCode::CONFLICT, "username_or_email_already_exists");

#[derive(Deserialize)]
pub(super) struct CreateUser {
    username: String,
    password: String,
    #[serde(default = "regular_role")]
    role: String,
    group_id: Option<i32>,
    profile: Option<Profile>,
    #[serde(default = "force_password_change")]
    force_password_change: bool,
}

fn regular_role() -> String {
    "user".to_owned()
}
fn force_password_change() -> bool {
    true
}

#[derive(Deserialize)]
struct Profile {
    family_name: String,
    middle_name: Option<String>,
    given_name: String,
    email: String,
    mobile_number: Option<String>,
}

impl Profile {
    fn normalise(&mut self) -> Result<(), ApiError> {
        for (value, maximum) in [
            (Some(self.family_name.as_str()), 50),
            (Some(self.given_name.as_str()), 50),
            (self.middle_name.as_deref(), 50),
            (self.mobile_number.as_deref(), 20),
        ] {
            if value.is_some_and(|value| {
                value.chars().count() > maximum || value.chars().any(char::is_control)
            }) {
                return Err(ApiError::new(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "Invalid account profile.",
                ));
            }
        }
        self.email = normalise_email(&self.email).ok_or_else(|| {
            ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, "Invalid email address.")
        })?;
        Ok(())
    }
}

fn allowed_group(
    operator_group: Option<i32>,
    requested: Option<i32>,
    role: &str,
) -> Result<Option<i32>, ApiError> {
    if !roles_for(operator_group, None).contains(&role) {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "Role assignment is not allowed",
        ));
    }
    if operator_group.is_some() && requested.is_some() && operator_group != requested {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "Cannot operate on other group",
        ));
    }
    Ok(requested.or(operator_group))
}

struct Provisioning {
    operation: String,
    operator: String,
    requested_group: Option<i32>,
    group: Option<i32>,
    username: String,
    password: String,
    role: String,
    profile: Profile,
    temporary_password: bool,
}

pub(super) async fn add_user(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    Extension(runtime): Extension<Lifecycle>,
    headers: HeaderMap,
    Json(body): Json<CreateUser>,
) -> Reply {
    let actor = state.auth.authorize(&headers, Grant::AccountManage).await?;
    let reconciler = runtime.reconciler()?.clone();
    let mut tx = state.db.begin().await.map_err(database_unavailable)?;
    let account = scope.account_for_update(&mut tx, &actor.subject).await?;
    let group = allowed_group(admin_group(&account)?, body.group_id, &body.role)?;
    let mut profile = body.profile.ok_or(PROFILE_REQUIRED)?;
    validate_username(&body.username)?;
    password_policy(&body.password)?;
    profile.normalise()?;
    if let Some(group) = group {
        require_group(&mut tx, group).await?;
    }
    let operation = Uuid::new_v4().simple().to_string();
    sqlx::query(
        "INSERT INTO public.account_operations(id,kind,phase,tenant_id,username,role,group_id,next_attempt)
         VALUES($1,'provision','creating',$2,$3,$4,$5,now()+interval '30 seconds')",
    ).bind(&operation).bind(scope.tenant_id).bind(&body.username).bind(&body.role).bind(group)
        .execute(&mut *tx).await.map_err(database_unavailable)?;
    tx.commit().await.map_err(database_unavailable)?;

    let pending = Provisioning {
        operation,
        operator: actor.subject,
        requested_group: body.group_id,
        group,
        username: body.username,
        password: body.password,
        role: body.role,
        profile,
        temporary_password: body.force_password_change,
    };
    // Once remote creation starts, finish recording its result even when the
    // HTTP client disconnects. The journal covers process crashes as well.
    tokio::spawn(provision_account(state.db, scope, reconciler, pending))
        .await
        .map_err(|_| RECONCILE)?
}

async fn provision_account(
    db: PgPool,
    scope: DeploymentScope,
    reconciler: Reconciler,
    pending: Provisioning,
) -> Reply {
    let subject = create_remote(&db, scope, &reconciler.provider, &pending).await?;
    let operation = pending.operation.clone();
    let result = adopt(&db, scope, &reconciler.provider, pending, &subject).await;
    let synchronized = journal::process(&db, scope, &reconciler, &operation, true).await;
    let user = result?;
    if !synchronized {
        return Err(RECONCILE);
    }
    Ok(Json(user))
}

async fn lock_phase(
    db: &mut PgConnection,
    scope: DeploymentScope,
    operation: &str,
    phase: &str,
) -> Result<bool, sqlx::Error> {
    journal::lock_operation(db, operation, true).await?;
    let found: Option<String> = sqlx::query_scalar(
        "SELECT id FROM public.account_operations WHERE id=$1 AND tenant_id=$2 AND kind='provision' AND phase=$3 FOR UPDATE",
    ).bind(operation).bind(scope.tenant_id).bind(phase).fetch_optional(db).await?;
    Ok(found.is_some())
}

async fn create_remote(
    db: &PgPool,
    scope: DeploymentScope,
    provider: &Keycloak,
    pending: &Provisioning,
) -> Result<String, super::ResourceError> {
    let mut tx = db.begin().await.map_err(database_unavailable)?;
    if !lock_phase(&mut tx, scope, &pending.operation, "creating")
        .await
        .map_err(database_unavailable)?
    {
        return Err(RECONCILE);
    }
    let new_identity = json!({
        "username": pending.username, "email": pending.profile.email, "emailVerified": true,
        "firstName": pending.profile.given_name, "lastName": pending.profile.family_name,
        "enabled": false, "attributes": {"visionnaire_operation_id": [pending.operation]},
    });
    let subject = match provider.create(new_identity).await {
        Ok(subject) => subject,
        Err(error) => {
            let rejected = matches!(
                error.status(),
                StatusCode::BAD_REQUEST | StatusCode::CONFLICT
            );
            sqlx::query(
                "UPDATE public.account_operations SET phase=$2,last_error=$3,
                  completed_at=CASE WHEN $2='completed' THEN clock_timestamp() END WHERE id=$1",
            )
            .bind(&pending.operation)
            .bind(if rejected { "completed" } else { "uncertain" })
            .bind(if rejected {
                "remote_create_rejected"
            } else {
                "remote_create_uncertain"
            })
            .execute(&mut *tx)
            .await
            .map_err(database_unavailable)?;
            tx.commit().await.map_err(database_unavailable)?;
            return Err(error.into());
        }
    };
    let saved = sqlx::query(
        "UPDATE public.account_operations SET subject=$2,phase='remote_created',next_attempt=clock_timestamp()+interval '30 seconds' WHERE id=$1",
    ).bind(&pending.operation).bind(&subject).execute(&mut *tx).await;
    if saved.is_err() || tx.commit().await.is_err() {
        tracing::error!(operation=%pending.operation, subject=%subject, "Account subject requires recovery");
        return Err(RECONCILE);
    }
    Ok(subject)
}

async fn adopt(
    db: &PgPool,
    scope: DeploymentScope,
    provider: &Keycloak,
    pending: Provisioning,
    subject: &str,
) -> Result<Value, super::ResourceError> {
    let mut tx = db.begin().await.map_err(database_unavailable)?;
    if !lock_phase(&mut tx, scope, &pending.operation, "remote_created")
        .await
        .map_err(database_unavailable)?
    {
        return Err(RECONCILE);
    }
    let account = scope.account_for_update(&mut tx, &pending.operator).await?;
    let group = allowed_group(
        admin_group(&account)?,
        pending.requested_group,
        &pending.role,
    )?;
    if group != pending.group {
        return Err(ApiError::new(StatusCode::FORBIDDEN, "Cannot operate on other group").into());
    }
    if let Some(group) = group {
        require_group(&mut tx, group).await?;
    }
    let existing = identity_owner(&mut tx, subject)
        .await
        .map_err(database_unavailable)?;
    if existing.is_some() {
        return Err(DUPLICATE.into());
    }
    provider
        .reset_password(subject, &pending.password, pending.temporary_password)
        .await?;
    drop(pending.password);

    let (id, revision): (i32, i32) = sqlx::query_as(
        "INSERT INTO users(username,role,group_id,tenant_id,status,created_at,updated_at)
         VALUES($1,$2,$3,$4,'active',now(),now()) RETURNING id,state_revision",
    )
    .bind(&pending.username)
    .bind(&pending.role)
    .bind(group)
    .bind(scope.tenant_id)
    .fetch_one(&mut *tx)
    .await
    .map_err(local_write_error)?;
    let profile = pending.profile;
    sqlx::query(
        "INSERT INTO user_profiles(user_id,family_name,middle_name,given_name,email,mobile_number,created_at,updated_at)
         VALUES($1,$2,$3,$4,$5,$6,now(),now())",
    ).bind(id).bind(&profile.family_name).bind(&profile.middle_name).bind(&profile.given_name)
        .bind(&profile.email).bind(&profile.mobile_number).execute(&mut *tx).await.map_err(local_write_error)?;
    let display_name = format!("{} {}", profile.family_name, profile.given_name);
    sqlx::query(
        "INSERT INTO public.user_identities(user_id,provider,provider_user_id,email,email_verified,display_name,linked_at,updated_at)
         VALUES($1,'keycloak',$2,$3,true,$4,now(),now())",
    ).bind(id).bind(subject).bind(&profile.email).bind(display_name.trim())
        .execute(&mut *tx).await.map_err(local_write_error)?;
    if let Some(group) = group {
        sqlx::query(
            "INSERT INTO site_notification_preferences(user_id,site_id,is_enabled,updated_at)
             SELECT $1,site_id,true,now() FROM site_groups WHERE group_id=$2 ON CONFLICT DO NOTHING",
        ).bind(id).bind(group).execute(&mut *tx).await.map_err(local_write_error)?;
    }
    sqlx::query(
        "UPDATE public.account_operations SET user_id=$2,target_revision=$3,group_id=$4,phase='adopted',next_attempt=now() WHERE id=$1",
    ).bind(pending.operation).bind(id).bind(revision).bind(group).execute(&mut *tx).await.map_err(local_write_error)?;
    let query = format!(
        "SELECT {USER_JSON} FROM users u LEFT JOIN group_info g ON g.id=u.group_id
                        LEFT JOIN user_profiles p ON p.user_id=u.id WHERE u.id=$1"
    );
    let user = sqlx::query_scalar(&query)
        .bind(id)
        .fetch_one(&mut *tx)
        .await
        .map_err(local_write_error)?;
    tx.commit().await.map_err(local_write_error)?;
    Ok(user)
}

fn local_write_error(error: sqlx::Error) -> ApiError {
    if error
        .as_database_error()
        .is_some_and(|error| error.is_unique_violation())
    {
        return DUPLICATE;
    }
    tracing::warn!(code=?error.as_database_error().and_then(|error| error.code()), "New account persistence failed");
    PERSISTENCE
}

async fn identity_owner(db: &mut PgConnection, subject: &str) -> Result<Option<i32>, sqlx::Error> {
    sqlx::query_scalar("SELECT user_id FROM public.user_identities WHERE provider='keycloak' AND provider_user_id=$1 FOR SHARE NOWAIT")
        .bind(subject).fetch_optional(db).await
}

pub(super) async fn recover(
    db: &mut PgConnection,
    scope: DeploymentScope,
    provider: &Keycloak,
    operation: &Operation,
) -> Result<Option<i32>, Failure> {
    let subject = operation.subject.as_deref().ok_or(Failure::Scope)?;
    if operation
        .username
        .eq_ignore_ascii_case(SUPER_ADMIN_USERNAME)
    {
        return Err(Failure::Scope);
    }
    match operation.phase.as_str() {
        "remote_created" => {
            if identity_owner(db, subject).await?.is_some() {
                return Err(Failure::Scope);
            }
        }
        "adopted" => {
            let user = operation.user.ok_or(Failure::Scope)?;
            let account: Option<(Uuid, String, String, bool, i32)> = sqlx::query_as(
                "SELECT tenant_id,username,status,deletion_pending,state_revision FROM users WHERE id=$1 FOR UPDATE NOWAIT",
            ).bind(user).fetch_optional(&mut *db).await?;
            let owner = identity_owner(db, subject).await?;
            if let Some((tenant, username, status, deleting, revision)) = account {
                if tenant != scope.tenant_id
                    || username.eq_ignore_ascii_case(SUPER_ADMIN_USERNAME)
                    || owner != Some(user)
                {
                    return Err(Failure::Scope);
                }
                provider
                    .update(subject, json!({"enabled": status == "active" && !deleting}))
                    .await
                    .map_err(Failure::Provider)?;
                return Ok(Some(revision));
            }
            if owner.is_some() {
                return Err(Failure::Scope);
            }
        }
        _ => return Err(Failure::Scope),
    }
    provider.delete(subject).await.map_err(Failure::Provider)?;
    Ok(None)
}
