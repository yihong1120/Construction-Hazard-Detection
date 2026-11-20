use axum::{
    Extension, Json, Router,
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    routing::{delete, post, put},
};
use serde::Deserialize;
use serde_json::{Value, json};
use sqlx::{PgConnection, PgPool, Postgres, Transaction};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    ApiError, AppState,
    auth::Grant,
    tenancy::{DeploymentScope, database_unavailable},
};

use super::{
    SUPER_ADMIN_USERNAME, admin_group,
    models::ResourceError,
    users_admin::{Target, can_manage_target, user_target},
};

mod deletion;
mod journal;
pub(super) mod keycloak;
mod provision;

type Reply = Result<Json<Value>, ResourceError>;

const NO_PROVIDER: ApiError = ApiError::new(
    StatusCode::SERVICE_UNAVAILABLE,
    "keycloak_identity_service_unavailable",
);
const IDENTITY_MISSING: ApiError =
    ApiError::new(StatusCode::CONFLICT, "keycloak_identity_not_linked");
const RECONCILE: ResourceError = ResourceError::Code(
    StatusCode::SERVICE_UNAVAILABLE,
    "account_reconciliation_required",
);
const MIN_PASSWORD_LENGTH: usize = 8;

#[derive(Clone)]
pub(super) struct Lifecycle(Option<journal::Reconciler>);

impl Lifecycle {
    pub(super) async fn start(
        db: &PgPool,
        scope: DeploymentScope,
        shutdown: CancellationToken,
        run_worker: bool,
    ) -> anyhow::Result<Self> {
        let reconciler = keycloak::Keycloak::from_env()?
            .map(journal::Reconciler::new)
            .transpose()?;
        if run_worker && let Some(worker) = &reconciler {
            journal::start(db, scope, worker.clone(), shutdown).await?;
        }
        Ok(Self(reconciler))
    }

    pub(super) fn available(&self) -> bool {
        self.0.is_some()
    }

    fn provider(&self) -> Result<&keycloak::Keycloak, ApiError> {
        Ok(&self.reconciler()?.provider)
    }

    fn reconciler(&self) -> Result<&journal::Reconciler, ApiError> {
        self.0.as_ref().ok_or(NO_PROVIDER)
    }
}

pub(super) fn routes() -> Router<AppState> {
    Router::new()
        .route("/add_user", post(provision::add_user))
        .route("/delete_user", delete(deletion::delete_user))
        .route("/set_user_status", put(set_status))
        .route("/update_user_profile", put(update_profile))
        .route("/update_username", put(rename_by_name))
        .route("/update_username_id", put(rename_by_id))
        .route("/admin_update_password", put(password_by_name))
        .route("/admin_update_password_userid", put(password_by_id))
}

enum UserSelection {
    Id(i32),
    Username(String),
}

async fn locked_target<'a>(
    state: &'a AppState,
    scope: DeploymentScope,
    headers: &HeaderMap,
    selected: UserSelection,
) -> Result<(Transaction<'a, Postgres>, i32, Target), ApiError> {
    let actor = state.auth.authorize(headers, Grant::AccountManage).await?;
    let mut tx = state.db.begin().await.map_err(database_unavailable)?;
    let account = scope.account_for_update(&mut tx, &actor.subject).await?;
    let group = admin_group(&account)?;
    let id = match selected {
        UserSelection::Id(id) => id,
        UserSelection::Username(username) => {
            sqlx::query_scalar("SELECT id FROM users WHERE tenant_id=$1 AND username=$2 FOR UPDATE")
                .bind(scope.tenant_id)
                .bind(username)
                .fetch_optional(&mut *tx)
                .await
                .map_err(database_unavailable)?
                .ok_or_else(|| ApiError::new(StatusCode::NOT_FOUND, "User not found."))?
        }
    };
    let target = user_target(&mut tx, id, true).await?;
    can_manage_target(group, scope, &target)?;
    Ok((tx, id, target))
}

async fn linked_subject(db: &mut PgConnection, user: i32) -> Result<Option<String>, ApiError> {
    let subject: Option<String> = sqlx::query_scalar(
        "SELECT provider_user_id FROM public.user_identities WHERE user_id=$1 AND provider='keycloak' FOR SHARE",
    ).bind(user).fetch_optional(db).await.map_err(database_unavailable)?;
    Ok(subject.filter(|subject| !subject.trim().is_empty()))
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum AccountStatus {
    Active,
    EmailUnverified,
    PendingAdminApproval,
    Rejected,
    Suspended,
}

impl AccountStatus {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::EmailUnverified => "email_unverified",
            Self::PendingAdminApproval => "pending_admin_approval",
            Self::Rejected => "rejected",
            Self::Suspended => "suspended",
        }
    }
}

#[derive(Deserialize)]
struct StatusInput {
    user_id: i32,
    status: AccountStatus,
}

#[derive(Deserialize)]
struct ProfileInput {
    user_id: i32,
    family_name: Option<String>,
    middle_name: Option<String>,
    given_name: Option<String>,
    email: Option<String>,
    mobile_number: Option<String>,
}

impl ProfileInput {
    fn normalise(&mut self) -> Result<(), ApiError> {
        for (value, maximum) in [
            (&self.family_name, 50),
            (&self.middle_name, 50),
            (&self.given_name, 50),
            (&self.mobile_number, 20),
        ] {
            if value.as_ref().is_some_and(|value| {
                value.chars().count() > maximum || value.chars().any(char::is_control)
            }) {
                return Err(ApiError::new(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "Invalid account profile.",
                ));
            }
        }
        if let Some(email) = &mut self.email {
            *email = normalise_email(email).ok_or_else(|| {
                ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, "Invalid email address.")
            })?;
        }
        Ok(())
    }

    fn has_changes(&self) -> bool {
        [
            &self.family_name,
            &self.middle_name,
            &self.given_name,
            &self.email,
            &self.mobile_number,
        ]
        .into_iter()
        .any(Option::is_some)
    }
}

fn normalise_email(input: &str) -> Option<String> {
    let input = input.trim();
    let (local, domain) = input.rsplit_once('@')?;
    if input.len() > 254
        || local.is_empty()
        || local.len() > 64
        || local.starts_with('.')
        || local.ends_with('.')
        || local.contains("..")
        || local
            .chars()
            .any(|ch| ch.is_control() || ch.is_whitespace() || "@<>()[]:;\\,\"".contains(ch))
        || domain.ends_with('.')
    {
        return None;
    }
    let url::Host::Domain(domain) = url::Host::parse(domain).ok()? else {
        return None;
    };
    if !domain.contains('.')
        || domain.split('.').any(|part| {
            part.is_empty()
                || part.len() > 63
                || part.starts_with('-')
                || part.ends_with('-')
                || !part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        })
    {
        return None;
    }
    Some(format!("{local}@{domain}"))
}

#[derive(Deserialize)]
struct RenameByName {
    old_username: String,
    new_username: String,
}

#[derive(Deserialize)]
struct RenameById {
    user_id: i32,
    new_username: String,
}

enum Mutation {
    Status(AccountStatus),
    Profile(ProfileInput),
    Rename(String),
}

impl Mutation {
    fn kind(&self) -> &'static str {
        match self {
            Self::Status(_) => "status",
            Self::Profile(_) => "profile",
            Self::Rename(_) => "rename",
        }
    }

    fn message(&self) -> &'static str {
        match self {
            Self::Status(_) => "User status updated successfully.",
            Self::Profile(_) => "User profile updated successfully.",
            Self::Rename(_) => "Username updated successfully.",
        }
    }

    fn validate(&mut self) -> Result<(), ApiError> {
        match self {
            Self::Profile(profile) => profile.normalise(),
            Self::Rename(name) => validate_username(name),
            _ => Ok(()),
        }
    }

    async fn apply(self, db: &mut PgConnection, id: i32) -> Result<i32, ApiError> {
        match self {
            Self::Status(status) => sqlx::query_scalar(
                "UPDATE users SET status=$2,state_revision=state_revision+1,updated_at=now() WHERE id=$1 RETURNING state_revision",
            ).bind(id).bind(status.as_str()).fetch_one(db).await.map_err(database_unavailable),
            Self::Rename(username) => sqlx::query_scalar(
                "UPDATE users SET username=$2,state_revision=state_revision+1,updated_at=now() WHERE id=$1 RETURNING state_revision",
            ).bind(id).bind(username).fetch_one(db).await.map_err(|error| constraint(error, "Username already exists.")),
            Self::Profile(profile) => {
                let updated = sqlx::query(
                    "UPDATE user_profiles SET family_name=COALESCE($2,family_name), middle_name=COALESCE($3,middle_name),
                        given_name=COALESCE($4,given_name), email=COALESCE($5,email), mobile_number=COALESCE($6,mobile_number),
                        updated_at=now() WHERE user_id=$1",
                ).bind(id).bind(profile.family_name).bind(profile.middle_name).bind(profile.given_name)
                    .bind(profile.email).bind(profile.mobile_number).execute(&mut *db).await
                    .map_err(|error| constraint(error, "Duplicate email or mobile number."))?;
                if updated.rows_affected() == 0 {
                    return Err(ApiError::new(StatusCode::NOT_FOUND, "Profile not found."));
                }
                sqlx::query_scalar("UPDATE users SET state_revision=state_revision+1 WHERE id=$1 RETURNING state_revision")
                    .bind(id).fetch_one(db).await.map_err(database_unavailable)
            }
        }
    }
}

fn validate_username(username: &str) -> Result<(), ApiError> {
    if username.eq_ignore_ascii_case(SUPER_ADMIN_USERNAME) {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "Username already exists.",
        ));
    }
    if username.trim().is_empty()
        || username.chars().count() > 80
        || username.chars().any(char::is_control)
    {
        return Err(ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "Invalid username.",
        ));
    }
    Ok(())
}

fn constraint(error: sqlx::Error, detail: &'static str) -> ApiError {
    if error
        .as_database_error()
        .is_some_and(|error| error.is_unique_violation())
    {
        ApiError::new(StatusCode::BAD_REQUEST, detail)
    } else {
        database_unavailable(error)
    }
}

async fn mutate(
    state: &AppState,
    scope: DeploymentScope,
    lifecycle: &Lifecycle,
    headers: &HeaderMap,
    selected: UserSelection,
    mut mutation: Mutation,
) -> Reply {
    let (mut tx, id, target) = locked_target(state, scope, headers, selected).await?;
    let message = mutation.message();
    mutation.validate()?;
    if matches!(&mutation, Mutation::Profile(profile) if !profile.has_changes()) {
        return Ok(Json(json!({"message":message})));
    }
    let reconciler = lifecycle.reconciler()?;
    let subject = linked_subject(&mut tx, id).await?.ok_or(IDENTITY_MISSING)?;
    let operation = Uuid::new_v4().simple().to_string();
    let kind = mutation.kind();
    let revision = mutation.apply(&mut tx, id).await?;
    journal::enqueue(
        &mut tx,
        &operation,
        kind,
        id,
        target,
        Some(&subject),
        revision,
    )
    .await?;
    tx.commit().await.map_err(database_unavailable)?;
    if !journal::process(&state.db, scope, reconciler, &operation, true).await {
        return Err(RECONCILE);
    }
    Ok(Json(json!({"message":message})))
}

async fn set_status(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    Extension(runtime): Extension<Lifecycle>,
    headers: HeaderMap,
    Json(body): Json<StatusInput>,
) -> Reply {
    mutate(
        &state,
        scope,
        &runtime,
        &headers,
        UserSelection::Id(body.user_id),
        Mutation::Status(body.status),
    )
    .await
}

async fn update_profile(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    Extension(runtime): Extension<Lifecycle>,
    headers: HeaderMap,
    Json(body): Json<ProfileInput>,
) -> Reply {
    mutate(
        &state,
        scope,
        &runtime,
        &headers,
        UserSelection::Id(body.user_id),
        Mutation::Profile(body),
    )
    .await
}

async fn rename_by_name(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    Extension(runtime): Extension<Lifecycle>,
    headers: HeaderMap,
    Json(body): Json<RenameByName>,
) -> Reply {
    mutate(
        &state,
        scope,
        &runtime,
        &headers,
        UserSelection::Username(body.old_username),
        Mutation::Rename(body.new_username),
    )
    .await
}

async fn rename_by_id(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    Extension(runtime): Extension<Lifecycle>,
    headers: HeaderMap,
    Json(body): Json<RenameById>,
) -> Reply {
    mutate(
        &state,
        scope,
        &runtime,
        &headers,
        UserSelection::Id(body.user_id),
        Mutation::Rename(body.new_username),
    )
    .await
}

#[derive(Deserialize)]
struct PasswordByName {
    username: String,
    new_password: String,
}

#[derive(Deserialize)]
struct PasswordById {
    user_id: i32,
    new_password: String,
}

#[derive(Deserialize)]
struct PasswordQuery {
    temporary: Option<String>,
}

impl PasswordQuery {
    fn temporary(&self) -> Result<bool, ApiError> {
        let Some(value) = &self.temporary else {
            return Ok(true);
        };
        match value.to_ascii_lowercase().as_str() {
            "true" | "t" | "yes" | "y" | "on" | "1" => Ok(true),
            "false" | "f" | "no" | "n" | "off" | "0" => Ok(false),
            _ => Err(ApiError::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "Invalid temporary password setting.",
            )),
        }
    }
}

fn password_policy(password: &str) -> Result<(), ResourceError> {
    if password.chars().count() < MIN_PASSWORD_LENGTH {
        return Err(ResourceError::Detail(
            StatusCode::BAD_REQUEST,
            json!({
                "code":"password_too_short", "message":"Password is too short.", "min_length":MIN_PASSWORD_LENGTH
            }),
        ));
    }
    Ok(())
}

async fn reset_password(
    state: &AppState,
    scope: DeploymentScope,
    lifecycle: &Lifecycle,
    headers: &HeaderMap,
    selected: UserSelection,
    password: &str,
    temporary: bool,
) -> Reply {
    let by_id = matches!(selected, UserSelection::Id(_));
    let (mut tx, id, _) = locked_target(state, scope, headers, selected).await?;
    password_policy(password)?;
    let subject = linked_subject(&mut tx, id).await?.ok_or(IDENTITY_MISSING)?;
    lifecycle
        .provider()?
        .reset_password(&subject, password, temporary)
        .await?;
    tx.commit().await.map_err(database_unavailable)?;
    let message = if by_id {
        "Password updated successfully by user ID."
    } else {
        "Password updated successfully."
    };
    Ok(Json(json!({"message":message})))
}

async fn password_by_name(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    Extension(runtime): Extension<Lifecycle>,
    headers: HeaderMap,
    Json(body): Json<PasswordByName>,
) -> Reply {
    reset_password(
        &state,
        scope,
        &runtime,
        &headers,
        UserSelection::Username(body.username),
        &body.new_password,
        true,
    )
    .await
}

async fn password_by_id(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    Extension(runtime): Extension<Lifecycle>,
    headers: HeaderMap,
    Query(query): Query<PasswordQuery>,
    Json(body): Json<PasswordById>,
) -> Reply {
    reset_password(
        &state,
        scope,
        &runtime,
        &headers,
        UserSelection::Id(body.user_id),
        &body.new_password,
        query.temporary()?,
    )
    .await
}
