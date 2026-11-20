//! Stable HLS sessions backed by the existing Redis camera-demand contract.

mod facade;
mod languages;
mod legacy;
pub(super) mod overlay_lease;
mod playlist;
mod store;

use std::{
    collections::HashMap,
    sync::{Arc, OnceLock},
    time::Duration,
};

use anyhow::{Context, Result, ensure};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path, Query, State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use base64::{
    Engine,
    engine::general_purpose::{URL_SAFE, URL_SAFE_NO_PAD},
};
use redis::aio::ConnectionManager;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use url::Url;

use super::media_auth::{Settings, bounded_env, normalise_label_language};
use crate::{
    ApiError, AppState,
    auth::{Actor, Grant},
    tenancy::{DeploymentScope, database_unavailable},
};

const MAX_BATCH: usize = 24;
const MAX_SITE_STREAMS: usize = 128;
const STREAM_FILTER: &str =
    "sc.recognition_enabled AND (sc.expire_date IS NULL OR sc.expire_date > CURRENT_TIMESTAMP)";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    internal_hls_base_url: String,
    #[serde(default)]
    demand_prefix: String,
    #[serde(default = "public_base")]
    public_base_path: String,
    #[serde(default = "public_hls")]
    public_hls_base_path: String,
    #[serde(default = "public_webrtc")]
    public_webrtc_base_path: String,
    #[serde(default = "startup_wait")]
    startup_wait_seconds: f64,
}

fn public_base() -> String {
    "/hazard/api/stream-playback".into()
}
fn public_hls() -> String {
    "/hazard/media".into()
}
fn public_webrtc() -> String {
    "/hazard/media/webrtc".into()
}
fn startup_wait() -> f64 {
    3.0
}

#[derive(Clone)]
pub struct Playback {
    redis: ConnectionManager,
    settings: Arc<PlaybackSettings>,
}

struct PlaybackSettings {
    scope: Option<DeploymentScope>,
    public_base: String,
    public_hls: String,
    public_webrtc: String,
    upstream: Url,
    http: reqwest::Client,
    startup_wait: Duration,
    demand_prefix: String,
    overlay_demand_ttl: u32,
    clean_demand_ttl: u32,
    change: redis::Script,
    language_response: OnceLock<axum::body::Bytes>,
}

impl Playback {
    pub fn new(
        config: Config,
        redis: ConnectionManager,
        scope: Option<DeploymentScope>,
    ) -> Result<Self> {
        let upstream =
            Url::parse(&config.internal_hls_base_url).context("Invalid internal HLS URL")?;
        ensure!(
            matches!(upstream.scheme(), "http" | "https")
                && upstream.has_host()
                && upstream.username().is_empty()
                && upstream.password().is_none()
                && upstream.query().is_none()
                && upstream.fragment().is_none(),
            "The internal HLS URL requires an HTTP(S) origin without credentials, query or fragment",
        );
        let public_base = validate_public_path(&config.public_base_path)?;
        let public_hls = validate_public_path(&config.public_hls_base_path)?;
        let public_webrtc = validate_public_path(&config.public_webrtc_base_path)?;
        ensure!(
            public_hls.ends_with("/media"),
            "The public HLS base path must end with /media"
        );
        ensure!(
            public_webrtc.ends_with("/media/webrtc"),
            "The public WHEP base path must end with /media/webrtc"
        );
        ensure!(
            (0.0..=30.0).contains(&config.startup_wait_seconds),
            "Playback startup wait must be between zero and 30 seconds"
        );
        ensure!(
            config.demand_prefix.len() <= 128
                && config
                    .demand_prefix
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric()
                        || matches!(byte, b'-' | b'_' | b'.' | b':')),
            "Playback demand prefix contains an invalid character"
        );
        let overlay_demand_ttl = bounded_env("MEDIA_OVERLAY_DEMAND_TTL_SECONDS", 90, 1..=u32::MAX)?;
        let clean_demand_ttl = bounded_env(
            "MEDIA_CLEAN_DEMAND_TTL_SECONDS",
            overlay_demand_ttl,
            1..=u32::MAX,
        )?;
        let http = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::custom(|attempt| {
                // MediaMTX appends a cookie check to the initial playlist URL.
                // Keep the internal request on the same origin and camera.
                if attempt.previous().len() <= 2
                    && attempt
                        .previous()
                        .first()
                        .is_some_and(|initial| playlist::same_playlist(initial, attempt.url()))
                {
                    attempt.follow()
                } else {
                    attempt.stop()
                }
            }))
            .connect_timeout(Duration::from_secs(3))
            .timeout(Duration::from_secs(10))
            .build()?;
        Ok(Self {
            redis,
            settings: Arc::new(PlaybackSettings {
                scope,
                public_base,
                public_hls,
                public_webrtc,
                upstream,
                http,
                startup_wait: Duration::from_secs_f64(config.startup_wait_seconds),
                demand_prefix: config.demand_prefix,
                overlay_demand_ttl,
                clean_demand_ttl,
                change: redis::Script::new(include_str!("session.lua")),
                language_response: OnceLock::new(),
            }),
        })
    }

    pub(super) fn demand_prefix(&self) -> &str {
        &self.settings.demand_prefix
    }

    async fn live_memberships(
        &self,
        db: &sqlx::PgPool,
        actor: &Actor,
        requested: &[i64],
    ) -> Result<Option<HashMap<i64, String>>, ApiError> {
        let Some(scope) = self.settings.scope else {
            return Ok(None);
        };
        let account = scope.account(db, &actor.subject).await?;
        let ids: Vec<_> = requested
            .iter()
            .filter_map(|id| i32::try_from(*id).ok())
            .collect();
        let allowed = sqlx::query_as(
            "SELECT s.id::bigint, s.name FROM sites s
             WHERE s.id = ANY($3) AND ($4 OR EXISTS (
                SELECT 1 FROM user_sites us
                JOIN site_groups sg ON sg.site_id = us.site_id AND sg.group_id = $2
                WHERE us.user_id = $1 AND us.site_id = s.id
             ))",
        )
        .bind(account.id)
        .bind(account.group_id)
        .bind(ids)
        .bind(account.role == "super_admin")
        .fetch_all(db)
        .await
        .map_err(database_unavailable)?;
        Ok(Some(allowed.into_iter().collect()))
    }

    async fn require_memberships(
        &self,
        db: &sqlx::PgPool,
        actor: &Actor,
        sites: &[(i64, &str)],
    ) -> Result<(), Failure> {
        let ids: Vec<_> = sites.iter().map(|(id, _)| *id).collect();
        let allowed = self
            .live_memberships(db, actor, &ids)
            .await
            .map_err(|error| Failure::new(error.status(), "site_membership_unavailable"))?;
        if allowed.is_some_and(|allowed| {
            sites
                .iter()
                .any(|(id, label)| allowed.get(id).map(String::as_str) != Some(*label))
        }) {
            return Err(Failure::new(StatusCode::FORBIDDEN, "site_access_denied"));
        }
        Ok(())
    }

    pub(super) async fn visible_sites(
        &self,
        db: &sqlx::PgPool,
        actor: &Actor,
    ) -> Result<Vec<(i64, String)>, ApiError> {
        let mut sites = actor.sites();
        let ids: Vec<_> = sites.iter().map(|(id, _)| *id).collect();
        if let Some(allowed) = self.live_memberships(db, actor, &ids).await? {
            sites.retain(|(id, name)| allowed.get(id) == Some(name));
        }
        Ok(sites)
    }

    pub(super) fn has_live_scope(&self) -> bool {
        self.settings.scope.is_some()
    }

    pub(super) async fn allows_live_site(
        &self,
        db: &sqlx::PgPool,
        actor: &Actor,
        id: i64,
        label: &str,
    ) -> Result<bool, ApiError> {
        if !actor.allows_site(id, label) {
            return Ok(false);
        }
        let allowed = self.live_memberships(db, actor, &[id]).await?;
        Ok(allowed.is_none_or(|allowed| allowed.get(&id).map(String::as_str) == Some(label)))
    }
}

fn validate_public_path(path: &str) -> Result<String> {
    let path = path.trim_end_matches('/');
    ensure!(
        path.len() <= 256
            && path.strip_prefix('/').is_some_and(|suffix| {
                suffix.split('/').all(|part| {
                    !part.is_empty()
                        && part != "."
                        && part != ".."
                        && part.bytes().all(|byte| {
                            byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.')
                        })
                })
            }),
        "The public playback path must be an absolute URL path"
    );
    Ok(path.into())
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .merge(facade::routes())
        .route("/overlay-languages", get(languages::get))
        .route("/streams/{label}", get(streams))
        .route("/stream-playback", post(single))
        .route("/stream-playback/batch", post(batch))
        .route("/stream-playback/release", post(release))
        .route(
            "/stream-playback/sessions/{session_id}/index.m3u8",
            get(playlist::stable),
        )
        .layer(DefaultBodyLimit::max(64 * 1024))
}

pub(crate) fn facade_routes() -> Router<AppState> {
    facade::routes().layer(DefaultBodyLimit::max(64 * 1024))
}

#[derive(Debug)]
struct Failure(StatusCode, Value);

impl Failure {
    fn new(status: StatusCode, detail: &'static str) -> Self {
        Self(status, detail.into())
    }

    fn invalid(detail: &'static str) -> Self {
        Self::new(StatusCode::UNPROCESSABLE_ENTITY, detail)
    }

    fn unavailable() -> Self {
        Self::new(StatusCode::SERVICE_UNAVAILABLE, "playback_unavailable")
    }
}

impl IntoResponse for Failure {
    fn into_response(self) -> Response {
        let mut response = response(self.0, json!({"detail": self.1}));
        if self.0 == StatusCode::UNAUTHORIZED {
            response
                .headers_mut()
                .insert(header::WWW_AUTHENTICATE, "Bearer".parse().unwrap());
        }
        response
    }
}

impl From<redis::RedisError> for Failure {
    fn from(error: redis::RedisError) -> Self {
        tracing::warn!(kind = ?error.kind(), "Playback storage unavailable");
        Self::unavailable()
    }
}

impl From<sqlx::Error> for Failure {
    fn from(error: sqlx::Error) -> Self {
        tracing::warn!(code = ?error.as_database_error().and_then(|error| error.code()), "Playback catalogue unavailable");
        Self::unavailable()
    }
}

fn response(status: StatusCode, body: Value) -> Response {
    (
        status,
        [(header::CACHE_CONTROL, "private, no-store")],
        Json(body),
    )
        .into_response()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Profile {
    Clean,
    Overlay,
}

impl Profile {
    fn parse(value: &str) -> Result<Self, Failure> {
        match value.trim().to_ascii_lowercase().as_str() {
            "clean" => Ok(Self::Clean),
            "overlay" => Ok(Self::Overlay),
            _ => Err(Failure::invalid("unsupported_profile")),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Rendition {
    #[default]
    Detail,
    Preview,
}

#[derive(Debug, Default, Deserialize)]
struct Request {
    label: Option<String>,
    stream_id: Option<String>,
    key: Option<String>,
    session_id: Option<String>,
    profile: Option<String>,
    rendition: Option<Rendition>,
    language: Option<String>,
}

#[derive(Deserialize)]
struct BatchRequest {
    label: Option<String>,
    streams: Option<Vec<Request>>,
    profile: Option<String>,
    rendition: Option<Rendition>,
    language: Option<String>,
}

#[derive(Deserialize)]
struct StreamQuery {
    overlay: Option<String>,
    language: Option<String>,
}

struct Plan {
    label: String,
    site_id: i64,
    key: String,
    session_id: Option<String>,
    profile: Profile,
    rendition: Rendition,
    language: Option<&'static str>,
}

impl Plan {
    fn from_request(input: Request, actor: &Actor, media: &Settings) -> Result<Self, Failure> {
        let label = input
            .label
            .filter(|label| !label.is_empty())
            .ok_or_else(|| Failure::invalid("label_required"))?;
        let site_id = allowed_site(actor, &label)?;
        if !valid_camera(&label) {
            return Err(Failure::invalid("invalid_label"));
        }
        let key = match (input.key.filter(|key| !key.is_empty()), input.stream_id) {
            (Some(key), _) => key,
            (None, Some(encoded)) => {
                decode_camera(&encoded).ok_or_else(|| Failure::invalid("invalid_stream_id"))?
            }
            (None, None) => return Err(Failure::invalid("stream_id_or_key_required")),
        };
        if !valid_camera(&key) {
            return Err(Failure::invalid("invalid_camera_name"));
        }
        let profile = Profile::parse(input.profile.as_deref().unwrap_or("clean"))?;
        let language = if profile == Profile::Overlay {
            let value = input
                .language
                .as_deref()
                .filter(|value| !value.trim().is_empty())
                .map(normalise_label_language)
                .unwrap_or("en");
            if !media.languages.contains(&value) {
                return Err(Failure::invalid("unsupported_language"));
            }
            Some(value)
        } else {
            None
        };
        if input
            .session_id
            .as_deref()
            .is_some_and(|id| !valid_session_id(id))
        {
            return Err(Failure::invalid("invalid_session_id"));
        }
        Ok(Self {
            label,
            site_id,
            key,
            session_id: input.session_id,
            profile,
            rendition: input.rendition.unwrap_or_default(),
            language,
        })
    }
}

fn allowed_site(actor: &Actor, label: &str) -> Result<i64, Failure> {
    actor
        .sites()
        .into_iter()
        .find_map(|(id, name)| (name == label).then_some(id))
        .ok_or_else(|| Failure::new(StatusCode::FORBIDDEN, "site_access_denied"))
}

fn valid_camera(camera: &str) -> bool {
    !camera.trim().is_empty() && camera.chars().count() <= 80 && !camera.contains('\0')
}

fn decode_camera(encoded: &str) -> Option<String> {
    if encoded.len() > 428 {
        return None;
    }
    let bytes = URL_SAFE
        .decode(encoded)
        .or_else(|_| URL_SAFE_NO_PAD.decode(encoded))
        .ok()?;
    String::from_utf8(bytes).ok()
}

fn valid_session_id(id: &str) -> bool {
    (16..=128).contains(&id.len())
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

fn base_path(label: &str, key: &str) -> String {
    format!(
        "hazard_{}_{}",
        URL_SAFE_NO_PAD.encode(label),
        URL_SAFE_NO_PAD.encode(key)
    )
}

fn validated_json<T>(body: Result<Json<T>, JsonRejection>) -> Result<T, Failure> {
    body.map(|Json(data)| data)
        .map_err(|error| Failure::new(error.status(), "invalid_playback_request"))
}

fn runtime(state: &AppState) -> Result<&Playback, Failure> {
    state.playback.as_ref().ok_or_else(Failure::unavailable)
}

async fn list_cameras(
    db: &sqlx::PgPool,
    site_id: i64,
    label: &str,
    max: usize,
) -> Result<Vec<String>, Failure> {
    let query = format!(
        "SELECT DISTINCT sc.stream_name FROM stream_configs sc
         JOIN sites s ON s.id = sc.site_id
         WHERE sc.site_id = $1 AND s.name = $2 AND {STREAM_FILTER}
         ORDER BY sc.stream_name LIMIT $3"
    );
    let cameras = sqlx::query_scalar(&query)
        .bind(site_id)
        .bind(label)
        .bind(max as i64 + 1)
        .fetch_all(db)
        .await?;
    Ok(cameras)
}

// Verify all requested cameras in one indexed query after checking site grants.
async fn verify_cameras(db: &sqlx::PgPool, plans: &[Plan]) -> Result<(), Failure> {
    let ids: Vec<_> = plans.iter().map(|plan| plan.site_id).collect();
    let labels: Vec<_> = plans.iter().map(|plan| &plan.label).collect();
    let keys: Vec<_> = plans.iter().map(|plan| &plan.key).collect();
    let query = format!(
        "SELECT NOT EXISTS (
            SELECT FROM unnest($1::bigint[], $2::text[], $3::text[]) AS wanted(site_id,label,name)
            WHERE NOT EXISTS (
                SELECT FROM stream_configs sc JOIN sites s ON s.id = sc.site_id
                WHERE sc.site_id = wanted.site_id AND s.name = wanted.label
                    AND sc.stream_name = wanted.name AND {STREAM_FILTER}
            )
        )"
    );
    if !sqlx::query_scalar::<_, bool>(&query)
        .bind(ids)
        .bind(labels)
        .bind(keys)
        .fetch_one(db)
        .await?
    {
        return Err(Failure::new(StatusCode::NOT_FOUND, "stream_not_found"));
    }
    Ok(())
}

async fn single(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Result<Json<Request>, JsonRejection>,
) -> Result<Response, Response> {
    let actor = state
        .auth
        .authorize(&headers, Grant::StreamRead)
        .await
        .map_err(ApiError::into_response)?;
    single_inner(&state, actor, body)
        .await
        .map_err(Failure::into_response)
}

async fn single_inner(
    state: &AppState,
    actor: Actor,
    body: Result<Json<Request>, JsonRejection>,
) -> Result<Response, Failure> {
    let playback = runtime(state)?;
    let media = state.media_auth.playback_settings();
    let plan = Plan::from_request(validated_json(body)?, &actor, media)?;
    playback
        .require_memberships(&state.db, &actor, &[(plan.site_id, &plan.label)])
        .await?;
    verify_cameras(&state.db, std::slice::from_ref(&plan)).await?;
    let mut bodies = playback.negotiate(vec![plan], &actor, media).await?;
    let body = bodies.pop().expect("one playback descriptor");
    Ok(response(body.status(), json!(body)))
}

async fn batch(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Result<Json<BatchRequest>, JsonRejection>,
) -> Result<Response, Response> {
    let actor = state
        .auth
        .authorize(&headers, Grant::StreamRead)
        .await
        .map_err(ApiError::into_response)?;
    batch_inner(&state, actor, body)
        .await
        .map_err(Failure::into_response)
}

async fn batch_inner(
    state: &AppState,
    actor: Actor,
    body: Result<Json<BatchRequest>, JsonRejection>,
) -> Result<Response, Failure> {
    let playback = runtime(state)?;
    let media = state.media_auth.playback_settings();
    let body = validated_json(body)?;
    let profile = body.profile.or(Some("overlay".into()));
    let mut requests = body.streams.unwrap_or_default();
    let site_wall = requests.is_empty();
    if site_wall {
        let label = body
            .label
            .as_deref()
            .filter(|label| !label.is_empty())
            .ok_or_else(|| Failure::invalid("label_required"))?;
        let id = allowed_site(&actor, label)?;
        playback
            .require_memberships(&state.db, &actor, &[(id, label)])
            .await?;
        requests = list_cameras(&state.db, id, label, MAX_BATCH)
            .await?
            .into_iter()
            .map(|key| Request {
                key: Some(key),
                ..Default::default()
            })
            .collect();
    }
    if requests.len() > MAX_BATCH {
        return Err(Failure(
            StatusCode::UNPROCESSABLE_ENTITY,
            json!({"code":"stream_batch_limit_exceeded","count":requests.len(),"max_streams":MAX_BATCH}),
        ));
    }
    let plans: Vec<_> = requests
        .into_iter()
        .map(|item| {
            Plan::from_request(
                Request {
                    label: item
                        .label
                        .filter(|label| !label.is_empty())
                        .or_else(|| body.label.clone()),
                    profile: item.profile.or_else(|| profile.clone()),
                    rendition: item.rendition.or(body.rendition),
                    language: item.language.or_else(|| body.language.clone()),
                    ..item
                },
                &actor,
                media,
            )
        })
        .collect::<Result<_, _>>()?;
    if !site_wall {
        let sites: Vec<_> = plans
            .iter()
            .map(|plan| (plan.site_id, plan.label.as_str()))
            .collect();
        playback
            .require_memberships(&state.db, &actor, &sites)
            .await?;
        verify_cameras(&state.db, &plans).await?;
    }
    let items = playback.negotiate(plans, &actor, media).await?;
    let status = if items
        .iter()
        .any(|body| body.status() == StatusCode::ACCEPTED)
    {
        StatusCode::ACCEPTED
    } else {
        StatusCode::OK
    };
    let base = &playback.settings.public_base;
    Ok(response(
        status,
        json!({
            "count": items.len(),
            "items": items,
            "stream_playback_endpoint": base,
            "batch_endpoint": format!("{base}/batch"),
            "release_endpoint": format!("{base}/release"),
            "max_streams": MAX_BATCH,
        }),
    ))
}

async fn streams(
    State(state): State<AppState>,
    Path(label): Path<String>,
    Query(query): Query<StreamQuery>,
    headers: HeaderMap,
) -> Result<Response, Response> {
    let actor = state
        .auth
        .authorize(&headers, Grant::StreamRead)
        .await
        .map_err(ApiError::into_response)?;
    streams_inner(&state, actor, label, query)
        .await
        .map_err(Failure::into_response)
}

async fn streams_inner(
    state: &AppState,
    actor: Actor,
    label: String,
    query: StreamQuery,
) -> Result<Response, Failure> {
    let id = allowed_site(&actor, &label)?;
    let playback = runtime(state)?;
    playback
        .require_memberships(&state.db, &actor, &[(id, &label)])
        .await?;
    let media = state.media_auth.playback_settings();
    let profile = if overlay_lease::backend_mode(query.overlay.as_deref()) {
        "overlay"
    } else {
        "clean"
    };
    let keys = list_cameras(&state.db, id, &label, MAX_SITE_STREAMS).await?;
    if keys.len() > MAX_SITE_STREAMS {
        return Err(Failure::invalid("site_stream_limit_exceeded"));
    }
    let plans = keys
        .into_iter()
        .map(|key| {
            Plan::from_request(
                Request {
                    label: Some(label.clone()),
                    key: Some(key),
                    profile: Some(profile.into()),
                    language: query.language.clone(),
                    ..Default::default()
                },
                &actor,
                media,
            )
        })
        .collect::<Result<_, _>>()?;
    let items = playback.negotiate(plans, &actor, media).await?;
    Ok(response(StatusCode::OK, json!({"streams":items})))
}

async fn release(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Result<Json<Request>, JsonRejection>,
) -> Result<Response, Response> {
    let actor = state
        .auth
        .authorize(&headers, Grant::StreamRead)
        .await
        .map_err(ApiError::into_response)?;
    release_inner(&state, actor, body)
        .await
        .map_err(Failure::into_response)
}

async fn release_inner(
    state: &AppState,
    actor: Actor,
    body: Result<Json<Request>, JsonRejection>,
) -> Result<Response, Failure> {
    let id = validated_json(body)?
        .session_id
        .ok_or_else(|| Failure::invalid("session_id_required"))?;
    if !valid_session_id(&id) {
        return Err(Failure::invalid("invalid_session_id"));
    }
    let profile = runtime(state)?
        .release(&id, &actor, state.media_auth.playback_settings())
        .await?;
    Ok(response(
        StatusCode::OK,
        json!({"status":"released","session_id":id,"profile":profile}),
    ))
}
