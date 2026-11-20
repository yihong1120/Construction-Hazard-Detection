//! The shared Flutter playback contract for BFF and bearer clients.
use std::collections::HashSet;

use axum::{
    extract::{Path, State, rejection::JsonRejection},
    response::IntoResponse,
    routing::{delete, post},
};
use sha2::{Digest, Sha256};

use super::{
    Actor, ApiError, AppState, Failure, Grant, Json, Plan, Playback, Profile, Rendition, Request,
    Response, StatusCode, list_cameras, response, runtime, valid_session_id, validated_json,
    verify_cameras,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

const MAX_WALL_CAMERAS: usize = 32;
const API: &str = "/hazard/api/db_management/api/playback";

fn validate_wall_cameras(cameras: &[String]) -> Result<(), Failure> {
    let mut unique = HashSet::new();
    if cameras.len() > MAX_WALL_CAMERAS
        || cameras
            .iter()
            .any(|name| name.trim().is_empty() || !unique.insert(name))
    {
        return Err(Failure::invalid("invalid_wall_camera_selection"));
    }
    Ok(())
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Transport {
    #[default]
    Hls,
}

#[derive(Deserialize)]
struct SessionRequest {
    site: String,
    camera: String,
    profile: Option<Profile>,
    language: Option<String>,
    #[serde(rename = "transport", default)]
    _transport: Transport,
    session_id: Option<String>,
}

#[derive(Deserialize)]
struct WallRequest {
    site: String,
    cameras: Option<Vec<String>>,
    profile: Option<Profile>,
    language: Option<String>,
    #[serde(rename = "transport", default)]
    _transport: Transport,
}

#[derive(Deserialize)]
struct RenewRequest {
    id: String,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Single,
    MultiStream,
}

#[derive(Serialize, Deserialize)]
struct Member {
    id: String,
    revision: String,
    camera: String,
    capability_key: String,
}

#[derive(Serialize, Deserialize)]
struct Lease {
    owner: String,
    site: String,
    site_id: i64,
    mode: Mode,
    members: Vec<Member>,
}

impl Lease {
    const PREFIX: &str = "playback_facade:rust:";

    fn key(id: &str) -> String {
        format!("{}{id}", Self::PREFIX)
    }

    fn matches_owner(&self, actor: &Actor) -> Result<(), Failure> {
        if self.owner != actor.subject {
            return Err(expired());
        }
        Ok(())
    }
}

pub(super) fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/playback/sessions", post(create_session))
        .route("/api/playback/walls", post(create_wall))
        .route("/api/playback/sessions/renew", post(renew))
        .route("/api/playback/sessions/{id}", delete(release))
}

fn expired() -> Failure {
    Failure::new(StatusCode::UNAUTHORIZED, "expired_media_session")
}

fn resolve_site(actor: &Actor, site: &str) -> Result<(i64, String), Failure> {
    let sites = actor.sites();
    if let Some((id, label)) = sites.iter().find(|(_, label)| label == site) {
        return Ok((*id, label.clone()));
    }
    if let Ok(site_id) = site.parse::<i64>()
        && let Some(site) = sites.into_iter().find(|(id, _)| *id == site_id)
    {
        return Ok(site);
    }
    Err(Failure::new(StatusCode::FORBIDDEN, "site_access_denied"))
}

async fn actor(state: &AppState, headers: &axum::http::HeaderMap) -> Result<Actor, Response> {
    state
        .auth
        .authorize(headers, Grant::StreamRead)
        .await
        .map_err(ApiError::into_response)
}

async fn create_session(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    body: Result<Json<SessionRequest>, JsonRejection>,
) -> Result<Response, Response> {
    let actor = actor(&state, &headers).await?;
    let body = validated_json(body).map_err(Failure::into_response)?;
    create_inner(
        &state,
        actor,
        LeaseRequest {
            site: body.site,
            cameras: vec![body.camera],
            profile: body.profile.unwrap_or(Profile::Overlay),
            language: body.language,
            replacing: body.session_id,
            mode: Mode::Single,
        },
    )
    .await
    .map_err(Failure::into_response)
}

async fn create_wall(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    body: Result<Json<WallRequest>, JsonRejection>,
) -> Result<Response, Response> {
    let actor = actor(&state, &headers).await?;
    let body = validated_json(body).map_err(Failure::into_response)?;
    let profile = body.profile.unwrap_or(Profile::Overlay);
    let cameras = body.cameras.unwrap_or_default();
    validate_wall_cameras(&cameras).map_err(Failure::into_response)?;
    create_inner(
        &state,
        actor,
        LeaseRequest {
            site: body.site,
            cameras,
            profile,
            language: body.language,
            replacing: None,
            mode: Mode::MultiStream,
        },
    )
    .await
    .map_err(Failure::into_response)
}

struct LeaseRequest {
    site: String,
    cameras: Vec<String>,
    profile: Profile,
    language: Option<String>,
    replacing: Option<String>,
    mode: Mode,
}

async fn create_inner(
    state: &AppState,
    actor: Actor,
    mut input: LeaseRequest,
) -> Result<Response, Failure> {
    let playback = runtime(state)?;
    let media = state.media_auth.playback_settings();
    let (site_id, site) = resolve_site(&actor, &input.site)?;
    playback
        .require_memberships(&state.db, &actor, &[(site_id, &site)])
        .await?;
    let default_wall = input.cameras.is_empty();
    if default_wall {
        input.cameras = list_cameras(&state.db, site_id, &site, MAX_WALL_CAMERAS).await?;
        if input.cameras.len() > MAX_WALL_CAMERAS {
            return Err(Failure::invalid("stream_batch_limit_exceeded"));
        }
    }
    if input.cameras.is_empty() {
        return Err(Failure::new(StatusCode::NOT_FOUND, "cameras_not_found"));
    }
    let is_wall = matches!(input.mode, Mode::MultiStream);
    let rendition = if is_wall {
        Rendition::Preview
    } else {
        Rendition::Detail
    };
    let profile = match input.profile {
        Profile::Clean => "clean",
        Profile::Overlay => "overlay",
    };
    let plans = input
        .cameras
        .into_iter()
        .map(|camera| {
            Plan::from_request(
                Request {
                    label: Some(site.clone()),
                    key: Some(camera),
                    session_id: input.replacing.clone(),
                    profile: Some(profile.to_owned()),
                    rendition: Some(rendition),
                    language: input.language.clone(),
                    ..Default::default()
                },
                &actor,
                media,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    if !default_wall {
        verify_cameras(&state.db, &plans).await?;
    }
    let id = super::store::opaque_id::<24>()?;
    let mut status = StatusCode::OK;
    let descriptors = playback.negotiate(plans, &actor, media).await?;
    let mut members = Vec::with_capacity(descriptors.len());
    let mut items = Vec::with_capacity(descriptors.len());
    let mut expires = u64::MAX;
    for descriptor in descriptors {
        if descriptor.status() == StatusCode::ACCEPTED {
            status = StatusCode::ACCEPTED;
        }
        let (_, token) = descriptor
            .playback_url
            .rsplit_once("?mt=")
            .ok_or_else(Failure::unavailable)?;
        let capability_key = format!("media:session:{:x}", Sha256::digest(token.as_bytes()));
        let id = descriptor.session_id.clone();
        let revision = descriptor.revision.clone();
        let mut item = serde_json::to_value(descriptor).map_err(|_| Failure::unavailable())?;
        let member = Member {
            id,
            revision,
            capability_key,
            camera: item["key"]
                .as_str()
                .ok_or_else(Failure::unavailable)?
                .to_owned(),
        };
        expires = expires.min(
            item["expires_in"]
                .as_u64()
                .ok_or_else(Failure::unavailable)?,
        );
        if is_wall {
            item["preview_hls_url"] = item["playback_url"].clone();
            item["detail_camera"] = json!(member.camera);
            item["state"] = serde_json::Value::Null;
        }
        item["hls_url"] = item["playback_url"].clone();
        item["title"] = json!(member.camera);
        item["camera"] = json!(member.camera);
        items.push(item);
        members.push(member);
    }
    let lease = Lease {
        owner: actor.subject.clone(),
        site,
        site_id,
        mode: input.mode,
        members,
    };
    // Register the aggregate only if every camera and capability is from this negotiation.
    let record = serde_json::to_vec(&lease).map_err(|_| Failure::unavailable())?;
    let ttl = match mutate(
        playback,
        "register",
        &id,
        &record,
        &actor,
        media.session_ttl,
    )
    .await
    {
        Ok(ttl) => ttl,
        Err(error) => {
            // A failed reply can leave newly negotiated sessions in Redis.
            // Remove only the exact revisions prepared for this group.
            let _ = mutate(playback, "abort", &id, &record, &actor, media.session_ttl).await;
            return Err(error);
        }
    };
    let mut body = if is_wall {
        json!({"items": items, "count": items.len(), "layout": "responsive", "max_streams": MAX_WALL_CAMERAS})
    } else {
        items.pop().ok_or_else(Failure::unavailable)?
    };
    body["id"] = json!(id);
    body["site"] = json!(lease.site);
    body["mode"] = json!(lease.mode);
    body["profile"] = json!(profile);
    body["quality"] = json!(rendition);
    body["token_transport"] = json!("query");
    body["expires_in"] = json!(expires.min(ttl));
    body["single_endpoint"] = json!(format!("{API}/sessions"));
    body["wall_endpoint"] = json!(format!("{API}/walls"));
    body["renew_endpoint"] = json!(format!("{API}/sessions/renew"));
    Ok(response(status, body))
}

async fn load(playback: &Playback, id: &str, actor: &Actor) -> Result<(Vec<u8>, Lease), Failure> {
    if !valid_session_id(id) {
        return Err(expired());
    }
    let mut connection = playback.redis.clone();
    let record = redis::cmd("GET")
        .arg(Lease::key(id))
        .query_async::<Option<Vec<u8>>>(&mut connection)
        .await?
        .ok_or_else(expired)?;
    let lease: Lease = serde_json::from_slice(&record).map_err(|_| Failure::unavailable())?;
    if lease.members.is_empty() || lease.members.len() > MAX_WALL_CAMERAS {
        return Err(Failure::unavailable());
    }
    lease.matches_owner(actor)?;
    Ok((record, lease))
}

async fn mutate(
    playback: &Playback,
    action: &str,
    id: &str,
    record: &[u8],
    actor: &Actor,
    session_ttl: u32,
) -> Result<u64, Failure> {
    let result = redis::Script::new(include_str!("facade.lua"))
        .key(Lease::key(id))
        .arg(action)
        .arg(record)
        .arg(&actor.subject)
        .arg(actor.access_expires_at)
        .arg(&actor.access_revocation_key)
        .arg(&actor.oidc_revocation_key)
        .arg(session_ttl)
        .arg(playback.demand_prefix())
        .invoke_async::<i64>(&mut playback.redis.clone())
        .await?;
    if result < 0 {
        return Err(if result == -1 {
            expired()
        } else {
            Failure::unavailable()
        });
    }
    Ok(result as u64)
}

async fn renew(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    body: Result<Json<RenewRequest>, JsonRejection>,
) -> Result<Response, Response> {
    let actor = actor(&state, &headers).await?;
    renew_inner(
        &state,
        actor,
        &headers,
        validated_json(body).map_err(Failure::into_response)?.id,
    )
    .await
    .map_err(Failure::into_response)
}

async fn renew_inner(
    state: &AppState,
    actor: Actor,
    headers: &axum::http::HeaderMap,
    id: String,
) -> Result<Response, Failure> {
    let playback = runtime(state)?;
    let (record, lease) = match load(playback, &id, &actor).await {
        Ok(loaded) => loaded,
        Err(Failure(StatusCode::UNAUTHORIZED, _)) => {
            let (mode, ttl) = super::legacy::renew(state, playback, &actor, headers, &id)
                .await?
                .ok_or_else(expired)?;
            return Ok(renewed(&id, json!(mode), ttl));
        }
        Err(error) => return Err(error),
    };
    if !actor.allows_site(lease.site_id, &lease.site) {
        return Err(expired());
    }
    playback
        .require_memberships(&state.db, &actor, &[(lease.site_id, &lease.site)])
        .await?;
    let plans: Vec<_> = lease
        .members
        .iter()
        .map(|member| Plan {
            label: lease.site.clone(),
            site_id: lease.site_id,
            key: member.camera.clone(),
            session_id: None,
            profile: Profile::Clean,
            rendition: Rendition::Detail,
            language: None,
        })
        .collect();
    verify_cameras(&state.db, &plans).await?;
    let ttl = mutate(
        playback,
        "renew",
        &id,
        &record,
        &actor,
        state.media_auth.playback_settings().session_ttl,
    )
    .await?;
    Ok(renewed(&id, json!(lease.mode), ttl))
}

fn renewed(id: &str, mode: serde_json::Value, ttl: u64) -> Response {
    response(
        StatusCode::OK,
        json!({
            "id": id, "mode": mode, "renew_endpoint": format!("{API}/sessions/renew"),
            "expires_in": ttl, "renewed": true, "hls_urls_changed": false,
        }),
    )
}

async fn release(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: axum::http::HeaderMap,
) -> Result<StatusCode, Response> {
    let actor = actor(&state, &headers).await?;
    let playback = runtime(&state).map_err(Failure::into_response)?;
    match load(playback, &id, &actor).await {
        Ok((record, _)) => {
            mutate(
                playback,
                "release",
                &id,
                &record,
                &actor,
                state.media_auth.playback_settings().session_ttl,
            )
            .await
            .map_err(Failure::into_response)?;
        }
        Err(Failure(StatusCode::UNAUTHORIZED, _)) => {
            if !super::legacy::release(&state, playback, &actor, &headers, &id)
                .await
                .map_err(Failure::into_response)?
                && id.len() == 16
            {
                return Err(
                    Failure::new(StatusCode::NOT_FOUND, "session_not_found").into_response()
                );
            }
        }
        Err(error) => return Err(error.into_response()),
    }
    Ok(StatusCode::NO_CONTENT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wall_camera_selection_accepts_thirty_two_unique_names() {
        let mut cameras: Vec<_> = (0..32).map(|index| format!("camera-{index}")).collect();
        assert!(validate_wall_cameras(&[]).is_ok());
        assert!(validate_wall_cameras(&cameras).is_ok());
        cameras.push("camera-32".into());
        assert!(validate_wall_cameras(&cameras).is_err());
        cameras.pop();
        cameras[31] = cameras[0].clone();
        assert!(validate_wall_cameras(&cameras).is_err());
        cameras[31] = " ".into();
        assert!(validate_wall_cameras(&cameras).is_err());
    }
}
