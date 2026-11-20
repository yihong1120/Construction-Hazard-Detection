//! Renew and release media capabilities issued by the Python playback API.
use axum::http::HeaderMap;

use super::*;

const LEGACY_MEDIA_TTL: u64 = 900;

#[derive(Deserialize)]
struct Capability {
    id: String,
    user_id: i32,
    parent: String,
    site: String,
    cameras: Vec<String>,
    scope: String,
    profile: Profile,
    quality: Rendition,
    language: Option<String>,
}

struct Session {
    public_key: String,
    token_key: String,
    bytes: Vec<u8>,
    capability: Capability,
}

fn web_parent(headers: &HeaderMap) -> Option<String> {
    use sha2::{Digest, Sha256};
    let mut parent = None;
    for cookie in headers.get_all(axum::http::header::COOKIE) {
        for pair in cookie.to_str().ok()?.split(';') {
            let Some((name, value)) = pair.trim().split_once('=') else {
                continue;
            };
            if name != "__Host-vn_session" {
                continue;
            }
            if parent.is_some() || value.len() != 43 || !valid_session_id(value) {
                return None;
            }
            parent = Some(format!(
                "bff:session:{:x}",
                Sha256::digest(value.as_bytes())
            ));
        }
    }
    parent
}

async fn load(
    playback: &Playback,
    actor: &Actor,
    id: &str,
    headers: &HeaderMap,
) -> Result<Option<Session>, Failure> {
    let Some(user_id) = actor.user_id else {
        return Ok(None);
    };
    if !valid_session_id(id) {
        return Ok(None);
    }
    let mut redis = playback.redis.clone();
    let public_key = format!("media:public:{id}");
    let token_key: Option<String> = redis::cmd("GET")
        .arg(&public_key)
        .query_async(&mut redis)
        .await?;
    let Some(token_key) = token_key.filter(|key| {
        key.strip_prefix("media:session:").is_some_and(|digest| {
            digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
        })
    }) else {
        return Ok(None);
    };
    let bytes: Option<Vec<u8>> = redis::cmd("GET")
        .arg(&token_key)
        .query_async(&mut redis)
        .await?;
    let Some(bytes) = bytes else {
        return Ok(None);
    };
    let capability: Capability =
        serde_json::from_slice(&bytes).map_err(|_| Failure::unavailable())?;
    if capability.id != id
        || capability.user_id != user_id
        || !(capability.parent == format!("native:user:{user_id}")
            || web_parent(headers).as_deref() == Some(&capability.parent))
        || !valid_camera(&capability.site)
        || capability.cameras.is_empty()
        || capability.cameras.len() > 32
        || capability
            .cameras
            .iter()
            .any(|camera| !valid_camera(camera))
        || !matches!(capability.scope.as_str(), "camera" | "batch")
    {
        return Ok(None);
    }
    Ok(Some(Session {
        public_key,
        token_key,
        bytes,
        capability,
    }))
}

#[derive(Serialize)]
struct Demand {
    key: String,
    siblings: Vec<[String; 2]>,
    language_position: usize,
}

impl Demand {
    fn for_camera(
        playback: &Playback,
        config: &Settings,
        cap: &Capability,
        camera: &str,
    ) -> Result<Self, Failure> {
        let detail = base_path(&cap.site, camera);
        let base = match cap.quality {
            Rendition::Detail => detail.clone(),
            Rendition::Preview => format!("{detail}_preview"),
        };
        let prefix = playback.demand_prefix();
        if cap.profile == Profile::Clean {
            return Ok(Self {
                key: format!("{prefix}media_clean_demand:{base}"),
                siblings: Vec::new(),
                language_position: 0,
            });
        }
        let language =
            normalise_label_language(cap.language.as_deref().unwrap_or(config.default_language));
        let language_position = config
            .languages
            .iter()
            .position(|code| *code == language)
            .ok_or_else(|| Failure::invalid("unsupported_language"))?
            + 1;
        let encoded = URL_SAFE_NO_PAD.encode(language);
        Ok(Self {
            key: format!("{prefix}media_overlay_demand:{base}:{encoded}"),
            siblings: config
                .languages
                .iter()
                .map(|&code| {
                    let encoded = URL_SAFE_NO_PAD.encode(code);
                    [
                        format!("{prefix}media_overlay_demand:{detail}:{encoded}"),
                        format!("{prefix}media_overlay_demand:{detail}_preview:{encoded}"),
                    ]
                })
                .collect(),
            language_position,
        })
    }
}

impl Session {
    async fn mutate(
        self,
        playback: &Playback,
        actor: &Actor,
        config: &Settings,
        renew: bool,
        demands: &[Demand],
    ) -> Result<Option<u64>, Failure> {
        use sha2::{Digest, Sha256};
        let parent_digest = Sha256::digest(self.capability.parent.as_bytes());
        let demands = serde_json::to_string(demands).map_err(|_| Failure::unavailable())?;
        let result: i64 = redis::Script::new(include_str!("legacy.lua"))
            .key(&self.public_key)
            .key(&self.token_key)
            .key(&self.capability.parent)
            .key(format!("media:parent:generation:{parent_digest:x}"))
            .key(format!("media:parent:{parent_digest:x}"))
            .arg(&self.bytes)
            .arg(u8::from(renew))
            .arg(actor.access_expires_at)
            .arg(LEGACY_MEDIA_TTL.min(u64::from(config.session_ttl)))
            .arg(&actor.access_revocation_key)
            .arg(&actor.oidc_revocation_key)
            .arg(u8::from(self.capability.parent.starts_with("bff:session:")))
            .arg(demands)
            .arg(config.overlay_limit)
            .invoke_async(&mut playback.redis.clone())
            .await?;
        match result {
            -1 => Ok(None),
            -2 => Err(Failure::new(
                StatusCode::TOO_MANY_REQUESTS,
                "overlay_language_limit_reached",
            )),
            0.. => Ok(Some(result as u64)),
            _ => Err(Failure::unavailable()),
        }
    }
}

pub(super) async fn renew(
    state: &AppState,
    playback: &Playback,
    actor: &Actor,
    headers: &HeaderMap,
    id: &str,
) -> Result<Option<(&'static str, u64)>, Failure> {
    let Some(session) = load(playback, actor, id, headers).await? else {
        return Ok(None);
    };
    let cap = &session.capability;
    let site_id = allowed_site(actor, &cap.site)?;
    playback
        .require_memberships(&state.db, actor, &[(site_id, &cap.site)])
        .await?;
    let plans: Vec<_> = cap
        .cameras
        .iter()
        .map(|camera| Plan {
            label: cap.site.clone(),
            site_id,
            key: camera.clone(),
            session_id: None,
            profile: cap.profile,
            rendition: cap.quality,
            language: None,
        })
        .collect();
    verify_cameras(&state.db, &plans).await?;
    let mode = if cap.scope == "batch" {
        "multi_stream"
    } else {
        "single"
    };
    let config = state.media_auth.playback_settings();
    let demands: Vec<_> = cap
        .cameras
        .iter()
        .map(|camera| Demand::for_camera(playback, config, cap, camera))
        .collect::<Result<_, _>>()?;
    Ok(session
        .mutate(playback, actor, config, true, &demands)
        .await?
        .map(|ttl| (mode, ttl)))
}

pub(super) async fn release(
    state: &AppState,
    playback: &Playback,
    actor: &Actor,
    headers: &HeaderMap,
    id: &str,
) -> Result<bool, Failure> {
    let Some(session) = load(playback, actor, id, headers).await? else {
        return Ok(false);
    };
    Ok(session
        .mutate(
            playback,
            actor,
            state.media_auth.playback_settings(),
            false,
            &[],
        )
        .await?
        .is_some())
}
