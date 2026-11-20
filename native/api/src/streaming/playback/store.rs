use std::collections::HashSet;

use chrono::{Duration, Utc};
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::*;

const SESSION_PREFIX: &str = "stream_playback_session:";
const MEDIA_SESSION_TTL: u64 = 900;

#[derive(Serialize, Deserialize)]
pub(super) struct Session {
    pub session_id: String,
    #[serde(default)]
    pub revision: String,
    pub username: String,
    pub label: String,
    pub stream_name: String,
    stream_id: String,
    pub profile: Profile,
    rendition: Rendition,
    language: Option<String>,
    pub base_media_path: String,
    overlay_media_path: Option<String>,
    pub created_at: chrono::DateTime<Utc>,
    expires_at: chrono::DateTime<Utc>,
}

impl Session {
    pub(super) fn selected_path(&self) -> Option<&str> {
        let detail = base_path(&self.label, &self.stream_name);
        let base = match self.rendition {
            Rendition::Detail => detail,
            Rendition::Preview => format!("{detail}_preview"),
        };
        if self.base_media_path != base
            || !valid_camera(&self.label)
            || !valid_camera(&self.stream_name)
        {
            return None;
        }
        match self.profile {
            Profile::Clean => Some(&self.base_media_path),
            Profile::Overlay => {
                let language = self.language.as_deref()?;
                if super::super::media_auth::canonical_language(language) != Some(language) {
                    return None;
                }
                let path = self.overlay_media_path.as_deref()?;
                (path == overlay_path(&base, language)).then_some(path)
            }
        }
    }
}

fn overlay_path(base: &str, language: &str) -> String {
    format!("{base}_annotated_{}", URL_SAFE_NO_PAD.encode(language))
}

#[derive(Serialize)]
struct Options<'a> {
    session_ttl: u32,
    overlay_demand_ttl: u32,
    clean_demand_ttl: u32,
    overlay_limit: u32,
    language_keys: Vec<String>,
    demand_prefix: &'a str,
}

impl<'a> Options<'a> {
    fn new(runtime: &'a Playback, media: &Settings) -> Self {
        Self {
            session_ttl: media.session_ttl,
            overlay_demand_ttl: runtime.settings.overlay_demand_ttl,
            clean_demand_ttl: runtime.settings.clean_demand_ttl,
            overlay_limit: media.overlay_limit,
            demand_prefix: runtime.demand_prefix(),
            language_keys: media
                .languages
                .iter()
                .map(|language| URL_SAFE_NO_PAD.encode(language))
                .collect(),
        }
    }
}

#[derive(Serialize)]
struct Change<'a> {
    key: String,
    id: &'a str,
    replace: bool,
    #[serde(flatten)]
    addition: Option<Addition<'a>>,
}

#[derive(Serialize)]
struct Addition<'a> {
    raw: String,
    detail_path: String,
    base_path: &'a str,
    selected_path: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    language_key: Option<String>,
    capability_key: String,
    capability: String,
    capability_expires: u64,
    revocation_key: &'a str,
    oidc_revocation_key: &'a str,
}

#[derive(Serialize)]
struct Capability<'a> {
    parent: &'a str,
    playback_revision: &'a str,
    username: &'a str,
    expires_at: u64,
    access_revocation_key: &'a str,
    oidc_revocation_key: &'a str,
    site: &'a str,
    cameras: [&'a str; 1],
    profile: Profile,
    quality: Rendition,
    #[serde(skip_serializing_if = "Option::is_none")]
    playback_language: Option<&'a str>,
}

struct Pending {
    session: Session,
    replace: bool,
    token: String,
    detail_path: String,
}

impl Pending {
    fn new(
        plan: Plan,
        actor: &Actor,
        options: &Options<'_>,
        now: chrono::DateTime<Utc>,
    ) -> Result<Self, Failure> {
        let detail_path = base_path(&plan.label, &plan.key);
        let base_path = match plan.rendition {
            Rendition::Detail => detail_path.clone(),
            Rendition::Preview => format!("{detail_path}_preview"),
        };
        let replace = plan.session_id.is_some();
        let session_id = match plan.session_id {
            Some(session_id) => session_id,
            None => opaque_id::<24>()?,
        };
        Ok(Self {
            session: Session {
                session_id,
                revision: opaque_id::<24>()?,
                username: actor.subject.clone(),
                label: plan.label,
                stream_id: URL_SAFE.encode(&plan.key),
                stream_name: plan.key,
                profile: plan.profile,
                rendition: plan.rendition,
                language: plan.language.map(str::to_owned),
                overlay_media_path: plan
                    .language
                    .map(|language| overlay_path(&base_path, language)),
                base_media_path: base_path,
                created_at: now,
                expires_at: now + Duration::seconds(i64::from(options.session_ttl)),
            },
            replace,
            token: opaque_id::<32>()?,
            detail_path,
        })
    }

    fn entry<'a>(
        &'a self,
        actor: &'a Actor,
        capability_expires: u64,
    ) -> Result<Change<'a>, Failure> {
        let session = &self.session;
        let key = format!("{SESSION_PREFIX}{}", session.session_id);
        let capability = Capability {
            parent: &key,
            playback_revision: &session.revision,
            username: &session.username,
            expires_at: capability_expires,
            access_revocation_key: &actor.access_revocation_key,
            oidc_revocation_key: &actor.oidc_revocation_key,
            site: &session.label,
            cameras: [&session.stream_name],
            profile: session.profile,
            quality: session.rendition,
            playback_language: session.language.as_deref(),
        };
        let capability = serde_json::to_string(&capability).map_err(|_| Failure::unavailable())?;
        Ok(Change {
            key,
            id: &session.session_id,
            replace: self.replace,
            addition: Some(Addition {
                raw: serde_json::to_string(session).map_err(|_| Failure::unavailable())?,
                detail_path: self.detail_path.clone(),
                base_path: &session.base_media_path,
                selected_path: session
                    .overlay_media_path
                    .as_deref()
                    .unwrap_or(&session.base_media_path),
                language_key: session
                    .language
                    .as_ref()
                    .map(|language| URL_SAFE_NO_PAD.encode(language)),
                capability_key: format!(
                    "media:session:{:x}",
                    Sha256::digest(self.token.as_bytes())
                ),
                capability,
                capability_expires,
                revocation_key: &actor.access_revocation_key,
                oidc_revocation_key: &actor.oidc_revocation_key,
            }),
        })
    }
}

pub(super) fn opaque_id<const BYTES: usize>() -> Result<String, Failure> {
    let mut bytes = [0; BYTES];
    aws_lc_rs::rand::fill(&mut bytes).map_err(|_| Failure::unavailable())?;
    Ok(URL_SAFE_NO_PAD.encode(bytes))
}

#[derive(Serialize)]
pub(super) struct Descriptor {
    pub(super) session_id: String,
    #[serde(skip)]
    pub(super) revision: String,
    stream_id: String,
    key: String,
    label: String,
    transport: &'static str,
    status: &'static str,
    profile: Profile,
    rendition: Rendition,
    playback_ready: bool,
    pub(super) playback_url: String,
    media_hls_url: String,
    webrtc_url: String,
    language: Option<String>,
    overlay_ready: bool,
    media_path: String,
    expires_at: String,
    expires_in: u32,
    demand_ttl_seconds: u32,
}

impl Descriptor {
    fn new(
        pending: Pending,
        settings: &PlaybackSettings,
        ready: bool,
        expires_at: chrono::DateTime<Utc>,
    ) -> Self {
        let Pending { session, token, .. } = pending;
        let media_path = session
            .overlay_media_path
            .unwrap_or(session.base_media_path);
        let status = if session.profile == Profile::Overlay && !ready {
            "starting"
        } else {
            "ready"
        };
        Self {
            playback_url: format!(
                "{}/sessions/{}/index.m3u8?mt={token}",
                settings.public_base, session.session_id
            ),
            media_hls_url: format!("{}/{media_path}/index.m3u8?mt={token}", settings.public_hls),
            webrtc_url: format!("{}/{media_path}/whep?mt={token}", settings.public_webrtc),
            media_path,
            session_id: session.session_id,
            revision: session.revision,
            stream_id: session.stream_id,
            key: session.stream_name,
            label: session.label,
            transport: "hls",
            status,
            profile: session.profile,
            rendition: session.rendition,
            playback_ready: true,
            language: session.language,
            overlay_ready: ready,
            expires_at: expires_at.to_rfc3339(),
            expires_in: (expires_at.timestamp() - session.created_at.timestamp()).max(1) as u32,
            demand_ttl_seconds: settings.overlay_demand_ttl,
        }
    }

    pub(super) fn status(&self) -> StatusCode {
        if self.status == "starting" {
            StatusCode::ACCEPTED
        } else {
            StatusCode::OK
        }
    }
}

impl Playback {
    pub(super) async fn negotiate(
        &self,
        plans: Vec<Plan>,
        actor: &Actor,
        media: &Settings,
    ) -> Result<Vec<Descriptor>, Failure> {
        if plans.is_empty() {
            return Ok(Vec::new());
        }
        let now = Utc::now();
        let options = Options::new(self, media);
        let pending: Vec<_> = plans
            .into_iter()
            .map(|plan| Pending::new(plan, actor, &options, now))
            .collect::<Result<_, _>>()?;
        let mut ids = HashSet::new();
        if pending
            .iter()
            .any(|item| !ids.insert(&item.session.session_id))
        {
            return Err(Failure::invalid("duplicate_playback_session"));
        }
        let max_ttl = MEDIA_SESSION_TTL.min(u64::from(media.session_ttl));
        let capability_expires = actor
            .access_expires_at
            .min(now.timestamp() as u64 + max_ttl);
        let expires_at = chrono::DateTime::from_timestamp(capability_expires as i64, 0)
            .ok_or_else(Failure::unavailable)?;
        let changes: Vec<_> = pending
            .iter()
            .map(|entry| entry.entry(actor, capability_expires))
            .collect::<Result<_, _>>()?;
        let flags = self.change("create", actor, &options, &changes).await?;
        if flags.len() != pending.len() {
            return Err(Failure::unavailable());
        }
        Ok(pending
            .into_iter()
            .zip(flags)
            .map(|(pending, ready)| {
                Descriptor::new(pending, &self.settings, ready == 1, expires_at)
            })
            .collect())
    }

    pub(super) async fn release(
        &self,
        id: &str,
        actor: &Actor,
        media: &Settings,
    ) -> Result<Profile, Failure> {
        let options = Options::new(self, media);
        let flags = self
            .change(
                "release",
                actor,
                &options,
                &[Change {
                    key: format!("{SESSION_PREFIX}{id}"),
                    id,
                    replace: true,
                    addition: None,
                }],
            )
            .await?;
        match flags.as_slice() {
            [0] => Ok(Profile::Clean),
            [1] => Ok(Profile::Overlay),
            _ => Err(Failure::unavailable()),
        }
    }

    async fn change(
        &self,
        action: &str,
        actor: &Actor,
        options: &Options<'_>,
        entries: &[Change<'_>],
    ) -> Result<Vec<u8>, Failure> {
        let options = serde_json::to_string(options).map_err(|_| Failure::unavailable())?;
        let entries = serde_json::to_string(entries).map_err(|_| Failure::unavailable())?;
        let mut redis = self.redis.clone();
        let result: Vec<u8> = self
            .settings
            .change
            .arg(action)
            .arg(&actor.subject)
            .arg(options)
            .arg(entries)
            .arg(actor.username.as_deref().unwrap_or(""))
            .invoke_async(&mut redis)
            .await?;
        match result.split_first() {
            Some((0, flags)) => Ok(flags.into()),
            Some((1, _)) => Err(Failure::new(StatusCode::NOT_FOUND, "session_not_found")),
            Some((2, _)) => Err(Failure::new(StatusCode::FORBIDDEN, "session_forbidden")),
            Some((3, _)) => Err(Failure::new(
                StatusCode::TOO_MANY_REQUESTS,
                "overlay_language_limit_reached",
            )),
            Some((4, _)) => Err(Failure::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "session_id_collision",
            )),
            Some((6, _)) => Err(Failure::new(
                StatusCode::UNAUTHORIZED,
                "invalid_access_session",
            )),
            _ => Err(Failure::unavailable()),
        }
    }

    pub(super) async fn load(&self, id: &str) -> Result<Session, Failure> {
        let mut redis = self.redis.clone();
        let value: Option<Vec<u8>> = redis.get(format!("{SESSION_PREFIX}{id}")).await?;
        let value =
            value.ok_or_else(|| Failure::new(StatusCode::NOT_FOUND, "session_not_found"))?;
        let session: Session =
            serde_json::from_slice(&value).map_err(|_| Failure::unavailable())?;
        if session.session_id != id || session.selected_path().is_none() {
            return Err(Failure::unavailable());
        }
        Ok(session)
    }
}
