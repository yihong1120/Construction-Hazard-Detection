use std::env;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, anyhow, ensure};
use axum::extract::{Request, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use base64::Engine;
use base64::engine::general_purpose::{URL_SAFE, URL_SAFE_NO_PAD};
use percent_encoding::percent_decode_str;
use redis::aio::{ConnectionManager, ConnectionManagerConfig};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use url::form_urlencoded;

use crate::AppState;

const LANGUAGES: [&str; 8] = ["en", "zh-TW", "zh-CN", "ja", "vi", "id", "fr", "th"];
const PREVIEW_SUFFIX: &str = "_preview";

/// One invocation checks the capability again before renewing any Redis lease.
/// Read and validate the bounded index batch before making writes.
const AUTHORISE: &str = r#"
local capability_key, generation_key, parent_key = KEYS[1], KEYS[2], KEYS[3]
local producer_key, media_index, demand_index, refresh_key = KEYS[4], KEYS[5], KEYS[6], KEYS[7]
local access_revocation_key, oidc_revocation_key = KEYS[8], KEYS[9]
if redis.call('GET', capability_key) ~= ARGV[1] then return 2 end
local time = redis.call('TIME')
local now = tonumber(time[1]) + tonumber(time[2]) / 1000000
local generation = tonumber(redis.call('GET', generation_key) or '0')
local expires_at = tonumber(ARGV[4])
if generation ~= tonumber(ARGV[2]) or now >= expires_at
    or (ARGV[3] == '1' and redis.call('EXISTS', parent_key) == 0) then return 1 end
if (access_revocation_key ~= '' and redis.call('EXISTS', access_revocation_key) > 0)
    or (oidc_revocation_key ~= '' and redis.call('EXISTS', oidc_revocation_key) > 0) then return 1 end
if ARGV[13] ~= '' then
    local parent_raw = redis.call('GET', parent_key)
    if not parent_raw then return 1 end
    local valid, parent = pcall(cjson.decode, parent_raw)
    if not valid or type(parent) ~= 'table' or parent.revision ~= ARGV[13] then return 1 end
end
local decision = tonumber(ARGV[5])
if decision ~= 0 then return decision end

local requested = tonumber(ARGV[12])
if requested > 0 then
    local count, already_active = 0, false
    for index = 10, #KEYS, 2 do
        if redis.call('EXISTS', KEYS[index], KEYS[index + 1]) > 0 then
            count = count + 1
            if (index - 10) / 2 + 1 == requested then already_active = true end
        end
    end
    if not already_active and count >= tonumber(ARGV[11]) then return 5 end
end

local batch = tonumber(ARGV[9])
local media_path = ARGV[10]
local expired, expired_demand, members, active, stale = {}, {}, {}, {}, {}
local refreshing = redis.call('GET', refresh_key)
if not refreshing or refreshing == '' then
    expired = redis.call('ZRANGEBYSCORE', media_index, '-inf', now, 'LIMIT', 0, batch)
    expired_demand = redis.call('ZRANGEBYSCORE', demand_index, '-inf', now, 'LIMIT', 0, batch)
    members = redis.call('ZRANGEBYSCORE', media_index, string.format('(%.6f', now), '+inf', 'LIMIT', 0, batch)
    for _, member in ipairs(members) do
        local session_raw = redis.call('GET', 'stream_playback_session:' .. member)
        local selected_path = nil
        if session_raw then
            local valid, session = pcall(cjson.decode, session_raw)
            if not valid or type(session) ~= 'table' then return 6 end
            selected_path = session.base_media_path
            if session.profile == 'overlay' then selected_path = session.overlay_media_path end
        end
        if selected_path == media_path then table.insert(active, member)
        else table.insert(stale, member) end
    end
end

if producer_key ~= '' then
    local lease = math.max(tonumber(ARGV[6]), expires_at - tonumber(time[1]))
    local producer_value = requested > 0 and '1' or time[1]
    redis.call('SET', producer_key, producer_value, 'EX', lease)
end
if #members + #expired + #expired_demand > 0 then
    for _, member in ipairs(expired) do redis.call('ZREM', media_index, member) end
    for _, member in ipairs(expired_demand) do redis.call('ZREM', demand_index, member) end
    for _, member in ipairs(stale) do
        redis.call('ZREM', media_index, member)
        redis.call('ZREM', demand_index, member)
    end
    local ttl = tonumber(ARGV[7])
    for _, member in ipairs(active) do
        redis.call('EXPIRE', 'stream_playback_session:' .. member, ttl)
        redis.call('ZADD', media_index, now + ttl, member)
        redis.call('ZADD', demand_index, now + ttl, member)
    end
    redis.call('EXPIRE', media_index, ttl)
    redis.call('EXPIRE', demand_index, ttl)
    redis.call('SET', refresh_key, '1', 'EX', ARGV[8])
end
return 0
"#;

#[derive(Clone)]
pub struct MediaAuth {
    redis: ConnectionManager,
    settings: Arc<Settings>,
    authorise: Arc<redis::Script>,
}

impl MediaAuth {
    pub async fn new(client: redis::Client) -> Result<Self> {
        let settings = Settings::from_env()?;
        let redis = client
            .get_connection_manager_with_config(
                ConnectionManagerConfig::new()
                    .set_connection_timeout(Duration::from_secs(3))
                    .set_response_timeout(Duration::from_secs(3)),
            )
            .await
            .map_err(|_| anyhow!("Could not connect to the media authorization Redis service"))?;
        Ok(Self::with_settings(redis, settings))
    }

    fn with_settings(redis: ConnectionManager, settings: Settings) -> Self {
        Self {
            redis,
            settings: Arc::new(settings),
            authorise: Arc::new(redis::Script::new(AUTHORISE)),
        }
    }

    pub(super) fn playback_settings(&self) -> &Settings {
        &self.settings
    }

    pub(super) async fn authorise_playback(
        &self,
        token: String,
        public_uri: &str,
        session_id: &str,
        revision: &str,
        demand_prefix: &str,
    ) -> Result<(), Failure> {
        let path = MediaPath::parse(public_uri).ok_or(Failure::InvalidPath)?;
        let parent = format!("stream_playback_session:{session_id}");
        self.authorise_for(
            MediaRequest { path, token },
            Some((&parent, revision)),
            demand_prefix,
        )
        .await
    }

    #[cfg(test)]
    async fn authorise(&self, access: MediaRequest) -> Result<(), Failure> {
        self.authorise_for(access, None, "").await
    }

    async fn authorise_for(
        &self,
        access: MediaRequest,
        playback: Option<(&str, &str)>,
        demand_prefix: &str,
    ) -> Result<(), Failure> {
        let token_key = format!(
            "media:session:{:x}",
            Sha256::digest(access.token.as_bytes())
        );
        let mut redis = self.redis.clone();
        for _ in 0..3 {
            let raw: Option<Vec<u8>> = redis::cmd("GET")
                .arg(&token_key)
                .query_async(&mut redis)
                .await?;
            let raw = raw.ok_or(Failure::Expired)?;
            let capability: Capability =
                serde_json::from_slice(&raw).map_err(|_| Failure::Expired)?;
            let prefix = capability
                .playback_revision
                .as_ref()
                .map_or("", |_| demand_prefix);
            let leases = Leases::new(&access.path, &self.settings, prefix, &capability);
            let generation_key = format!(
                "media:parent:generation:{:x}",
                Sha256::digest(capability.parent.as_bytes()),
            );
            let needs_parent = capability.parent.starts_with("bff:session:");
            let decision = if capability.user_active == Some(false) {
                3
            } else if !capability.allows(&access.path)
                || playback
                    .is_some_and(|(parent, revision)| !capability.allows_playlist(parent, revision))
            {
                4
            } else {
                0
            };

            let mut invocation = self.authorise.prepare_invoke();
            invocation
                .key(&token_key)
                .key(generation_key)
                .key(&capability.parent)
                .key(&leases.producer)
                .key(&leases.media_index)
                .key(&leases.demand_index)
                .key(&leases.refresh)
                .key(capability.access_revocation_key.as_deref().unwrap_or(""))
                .key(capability.oidc_revocation_key.as_deref().unwrap_or(""));
            for key in &leases.language_keys {
                invocation.key(key);
            }
            let status: u8 = invocation
                .arg(&raw)
                .arg(capability.parent_generation)
                .arg(u8::from(needs_parent))
                .arg(capability.expires_at)
                .arg(decision)
                .arg(self.settings.idle_grace)
                .arg(self.settings.session_ttl)
                .arg(self.settings.refresh_interval)
                .arg(self.settings.refresh_batch)
                .arg(&access.path.path)
                .arg(self.settings.overlay_limit)
                .arg(leases.overlay_position)
                .arg(capability.playback_revision.as_deref().unwrap_or(""))
                .invoke_async(&mut redis)
                .await?;
            match status {
                0 => return Ok(()),
                1 => return Err(Failure::Expired),
                2 => continue,
                3 => return Err(Failure::Inactive),
                4 => return Err(Failure::ScopeDenied),
                5 => return Err(Failure::LanguageLimit),
                _ => return Err(Failure::Unavailable),
            }
        }
        Err(Failure::UpdateBusy)
    }
}

pub fn routes() -> Router<AppState> {
    Router::new().route("/media-auth", get(authorise_media_request))
}

async fn authorise_media_request(
    State(state): State<AppState>,
    request: Request,
) -> Result<Response, Failure> {
    state
        .media_auth
        .authorise_for(
            MediaRequest::parse(&request)?,
            None,
            state
                .playback
                .as_ref()
                .map_or("", super::playback::Playback::demand_prefix),
        )
        .await?;
    Ok((
        StatusCode::NO_CONTENT,
        [
            (header::CACHE_CONTROL, "no-store"),
            (
                header::HeaderName::from_static("x-media-auth-mode"),
                "opaque_media_session",
            ),
        ],
    )
        .into_response())
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Failure {
    InvalidPath,
    Missing,
    Expired,
    Inactive,
    ScopeDenied,
    LanguageLimit,
    UpdateBusy,
    Unavailable,
}

impl From<redis::RedisError> for Failure {
    fn from(_: redis::RedisError) -> Self {
        // Do not disclose key names, tokens, or connection details.
        Self::Unavailable
    }
}

impl IntoResponse for Failure {
    fn into_response(self) -> Response {
        let (status, detail) = match self {
            Self::InvalidPath => (StatusCode::FORBIDDEN, "Invalid media path"),
            Self::Missing => (StatusCode::UNAUTHORIZED, "missing_media_token"),
            Self::Expired => (StatusCode::UNAUTHORIZED, "expired_media_session"),
            Self::Inactive => (StatusCode::UNAUTHORIZED, "inactive_user"),
            Self::ScopeDenied => (StatusCode::FORBIDDEN, "media_scope_denied"),
            Self::LanguageLimit => (
                StatusCode::TOO_MANY_REQUESTS,
                "overlay_language_limit_reached",
            ),
            Self::UpdateBusy => (StatusCode::SERVICE_UNAVAILABLE, "session_update_busy"),
            Self::Unavailable => (StatusCode::SERVICE_UNAVAILABLE, "media_auth_unavailable"),
        };
        let mut response = (status, Json(json!({"detail": detail}))).into_response();
        response
            .headers_mut()
            .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
        if status == StatusCode::UNAUTHORIZED {
            response
                .headers_mut()
                .insert("x-media-auth-error", HeaderValue::from_static(detail));
            response.headers_mut().insert(
                header::WWW_AUTHENTICATE,
                HeaderValue::from_str(&format!("Bearer error=\"{detail}\""))
                    .expect("static detail is valid in a header"),
            );
        }
        response
    }
}

pub(super) struct Settings {
    pub(super) languages: Vec<&'static str>,
    pub(super) default_language: &'static str,
    idle_grace: u32,
    pub(super) overlay_limit: u32,
    pub(super) session_ttl: u32,
    refresh_interval: u32,
    refresh_batch: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            languages: LANGUAGES.to_vec(),
            default_language: "zh-TW",
            idle_grace: 180,
            overlay_limit: 5,
            session_ttl: 4500,
            refresh_interval: 20,
            refresh_batch: 256,
        }
    }
}

impl Settings {
    fn from_env() -> Result<Self> {
        let mut config = Self::default();
        if let Some(value) = env_var("MEDIA_OVERLAY_ALLOWED_LANGUAGES")? {
            config.languages.clear();
            for language in value
                .split(',')
                .filter(|language| !language.trim().is_empty())
            {
                let canonical = canonical_language(language).ok_or_else(|| {
                    anyhow!(
                        "MEDIA_OVERLAY_ALLOWED_LANGUAGES contains an unsupported overlay language"
                    )
                })?;
                if !config.languages.contains(&canonical) {
                    config.languages.push(canonical);
                }
            }
            ensure!(
                !config.languages.is_empty(),
                "MEDIA_OVERLAY_ALLOWED_LANGUAGES must include at least one language"
            );
        }
        let default_language =
            env_var("MEDIA_DEFAULT_OVERLAY_LANGUAGE")?.unwrap_or_else(|| "zh-TW".into());
        config.default_language = canonical_language(&default_language)
            .filter(|code| config.languages.contains(code))
            .ok_or_else(|| {
                anyhow!("MEDIA_DEFAULT_OVERLAY_LANGUAGE must be an allowed overlay language")
            })?;
        config.idle_grace = bounded_env(
            "MEDIA_PUBLISHER_IDLE_GRACE_SECONDS",
            config.idle_grace,
            30..=u32::MAX,
        )?;
        config.overlay_limit = bounded_env(
            "MEDIA_OVERLAY_MAX_ACTIVE_LANGUAGES_PER_STREAM",
            config.overlay_limit,
            1..=u32::MAX,
        )?;
        config.session_ttl = bounded_env(
            "STREAM_PLAYBACK_SESSION_TTL_SECONDS",
            config.session_ttl,
            1..=u32::MAX,
        )?;
        config.refresh_interval = bounded_env(
            "STREAM_PLAYBACK_SESSION_REFRESH_INTERVAL_SECONDS",
            config.refresh_interval,
            1..=u32::MAX,
        )?;
        config.refresh_batch = bounded_env(
            "STREAM_PLAYBACK_REFRESH_BATCH_SIZE",
            config.refresh_batch,
            1..=2048,
        )?;
        Ok(config)
    }
}

fn env_var(name: &str) -> Result<Option<String>> {
    match env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(env::VarError::NotUnicode(_)) => Err(anyhow!("{name} must be Unicode")),
    }
}

pub(super) fn bounded_env(
    name: &str,
    default: u32,
    range: std::ops::RangeInclusive<u32>,
) -> Result<u32> {
    let Some(value) = env_var(name)? else {
        return Ok(default);
    };
    value
        .trim()
        .parse::<u32>()
        .ok()
        .filter(|value| range.contains(value))
        .ok_or_else(|| anyhow!("{name} must be an integer in {range:?}"))
}

pub(super) fn canonical_language(language: &str) -> Option<&'static str> {
    match language
        .trim()
        .replace('_', "-")
        .to_ascii_lowercase()
        .as_str()
    {
        "en" | "en-gb" | "en-us" => Some("en"),
        "zh" | "zh-tw" | "zh-hant" | "zh-hk" | "zh-mo" => Some("zh-TW"),
        "zh-cn" | "zh-hans" | "zh-sg" => Some("zh-CN"),
        "ja" | "jp" | "ja-jp" => Some("ja"),
        "vi" | "vi-vn" => Some("vi"),
        "id" | "id-id" => Some("id"),
        "fr" | "fr-fr" | "fr-ca" => Some("fr"),
        "th" | "th-th" => Some("th"),
        _ => None,
    }
}

pub(super) fn normalise_label_language(language: &str) -> &'static str {
    canonical_language(language).unwrap_or_else(|| {
        let language = language.trim().replace('_', "-").to_ascii_lowercase();
        let base = language.split('-').next().unwrap_or_default();
        LANGUAGES
            .into_iter()
            .find(|code| *code == base)
            .unwrap_or("en")
    })
}

struct MediaRequest {
    path: MediaPath,
    token: String,
}

impl MediaRequest {
    fn parse(request: &Request) -> Result<Self, Failure> {
        let original = request
            .headers()
            .get("x-original-uri")
            .and_then(|value| value.to_str().ok())
            .filter(|path| !path.is_empty());
        let forwarded = request
            .headers()
            .get("x-forwarded-uri")
            .and_then(|value| value.to_str().ok())
            .filter(|path| !path.is_empty());
        let uri = original
            .or(forwarded)
            .unwrap_or_else(|| request.uri().path());
        let path = MediaPath::parse(uri).ok_or(Failure::InvalidPath)?;
        let token = request
            .uri()
            .query()
            .and_then(|query| extract_token(query, true))
            .or_else(|| {
                [original, forwarded].into_iter().flatten().find_map(|uri| {
                    let without_fragment = uri.split('#').next().unwrap_or_default();
                    let (_, query) = without_fragment.split_once('?')?;
                    extract_token(query, false)
                })
            })
            .ok_or(Failure::Missing)?;
        Ok(Self { path, token })
    }
}

fn extract_token(query: &str, direct: bool) -> Option<String> {
    for name in ["mt", "media_token"] {
        let mut values = form_urlencoded::parse(query.as_bytes())
            .filter_map(|(key, value)| (key == name).then_some(value));
        let matched = if direct {
            values.last().filter(|value| !value.is_empty())
        } else {
            values.find(|value| !value.is_empty())
        };
        if let Some(value) = matched {
            return Some(value.into_owned());
        }
    }
    None
}

struct MediaPath {
    path: String,
    overlay: Option<(String, &'static str)>,
}

impl MediaPath {
    fn parse(uri: &str) -> Option<Self> {
        let mut parts = uri
            .split('?')
            .next()?
            .split('/')
            .filter(|part| !part.is_empty());
        parts.find(|part| percent_decode_str(part).decode_utf8_lossy() == "media")?;
        let mut path = percent_decode_str(parts.next()?).decode_utf8().ok()?;
        if path == "webrtc" {
            path = percent_decode_str(parts.next()?).decode_utf8().ok()?;
        }
        if !path.starts_with("hazard_") {
            return None;
        }
        let overlay = match path.rsplit_once("_annotated_") {
            Some((base, encoded)) if !encoded.is_empty() => {
                let decoded = URL_SAFE_NO_PAD
                    .decode(encoded)
                    .or_else(|_| URL_SAFE.decode(encoded));
                // The marker can occur inside a legal Base64 camera name. In
                // that case an invalid suffix still permits exact clean scope.
                decoded
                    .ok()
                    .and_then(|bytes| String::from_utf8(bytes).ok())
                    .map(|language| (base.to_owned(), normalise_label_language(&language)))
            }
            _ => None,
        };
        Some(Self {
            path: path.into_owned(),
            overlay,
        })
    }
}

#[derive(Deserialize)]
struct Capability {
    parent: String,
    playback_revision: Option<String>,
    playback_language: Option<String>,
    access_revocation_key: Option<String>,
    oidc_revocation_key: Option<String>,
    #[serde(default)]
    parent_generation: u64,
    expires_at: u64,
    user_active: Option<bool>,
    site: String,
    cameras: Vec<String>,
    profile: String,
    quality: Option<String>,
}

impl Capability {
    fn allows_playlist(&self, parent: &str, revision: &str) -> bool {
        if let Some(current) = self.playback_revision.as_deref() {
            return self.parent == parent && current == revision;
        }
        // Python stored opaque page capabilities under the native account or
        // browser session, with no playback revision. The camera, profile,
        // and rendition remain enforced, along with the parent's generation.
        self.parent.starts_with("native:user:") || self.parent.starts_with("bff:session:")
    }

    fn allows(&self, path: &MediaPath) -> bool {
        let suffix = match self.quality.as_deref() {
            Some("detail") => "",
            Some("preview") => PREVIEW_SUFFIX,
            _ => return false,
        };
        let selected = match self.profile.as_str() {
            "clean" => &path.path,
            "overlay" => match &path.overlay {
                Some((base, language))
                    if self
                        .playback_language
                        .as_deref()
                        .is_none_or(|allowed| allowed == *language) =>
                {
                    base
                }
                _ => return false,
            },
            _ => return false,
        };
        let prefix = format!("hazard_{}_", URL_SAFE_NO_PAD.encode(&self.site));
        let Some(encoded_camera) = selected
            .strip_prefix(&prefix)
            .and_then(|path| path.strip_suffix(suffix))
        else {
            return false;
        };
        self.cameras
            .iter()
            .any(|camera| URL_SAFE_NO_PAD.encode(camera) == encoded_camera)
    }
}

struct Leases {
    producer: String,
    media_index: String,
    demand_index: String,
    refresh: String,
    overlay_position: usize,
    language_keys: Vec<String>,
}

fn overlay_producer(path: &str, language: &str) -> String {
    format!(
        "media_overlay_demand:{path}:{}",
        URL_SAFE_NO_PAD.encode(language)
    )
}

impl Leases {
    fn new(
        path: &MediaPath,
        config: &Settings,
        demand_prefix: &str,
        capability: &Capability,
    ) -> Self {
        let overlay = path
            .overlay
            .as_ref()
            .filter(|_| capability.profile == "overlay");
        let (producer, demand_index, overlay_position, language_keys) =
            if let Some((base, language)) = overlay {
                let index = format!(
                    "stream_playback_demand_session:overlay:{base}:{}",
                    URL_SAFE_NO_PAD.encode(language)
                );
                if let Some(position) = config.languages.iter().position(|code| code == language) {
                    let clean = if capability.quality.as_deref() == Some("preview") {
                        base.strip_suffix(PREVIEW_SUFFIX).unwrap_or(base)
                    } else {
                        base
                    };
                    let preview = format!("{clean}{PREVIEW_SUFFIX}");
                    let keys = config
                        .languages
                        .iter()
                        .flat_map(|language| {
                            [
                                format!("{demand_prefix}{}", overlay_producer(clean, language)),
                                format!("{demand_prefix}{}", overlay_producer(&preview, language)),
                            ]
                        })
                        .collect();
                    (overlay_producer(base, language), index, position + 1, keys)
                } else {
                    (String::new(), index, 0, Vec::new())
                }
            } else {
                (
                    format!("media_clean_demand:{}", path.path),
                    format!("stream_playback_demand_session:clean:{}", path.path),
                    0,
                    Vec::new(),
                )
            };
        Self {
            producer: if producer.is_empty() {
                producer
            } else {
                format!("{demand_prefix}{producer}")
            },
            media_index: format!("stream_playback_media_session:{}", path.path),
            demand_index,
            refresh: format!("stream_playback_session_refresh:{}", path.path),
            overlay_position,
            language_keys,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::process::{Child, Command, Stdio};
    use std::time::{SystemTime, UNIX_EPOCH};

    use axum::body::Body;
    use redis::AsyncCommands;

    use super::*;

    #[test]
    fn selects_scope_from_the_encoded_camera_and_quality() {
        let json = json!({
            "parent": "service:fixture",
            "expires_at": 2_000_000_000,
            "site": "場區",
            "cameras": ["Cam 1", "Cam 2"],
            "profile": "overlay",
            "quality": "preview",
        });
        let capability: Capability = serde_json::from_value(json).unwrap();
        let site = URL_SAFE_NO_PAD.encode("場區");
        let camera = URL_SAFE_NO_PAD.encode("Cam 1");
        let language = URL_SAFE_NO_PAD.encode("zh_HK");
        let prefix = format!("/hazard/media/webrtc/hazard_{site}_{camera}");
        let path =
            |suffix: &str| MediaPath::parse(&format!("{prefix}{suffix}/index.m3u8")).unwrap();
        assert!(capability.allows(&path(&format!("_preview_annotated_{language}"))));
        assert!(!capability.allows(&path(&format!("_annotated_{language}"))));
        assert!(!capability.allows(&path("_preview")));
        let third_camera = URL_SAFE_NO_PAD.encode("Cam 3");
        assert!(!capability.allows(&MediaPath::parse(
            &format!("/hazard/media/hazard_{site}_{third_camera}_preview_annotated_{language}/index.m3u8"),
        ).unwrap()));
    }

    #[test]
    fn media_tokens_come_from_url_parameters() {
        let request = Request::builder()
            .uri("/media-auth?mt=old&mt=direct")
            .header(
                "X-Original-URI",
                "/hazard/media/%68azard_c2l0ZQ_Y2Ft/index.m3u8?mt=original",
            )
            .header("Authorization", "Bearer ignored")
            .body(Body::empty())
            .unwrap();
        let parsed = MediaRequest::parse(&request).unwrap();
        assert_eq!(parsed.path.path, "hazard_c2l0ZQ_Y2Ft");
        assert_eq!(parsed.token, "direct");
        assert_eq!(
            extract_token("mt=first&mt=second", false).as_deref(),
            Some("first")
        );
        let request = Request::builder()
            .uri("/media-auth")
            .header(
                "X-Original-URI",
                "/hazard/media/hazard_c2l0ZQ_Y2Ft/segment.m4s",
            )
            .header("Authorization", "Bearer ignored")
            .header("Cookie", "mt=ignored")
            .body(Body::empty())
            .unwrap();
        assert_eq!(MediaRequest::parse(&request).err(), Some(Failure::Missing));
        assert!(MediaPath::parse("/hazard/api/media-auth").is_none());
    }

    #[test]
    fn exposes_the_media_auth_challenge_headers() {
        let response = Failure::Expired.into_response();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            response.headers()["x-media-auth-error"],
            "expired_media_session"
        );
        assert_eq!(
            response.headers()[header::WWW_AUTHENTICATE],
            "Bearer error=\"expired_media_session\""
        );
        let response = Failure::ScopeDenied.into_response();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        assert!(!response.headers().contains_key(header::WWW_AUTHENTICATE));
    }

    struct RedisFixture {
        child: Child,
        _directory: tempfile::TempDir,
        connection: ConnectionManager,
    }

    impl RedisFixture {
        async fn new() -> Self {
            let directory = tempfile::tempdir().unwrap();
            let socket = directory.path().join("redis.sock");
            let mut child = Command::new("redis-server")
                .args(["--port", "0", "--unixsocket"])
                .arg(&socket)
                .args(["--save", "", "--appendonly", "no"])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .expect(
                    "redis-server must be installed to run media authorisation integration tests",
                );
            let client = redis::Client::open(format!("redis+unix://{}", socket.display())).unwrap();
            let connection = tokio::time::timeout(Duration::from_secs(3), async {
                loop {
                    if socket.exists() {
                        return client.get_connection_manager().await;
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await;
            let Ok(Ok(connection)) = connection else {
                let _ = child.kill();
                let _ = child.wait();
                panic!("isolated Redis did not become ready");
            };
            Self {
                child,
                _directory: directory,
                connection,
            }
        }
    }

    impl Drop for RedisFixture {
        fn drop(&mut self) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }

    fn seconds() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    async fn seed_capability(connection: &mut ConnectionManager, data: &serde_json::Value) {
        let key = format!("media:session:{:x}", Sha256::digest(b"test-token"));
        let _: () = connection.set(&key, data.to_string()).await.unwrap();
    }

    fn fixture_capability(profile: &str, quality: &str) -> serde_json::Value {
        json!({
            "parent": "bff:session:fixture",
            "expires_at": seconds() + 600,
            "parent_generation": 3,
            "user_active": true,
            "site": "site",
            "cameras": ["cam", "cam2"],
            "profile": profile,
            "quality": quality,
        })
    }

    async fn authorise_test(auth: &MediaAuth, suffix: &str) -> Result<(), Failure> {
        let request = Request::builder()
            .uri("/media-auth")
            .header(
                "X-Forwarded-URI",
                format!(
                    "/hazard/media/hazard_c2l0ZQ_Y2Ft{suffix}/index.m3u8?media_token=test-token"
                ),
            )
            .body(Body::empty())
            .unwrap();
        auth.authorise(MediaRequest::parse(&request)?).await
    }

    async fn allow_parent(connection: &mut ConnectionManager, generation: u64) {
        let key = format!(
            "media:parent:generation:{:x}",
            Sha256::digest(b"bff:session:fixture"),
        );
        let _: () = connection.set(key, generation).await.unwrap();
        let _: () = connection.set("bff:session:fixture", "{}").await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires an isolated redis-server process"]
    async fn rejects_expired_revoked_inactive_and_out_of_scope_capabilities() {
        let mut redis = RedisFixture::new().await;
        let auth = MediaAuth::with_settings(redis.connection.clone(), Settings::default());
        let mut data = fixture_capability("clean", "detail");
        seed_capability(&mut redis.connection, &data).await;
        assert_eq!(authorise_test(&auth, "").await, Err(Failure::Expired));
        allow_parent(&mut redis.connection, 4).await;
        assert_eq!(authorise_test(&auth, "").await, Err(Failure::Expired));
        allow_parent(&mut redis.connection, 3).await;
        assert_eq!(
            authorise_test(&auth, "_preview").await,
            Err(Failure::ScopeDenied)
        );
        data["user_active"] = json!(false);
        seed_capability(&mut redis.connection, &data).await;
        assert_eq!(authorise_test(&auth, "").await, Err(Failure::Inactive));
        let demand = "media_clean_demand:hazard_c2l0ZQ_Y2Ft";
        assert!(!redis.connection.exists::<_, bool>(demand).await.unwrap());
        data["user_active"] = json!(true);
        seed_capability(&mut redis.connection, &data).await;
        assert_eq!(authorise_test(&auth, "").await, Ok(()));
        assert!(redis.connection.ttl::<_, i64>(demand).await.unwrap() > 180);
        let _: () = redis.connection.del("bff:session:fixture").await.unwrap();
        assert_eq!(authorise_test(&auth, "").await, Err(Failure::Expired));
        allow_parent(&mut redis.connection, 3).await;
        data["expires_at"] = json!(seconds() - 10);
        seed_capability(&mut redis.connection, &data).await;
        assert_eq!(authorise_test(&auth, "").await, Err(Failure::Expired));
    }

    #[tokio::test]
    #[ignore = "requires an isolated redis-server process"]
    async fn scoped_playback_capability_requires_current_revision_language_and_issuer() {
        let mut redis = RedisFixture::new().await;
        let auth = MediaAuth::with_settings(redis.connection.clone(), Settings::default());
        let mut capability = fixture_capability("overlay", "preview");
        let parent = "stream_playback_session:session";
        capability["parent"] = json!(parent);
        capability["parent_generation"] = json!(0);
        capability["playback_revision"] = json!("first");
        capability["playback_language"] = json!("zh-TW");
        capability["oidc_revocation_key"] = json!("fixture:oidc-revoked");
        seed_capability(&mut redis.connection, &capability).await;
        let _: () = redis
            .connection
            .set(parent, r#"{"revision":"first"}"#)
            .await
            .unwrap();
        assert_eq!(
            authorise_test(&auth, "_preview_annotated_ZW4").await,
            Err(Failure::ScopeDenied)
        );
        let suffix = "_preview_annotated_emgtVFc";
        assert_eq!(authorise_test(&auth, suffix).await, Ok(()));
        let _: () = redis
            .connection
            .set("fixture:oidc-revoked", "1")
            .await
            .unwrap();
        assert_eq!(authorise_test(&auth, suffix).await, Err(Failure::Expired));
        let _: () = redis.connection.del("fixture:oidc-revoked").await.unwrap();
        let _: () = redis
            .connection
            .set(parent, r#"{"revision":"second"}"#)
            .await
            .unwrap();
        assert_eq!(authorise_test(&auth, suffix).await, Err(Failure::Expired));
        let _: () = redis.connection.del(parent).await.unwrap();
        assert_eq!(authorise_test(&auth, suffix).await, Err(Failure::Expired));
    }

    #[tokio::test]
    #[ignore = "requires an isolated redis-server process"]
    async fn overlay_language_limit_is_shared_across_renditions() {
        let mut redis = RedisFixture::new().await;
        let auth = MediaAuth::with_settings(
            redis.connection.clone(),
            Settings {
                overlay_limit: 1,
                ..Settings::default()
            },
        );
        allow_parent(&mut redis.connection, 3).await;
        seed_capability(
            &mut redis.connection,
            &fixture_capability("overlay", "preview"),
        )
        .await;
        let detail = overlay_producer("hazard_c2l0ZQ_Y2Ft", "zh-TW");
        let _: () = redis.connection.set_ex(detail, "1", 30).await.unwrap();
        assert_eq!(
            authorise_test(&auth, "_preview_annotated_ZW4").await,
            Err(Failure::LanguageLimit)
        );
        assert_eq!(
            authorise_test(&auth, "_preview_annotated_emgtVFc").await,
            Ok(())
        );
        let preview = overlay_producer("hazard_c2l0ZQ_Y2Ft_preview", "zh-TW");
        assert!(redis.connection.exists::<_, bool>(preview).await.unwrap());
    }

    #[tokio::test]
    #[ignore = "requires an isolated redis-server process"]
    async fn renews_only_a_bounded_batch_and_removes_stale_indices() {
        let mut redis = RedisFixture::new().await;
        let auth = MediaAuth::with_settings(
            redis.connection.clone(),
            Settings {
                refresh_batch: 2,
                ..Settings::default()
            },
        );
        let path = "hazard_c2l0ZQ_Y2Ft";
        let index = format!("stream_playback_media_session:{path}");
        let demand = format!("stream_playback_demand_session:clean:{path}");
        let stamp = seconds();
        allow_parent(&mut redis.connection, 3).await;
        seed_capability(
            &mut redis.connection,
            &fixture_capability("clean", "detail"),
        )
        .await;
        for (member, score) in [
            ("stale", stamp + 1),
            ("one", stamp + 2),
            ("two", stamp + 3),
            ("expired", stamp - 10),
        ] {
            let _: () = redis.connection.zadd(&index, member, score).await.unwrap();
            let _: () = redis.connection.zadd(&demand, member, score).await.unwrap();
            if member == "one" || member == "two" {
                let _: () = redis
                    .connection
                    .set_ex(
                        format!("stream_playback_session:{member}"),
                        json!({"profile": "clean", "base_media_path": path}).to_string(),
                        60,
                    )
                    .await
                    .unwrap();
            }
        }
        assert_eq!(authorise_test(&auth, "").await, Ok(()));
        assert!(
            redis
                .connection
                .ttl::<_, i64>("stream_playback_session:one")
                .await
                .unwrap()
                > 60
        );
        assert!(
            redis
                .connection
                .ttl::<_, i64>("stream_playback_session:two")
                .await
                .unwrap()
                <= 60
        );
        assert_eq!(
            redis
                .connection
                .zrange::<_, Vec<String>>(&index, 0, -1)
                .await
                .unwrap(),
            vec!["two", "one"]
        );
        assert_eq!(
            redis
                .connection
                .zrange::<_, Vec<String>>(&demand, 0, -1)
                .await
                .unwrap(),
            vec!["two", "one"]
        );
        assert_eq!(authorise_test(&auth, "").await, Ok(()));
        assert!(
            redis
                .connection
                .ttl::<_, i64>("stream_playback_session:two")
                .await
                .unwrap()
                <= 60
        );
        let _: () = redis
            .connection
            .del(format!("stream_playback_session_refresh:{path}"))
            .await
            .unwrap();
        assert_eq!(authorise_test(&auth, "").await, Ok(()));
        assert!(
            redis
                .connection
                .ttl::<_, i64>("stream_playback_session:two")
                .await
                .unwrap()
                > 60
        );
    }
}
