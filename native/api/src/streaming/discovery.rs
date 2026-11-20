use std::env::{self, VarError};

use anyhow::{Context, Result, ensure};
use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::get,
};
use base64::{Engine, engine::general_purpose::STANDARD};
use hmac::{Hmac, Mac};
use serde::Serialize;
use sha1::Sha1;

use crate::{ApiError, AppState, auth::Grant};

pub(crate) fn routes() -> Router<AppState> {
    Router::new()
        .route("/labels", get(labels))
        .route("/webrtc/ice-servers", get(ice_servers))
}

#[derive(Clone)]
pub(crate) struct IceSettings {
    stun_urls: Vec<String>,
    turn: Option<Turn>,
}

#[derive(Clone)]
struct Turn {
    urls: Vec<String>,
    secret: String,
    ttl_seconds: i64,
}

impl IceSettings {
    pub(crate) fn from_env() -> Result<Option<Self>> {
        let stun = optional_env("STREAMING_WEBRTC_STUN_URLS")?;
        let turn = optional_env("STREAMING_WEBRTC_TURN_URLS")?;
        let Some(stun) = stun else {
            ensure!(turn.is_none(), "Configure STUN URLs alongside TURN URLs");
            return Ok(None);
        };
        let stun_urls = parse_urls(&stun, &["stun", "stuns"], "STUN")?;
        let turn = if let Some(turn) = turn {
            let secret = optional_env("STREAMING_WEBRTC_TURN_SHARED_SECRET")?
                .filter(|secret| !secret.trim().is_empty())
                .context("Set STREAMING_WEBRTC_TURN_SHARED_SECRET")?;
            let ttl_seconds = optional_env("STREAMING_WEBRTC_TURN_TTL_SECONDS")?
                .context("Set STREAMING_WEBRTC_TURN_TTL_SECONDS")?
                .parse::<i64>()
                .map_err(|_| anyhow::anyhow!("Invalid STREAMING_WEBRTC_TURN_TTL_SECONDS"))?;
            Some(Turn {
                urls: parse_urls(&turn, &["turn", "turns"], "TURN")?,
                secret,
                ttl_seconds,
            })
        } else {
            None
        };
        Ok(Some(Self { stun_urls, turn }))
    }

    fn servers(&self, subject: &str, now: i64) -> IceServers<'_> {
        let mut servers = vec![IceServer {
            urls: &self.stun_urls,
            username: None,
            credential: None,
        }];
        if let Some(turn) = &self.turn {
            let username = format!("{}:{subject}", now.saturating_add(turn.ttl_seconds.max(60)));
            // coturn's REST credential protocol specifies HMAC-SHA1.
            let mut mac = Hmac::<Sha1>::new_from_slice(turn.secret.as_bytes())
                .expect("HMAC accepts arbitrary key lengths");
            mac.update(username.as_bytes());
            let credential = STANDARD.encode(mac.finalize().into_bytes());
            servers.push(IceServer {
                urls: &turn.urls,
                username: Some(username),
                credential: Some(credential),
            });
        }
        IceServers {
            ice_servers: servers,
        }
    }
}

fn optional_env(name: &str) -> Result<Option<String>> {
    match env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(VarError::NotPresent) => Ok(None),
        Err(_) => anyhow::bail!("Invalid {name}"),
    }
}

fn parse_urls(value: &str, schemes: &[&str], name: &str) -> Result<Vec<String>> {
    let urls: Vec<_> = value
        .split(',')
        .map(str::trim)
        .filter(|url| !url.is_empty())
        .map(str::to_owned)
        .collect();
    ensure!(!urls.is_empty(), "Configure at least one {name} URL");
    for url in &urls {
        let parsed = url::Url::parse(url).map_err(|_| anyhow::anyhow!("Invalid {name} URL"))?;
        ensure!(
            schemes.contains(&parsed.scheme()) && !parsed.path().is_empty(),
            "Invalid {name} URL"
        );
    }
    Ok(urls)
}

#[derive(Serialize)]
struct Labels {
    labels: Vec<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct IceServers<'a> {
    ice_servers: Vec<IceServer<'a>>,
}

#[derive(Serialize)]
struct IceServer<'a> {
    urls: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    username: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    credential: Option<String>,
}

async fn labels(State(state): State<AppState>, headers: HeaderMap) -> Result<Response, ApiError> {
    let actor = state.auth.authorize(&headers, Grant::StreamRead).await?;
    let sites = match &state.playback {
        Some(playback) if playback.has_live_scope() => {
            playback.visible_sites(&state.db, &actor).await?
        }
        _ => actor.sites(),
    };
    let mut labels: Vec<_> = sites.into_iter().map(|(_, label)| label).collect();
    labels.sort_unstable();
    Ok((
        [(header::CACHE_CONTROL, "private, no-store")],
        Json(Labels { labels }),
    )
        .into_response())
}

async fn ice_servers(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    let actor = state.auth.authorize(&headers, Grant::StreamRead).await?;
    let settings = state.ice.as_ref().ok_or_else(|| {
        ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "WebRTC ICE servers are not configured",
        )
    })?;
    let servers = settings.servers(&actor.subject, chrono::Utc::now().timestamp());
    Ok((
        [(header::CACHE_CONTROL, "private, no-store")],
        Json(servers),
    )
        .into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_short_lived_coturn_rest_credentials() {
        let settings = IceSettings {
            stun_urls: vec!["stun:relay.example:3478".into()],
            turn: Some(Turn {
                urls: vec!["turn:relay.example:3478?transport=udp".into()],
                secret: "fixture-secret".into(),
                ttl_seconds: 20,
            }),
        };
        let response = serde_json::to_value(settings.servers("fixture-subject", 1000)).unwrap();
        assert_eq!(
            response["iceServers"][0]["urls"][0],
            "stun:relay.example:3478"
        );
        assert!(response["iceServers"][0].get("credential").is_none());
        assert_eq!(
            response["iceServers"][1]["username"],
            "1060:fixture-subject"
        );
        assert_eq!(
            response["iceServers"][1]["credential"],
            "vcGYup8Qm2Dly/e4q++Uzfq9YMo="
        );
    }

    #[test]
    fn validates_configured_ice_url_schemes() {
        assert_eq!(
            parse_urls(
                " stun:localhost:3478, , stuns:localhost:5349 ",
                &["stun", "stuns"],
                "STUN"
            )
            .unwrap()
            .len(),
            2
        );
        assert!(parse_urls("https://relay.example", &["turn", "turns"], "TURN").is_err());
        assert!(parse_urls("turn:", &["turn"], "TURN").is_err());
    }
}
