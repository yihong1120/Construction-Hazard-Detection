mod legal;
mod oidc;
mod proxy;
mod session;

use std::{collections::HashMap, sync::Arc, time::Duration};

use anyhow::{Context, Result, ensure};
use aws_lc_rs::aead::{AES_256_GCM, RandomizedNonceKey};
use axum::{
    Router,
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{any, get, post},
};
use redis::aio::ConnectionManager;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use url::{Host, Url};

use crate::{ApiError, AppState};

const STATE_TTL: u64 = 300;
const SESSION_MAXIMUM: u64 = 7_776_000;
const MAX_TOKEN_RESPONSE: usize = 64 * 1024;
const SESSION_COOKIE: &str = "__Host-vn_session";
const FLOW_COOKIE: &str = "__Host-vn_oidc_flow";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Config {
    public_api_base_url: String,
    authorization_endpoint: String,
    token_endpoint: String,
    end_session_endpoint: String,
    account_url: String,
    post_logout_redirect_uri: String,
    client_id: String,
    client_secret_env: String,
    session_key_env: String,
    #[serde(default = "identity_provider")]
    identity_provider: String,
    #[serde(default = "session_ttl")]
    session_ttl_seconds: u64,
    #[serde(default)]
    shared_session_store: bool,
    #[serde(default)]
    upstreams: HashMap<String, String>,
    company_ai_network_proxy_key_env: Option<String>,
}

fn identity_provider() -> String {
    "keycloak".into()
}

fn session_ttl() -> u64 {
    604_800
}

impl Config {
    fn parsed(&self) -> Result<Urls> {
        let public_base = web_url(&self.public_api_base_url)?;
        ensure!(
            !public_base.path().contains('%'),
            "Invalid BFF API root path"
        );
        let root = public_base.as_str().trim_end_matches('/');
        let browser_origin = public_base.origin().ascii_serialization();
        let redirect_uri = format!("{browser_origin}/bff/auth/oidc/callback");
        let post_logout = web_url(&self.post_logout_redirect_uri)?;
        ensure!(
            post_logout.origin() == public_base.origin(),
            "BFF post-logout URI must use its public API origin"
        );
        let authorization = web_url(&self.authorization_endpoint)?;
        let token = web_url(&self.token_endpoint)?;
        let end_session = web_url(&self.end_session_endpoint)?;
        let account = web_url(&self.account_url)?;
        ensure!(
            [token.origin(), end_session.origin(), account.origin()]
                .iter()
                .all(|origin| *origin == authorization.origin()),
            "BFF identity endpoints must share the configured provider origin"
        );
        let upstreams = self
            .upstreams
            .iter()
            .map(|(service, endpoint)| {
                ensure!(
                    proxy::allowed_service(service),
                    "Invalid BFF upstream service"
                );
                let url = web_url(endpoint)?;
                ensure!(
                    proxy::allowed_upstream(service, &url, &public_base),
                    "BFF upstreams require a loopback origin or the public Chat API path"
                );
                Ok((service.clone(), url))
            })
            .collect::<Result<_>>()?;
        Ok(Urls {
            public_api_base_url: root.to_owned(),
            browser_origin,
            browser_authority: public_base[UrlPart::BeforeHost..UrlPart::AfterPort].to_owned(),
            secure: public_base.scheme() == "https",
            redirect_uri,
            authorization,
            token,
            end_session,
            account,
            post_logout,
            upstreams,
        })
    }

    pub(crate) fn validate(&self) -> Result<()> {
        self.parsed()?;
        ensure!(
            (60..=SESSION_MAXIMUM).contains(&self.session_ttl_seconds)
                && !self.client_id.is_empty()
                && self.client_id.len() <= 255
                && !self.identity_provider.is_empty()
                && self.identity_provider.len() <= 20,
            "Invalid BFF client or session lifetime"
        );
        ensure!(
            [&self.client_secret_env, &self.session_key_env]
                .into_iter()
                .all(|name| !name.is_empty()
                    && name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')),
            "BFF secrets require environment variable names"
        );
        if self.upstreams.contains_key("ai_access") {
            ensure!(
                self.company_ai_network_proxy_key_env
                    .as_ref()
                    .is_some_and(|name| {
                        !name.is_empty()
                            && name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
                    }),
                "AI Access requires the company network proxy key environment variable name"
            );
        }
        Ok(())
    }
}

use url::Position as UrlPart;

struct Urls {
    public_api_base_url: String,
    browser_origin: String,
    browser_authority: String,
    secure: bool,
    redirect_uri: String,
    authorization: Url,
    token: Url,
    end_session: Url,
    account: Url,
    post_logout: Url,
    upstreams: HashMap<String, Url>,
}

fn loopback(url: &Url) -> bool {
    match url.host() {
        Some(Host::Ipv4(ip)) => ip.is_loopback(),
        Some(Host::Ipv6(ip)) => ip.is_loopback(),
        _ => false,
    }
}

fn web_url(value: &str) -> Result<Url> {
    let url = Url::parse(value).map_err(|_| anyhow::anyhow!("Invalid BFF URL"))?;
    ensure!(
        url.has_host()
            && (url.scheme() == "https" || url.scheme() == "http" && loopback(&url))
            && url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none(),
        "BFF endpoints require HTTPS or literal loopback HTTP"
    );
    Ok(url)
}

#[derive(Clone)]
pub(crate) struct Bff(Arc<Inner>);

struct Inner {
    urls: Urls,
    client_id: String,
    client_secret: String,
    identity_provider: String,
    cors_origins: Vec<String>,
    session_ttl: u64,
    redis: ConnectionManager,
    session_key: RandomizedNonceKey,
    shared_session_key: Option<fernet::Fernet>,
    oidc_http: reqwest::Client,
    upstream_http: reqwest::Client,
    company_ai_network_proxy_key: Option<HeaderValue>,
}

impl Bff {
    pub(crate) fn new(
        config: Config,
        redis: ConnectionManager,
        cors_origins: Vec<String>,
    ) -> Result<Self> {
        config.validate()?;
        let urls = config.parsed()?;
        let client_secret = std::env::var(&config.client_secret_env)
            .context("Configure the BFF OIDC client secret environment variable")?;
        let secret = std::env::var(&config.session_key_env)
            .context("Configure the BFF session encryption environment variable")?;
        ensure!(
            !client_secret.trim().is_empty() && secret.len() >= 32,
            "BFF requires a client secret and a session encryption key of at least 32 bytes"
        );
        let session_key = RandomizedNonceKey::new(&AES_256_GCM, &Sha256::digest(&secret))
            .map_err(|_| anyhow::anyhow!("Cannot initialise BFF session encryption"))?;
        let shared_session_key = if config.shared_session_store {
            use base64::{Engine, engine::general_purpose::URL_SAFE};
            ensure!(
                urls.secure,
                "The shared browser session store requires HTTPS"
            );
            Some(
                fernet::Fernet::new(&URL_SAFE.encode(Sha256::digest(secret.trim())))
                    .context("Cannot initialise shared BFF session encryption")?,
            )
        } else {
            None
        };
        let company_ai_network_proxy_key = if urls.upstreams.contains_key("ai_access") {
            let name = config
                .company_ai_network_proxy_key_env
                .as_ref()
                .context("Set the AI Access company network proxy key environment variable name")?;
            let value =
                std::env::var(name).context("Set the AI Access company network proxy key")?;
            ensure!(
                value.len() >= 32,
                "AI Access company network proxy key is too short"
            );
            let mut header = HeaderValue::from_str(&value)
                .map_err(|_| anyhow::anyhow!("Invalid AI Access company network proxy key"))?;
            header.set_sensitive(true);
            Some(header)
        } else {
            None
        };
        let client = || {
            reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(3))
                .redirect(reqwest::redirect::Policy::none())
                .no_proxy()
        };
        Ok(Self(Arc::new(Inner {
            urls,
            client_id: config.client_id,
            client_secret,
            identity_provider: config.identity_provider,
            cors_origins,
            session_ttl: config.session_ttl_seconds,
            redis,
            session_key,
            shared_session_key,
            oidc_http: client().timeout(Duration::from_secs(10)).build()?,
            upstream_http: client().read_timeout(Duration::from_secs(30)).build()?,
            company_ai_network_proxy_key,
        })))
    }

    fn host(&self, headers: &HeaderMap) -> Result<(), ApiError> {
        if single_header(headers, header::HOST) != Some(self.0.urls.browser_authority.as_str()) {
            return Err(ApiError::new(
                StatusCode::CONFLICT,
                "deployment_origin_mismatch",
            ));
        }
        Ok(())
    }

    fn session_cookie(&self) -> &str {
        if self.0.urls.secure {
            SESSION_COOKIE
        } else {
            "chd_rust_session"
        }
    }

    fn flow_cookie(&self) -> &str {
        if self.0.urls.secure {
            FLOW_COOKIE
        } else {
            "chd_rust_oidc_flow"
        }
    }

    fn set_cookie(&self, response: &mut Response, name: &str, value: &str, ttl: u64) {
        let secure = if self.0.urls.secure { "; Secure" } else { "" };
        let cookie =
            format!("{name}={value}; Path=/; Max-Age={ttl}; HttpOnly; SameSite=Lax{secure}");
        match HeaderValue::from_str(&cookie) {
            Ok(value) => {
                response.headers_mut().append(header::SET_COOKIE, value);
            }
            Err(_) => unreachable!("internal opaque cookie is ASCII"),
        }
    }
}

fn state_bff(state: &AppState, headers: &HeaderMap) -> Result<Bff, ApiError> {
    let bff = state
        .bff
        .clone()
        .ok_or_else(|| ApiError::new(StatusCode::NOT_FOUND, "bff_route_not_allowed"))?;
    bff.host(headers)?;
    Ok(bff)
}

fn single_header(headers: &HeaderMap, name: header::HeaderName) -> Option<&str> {
    let values = headers.get_all(name);
    let mut values = values.iter();
    let first = values.next()?.to_str().ok()?;
    values.next().is_none().then_some(first)
}

fn authentication_required() -> ApiError {
    ApiError::new(StatusCode::UNAUTHORIZED, "app_session_expired")
}

fn unavailable() -> ApiError {
    ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "bff_session_unavailable")
}

fn configured_deployment_changed() -> ApiError {
    ApiError::new(StatusCode::CONFLICT, "deployment_configuration_changed")
}

fn no_store(response: impl IntoResponse) -> Response {
    let mut response = response.into_response();
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response.headers_mut().insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("no-referrer"),
    );
    response
}

fn redirect(status: StatusCode, location: &str) -> Result<Response, ApiError> {
    let location = HeaderValue::from_str(location).map_err(|_| unavailable())?;
    Ok(no_store((status, [(header::LOCATION, location)])))
}

pub(crate) fn routes() -> Router<AppState> {
    Router::new()
        .route("/legal/documents", get(legal::documents))
        .route("/bff/auth/account", get(oidc::account))
        .route("/bff/auth/oidc/login", get(oidc::login))
        .route("/bff/auth/oidc/callback", get(oidc::callback))
        .route("/bff/auth/oidc/logout", get(oidc::global_logout))
        .route("/bff/auth/session", get(session::current))
        .route("/bff/auth/csrf", get(session::csrf))
        .route("/bff/auth/logout", post(session::logout))
        .route("/bff/chat", any(proxy::chat_root))
        .route("/bff/chat/", any(proxy::chat_root))
        .route("/bff/{service}/{*path}", any(proxy::forward))
}
