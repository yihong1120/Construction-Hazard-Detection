mod link;
mod provider;
#[cfg(test)]
mod tests;

use std::{
    collections::HashMap,
    net::{IpAddr, SocketAddr},
    sync::Arc,
};

use anyhow::{Context, Result, ensure};
use aws_lc_rs::constant_time::verify_slices_are_equal;
use axum::{
    Extension, Json, Router,
    body::{Bytes, to_bytes},
    extract::{ConnectInfo, DefaultBodyLimit, Request, State},
    http::{HeaderMap, StatusCode, header, uri::Authority},
    response::{IntoResponse, Response},
    routing::post,
};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use hmac::{Hmac, Mac};
use redis::aio::ConnectionManager;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use url::Url;

use crate::{
    ApiError, AppState,
    accounts::Keycloak,
    auth::{Grant, bearer},
    tenancy::{DeploymentScope, private_response},
};

const MAX_BODY: usize = 24 * 1024;
const BEGIN_LIMIT: u64 = 20;
const BEGIN_WINDOW: u64 = 300;
const BROKER_TIME_WINDOW: u64 = 30;
const INVALID: ApiError = ApiError::new(StatusCode::UNAUTHORIZED, "native_social_exchange_invalid");
const EXPIRED: ApiError = ApiError::new(StatusCode::UNAUTHORIZED, "native_social_exchange_expired");
const LINK_EXPIRED: ApiError =
    ApiError::new(StatusCode::UNAUTHORIZED, "native_social_link_expired");
const UNAVAILABLE: ApiError = ApiError::new(
    StatusCode::SERVICE_UNAVAILABLE,
    "native_social_exchange_unavailable",
);
const BAD_REQUEST: ApiError = ApiError::new(
    StatusCode::UNPROCESSABLE_ENTITY,
    "invalid_native_social_request",
);
const REAUTHENTICATE: ApiError = ApiError::new(
    StatusCode::UNAUTHORIZED,
    "keycloak_reauthentication_required",
);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Config {
    shared_secret_env: String,
    allowed_clients: HashMap<String, Vec<String>>,
    exchange_ttl_seconds: u64,
    link_ttl_seconds: u64,
    link_max_auth_age_seconds: u64,
    #[serde(default)]
    trust_proxy_client_ip: bool,
    providers: provider::Config,
}

impl Config {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            environment_name(&self.shared_secret_env),
            "Invalid native-social secret environment name"
        );
        ensure!(
            (30..=300).contains(&self.exchange_ttl_seconds)
                && (30..=300).contains(&self.link_ttl_seconds)
                && (60..=900).contains(&self.link_max_auth_age_seconds),
            "Invalid native-social session policy"
        );
        ensure!(
            !self.allowed_clients.is_empty() && self.allowed_clients.len() <= 32,
            "Configure native-social client redirects"
        );
        for (client, redirects) in &self.allowed_clients {
            ensure!(
                (1..=128).contains(&client.len())
                    && client.bytes().all(|byte| {
                        byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')
                    }),
                "Invalid native-social client ID"
            );
            ensure!(
                !redirects.is_empty() && redirects.len() <= 32,
                "Missing native-social redirects"
            );
            for redirect in redirects {
                let url = Url::parse(redirect).context("Invalid native-social redirect")?;
                ensure!(
                    redirect.len() <= 2048
                        && !redirect.contains('*')
                        && !redirect.contains('\\')
                        && !redirect.chars().any(char::is_control)
                        && !url.cannot_be_a_base()
                        && !matches!(url.scheme(), "data" | "file" | "javascript")
                        && (url.has_host() || url.scheme() != "https")
                        && url.username().is_empty()
                        && url.password().is_none()
                        && url.fragment().is_none(),
                    "Unsafe native-social redirect URI"
                );
            }
        }
        self.providers.validate()
    }
}

pub(super) fn environment_name(name: &str) -> bool {
    (1..=128).contains(&name.len())
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

#[derive(Clone)]
struct Social(Arc<Settings>);

struct Settings {
    scope: DeploymentScope,
    allowed_clients: HashMap<String, Vec<String>>,
    authorization_endpoint: Url,
    issuer: String,
    exchange_ttl: u64,
    link_ttl: u64,
    max_auth_age: u64,
    trust_proxy: bool,
    redis: ConnectionManager,
    broker_key: Hmac<Sha256>,
    keycloak: Keycloak,
    providers: provider::Providers,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum Provider {
    Google,
    Apple,
}

impl Provider {
    fn name(self) -> &'static str {
        match self {
            Self::Google => "google",
            Self::Apple => "apple",
        }
    }
}

#[derive(Deserialize, Serialize)]
struct Binding {
    client_id: String,
    redirect_uri: String,
    code_challenge: String,
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
enum Transaction {
    LoginBegin {
        provider: Provider,
        nonce: String,
        #[serde(flatten)]
        binding: Binding,
        state: String,
    },
    LoginRedeem {
        provider: Provider,
        provider_subject_b64: String,
        #[serde(flatten)]
        binding: Binding,
    },
    LinkBegin {
        provider: Provider,
        nonce: String,
        keycloak_subject: String,
        keycloak_session_id: Option<String>,
    },
    EmailLinkConfirmation {
        provider: Provider,
        provider_subject_b64: String,
        local_user_id: i32,
        keycloak_subject: String,
    },
}

pub(crate) fn routes(
    config: Config,
    issuer: &str,
    scope: DeploymentScope,
    redis: ConnectionManager,
) -> Result<Router<AppState>> {
    config.validate()?;
    let secret = std::env::var(&config.shared_secret_env)
        .context("Set native-social broker shared secret")?;
    ensure!(
        secret.len() >= 32,
        "Native-social broker secret must contain at least 32 bytes"
    );
    let issuer = issuer.trim_end_matches('/').to_owned();
    let authorization_endpoint = Url::parse(&format!("{issuer}/protocol/openid-connect/auth"))?;
    let social = Social(Arc::new(Settings {
        scope,
        allowed_clients: config.allowed_clients,
        authorization_endpoint,
        issuer,
        exchange_ttl: config.exchange_ttl_seconds,
        link_ttl: config.link_ttl_seconds,
        max_auth_age: config.link_max_auth_age_seconds,
        trust_proxy: config.trust_proxy_client_ip,
        redis,
        broker_key: Hmac::new_from_slice(secret.as_bytes()).expect("HMAC can use any key length"),
        keycloak: Keycloak::from_env()?
            .context("Set the Keycloak linker credentials for native social")?,
        providers: provider::Providers::new(config.providers)?,
    }));
    Ok(router(social))
}

fn router(social: Social) -> Router<AppState> {
    Router::new()
        .route("/auth/native-social/exchanges", post(begin_exchange))
        .route(
            "/auth/native-social/exchanges/complete",
            post(complete_exchange),
        )
        .route("/auth/native-social/links", post(begin_link))
        .route("/auth/native-social/links/complete", post(complete_link))
        .route(
            "/auth/native-social/email-link-confirmations/complete",
            post(confirm_link),
        )
        .route("/auth/native-social/keycloak/redeem", post(redeem))
        .layer(DefaultBodyLimit::max(MAX_BODY))
        .layer(Extension(social))
        .layer(axum::middleware::map_response(private_response))
}

fn opaque() -> Result<String, ApiError> {
    let mut bytes = [0u8; 32];
    aws_lc_rs::rand::fill(&mut bytes).map_err(|_| UNAVAILABLE)?;
    Ok(URL_SAFE_NO_PAD.encode(bytes))
}

fn opaque_shape(value: &str) -> bool {
    (43..=128).contains(&value.len())
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
}

fn subject_shape(value: &str) -> bool {
    (1..=255).contains(&value.len())
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b':'))
}

fn digest(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

fn transaction_key(id: &str) -> String {
    format!("native-social:transaction:{}", digest(id))
}

impl Settings {
    async fn save(&self, id: &str, record: &Transaction, seconds: u64) -> Result<(), ApiError> {
        let body = serde_json::to_vec(record).map_err(|_| UNAVAILABLE)?;
        let result: Option<String> = redis::cmd("SET")
            .arg(transaction_key(id))
            .arg(body)
            .arg("EX")
            .arg(seconds)
            .arg("NX")
            .query_async(&mut self.redis.clone())
            .await
            .map_err(|_| UNAVAILABLE)?;
        result.map(|_| ()).ok_or(UNAVAILABLE)
    }

    async fn consume(&self, id: &str) -> Result<Transaction, ApiError> {
        if !opaque_shape(id) {
            return Err(EXPIRED);
        }
        let record: Option<Vec<u8>> = redis::cmd("GETDEL")
            .arg(transaction_key(id))
            .query_async(&mut self.redis.clone())
            .await
            .map_err(|_| UNAVAILABLE)?;
        let record = record.filter(|raw| raw.len() <= MAX_BODY).ok_or(EXPIRED)?;
        serde_json::from_slice(&record).map_err(|_| EXPIRED)
    }

    async fn begin_rate_limit(&self, peer: IpAddr) -> Result<Option<u64>, ApiError> {
        let key = format!("native-social:begin-rate:{}", digest(&peer.to_string()));
        let (count, ttl): (u64, i64) = redis::Script::new(
            "local n = redis.call('INCR', KEYS[1])
             local ttl = redis.call('TTL', KEYS[1])
             if ttl < 0 then
                 ttl = tonumber(ARGV[1])
                 redis.call('EXPIRE', KEYS[1], ttl)
             end
             return { n, ttl }",
        )
        .key(key)
        .arg(BEGIN_WINDOW)
        .invoke_async(&mut self.redis.clone())
        .await
        .map_err(|_| UNAVAILABLE)?;
        Ok((count > BEGIN_LIMIT).then_some(ttl.max(1) as u64))
    }

    fn check_binding(&self, binding: &Binding) -> Result<(), ApiError> {
        if !(1..=128).contains(&binding.client_id.len())
            || !(1..=2048).contains(&binding.redirect_uri.len())
            || !opaque_shape(&binding.code_challenge)
        {
            return Err(BAD_REQUEST);
        }
        if !self
            .allowed_clients
            .get(&binding.client_id)
            .is_some_and(|redirects| redirects.contains(&binding.redirect_uri))
        {
            return Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                "invalid_oidc_client",
            ));
        }
        Ok(())
    }
}

type Peer = Extension<ConnectInfo<SocketAddr>>;

fn single_header(headers: &HeaderMap, name: &str) -> Option<String> {
    let mut values = headers.get_all(name).iter();
    let value = values.next()?.to_str().ok()?;
    if values.next().is_some() {
        return None;
    }
    Some(value.to_owned())
}

fn client_addr(settings: &Settings, peer: Peer, headers: &HeaderMap) -> Result<IpAddr, ApiError> {
    let peer = peer.0.0.ip().to_canonical();
    if settings.trust_proxy
        && peer.is_loopback()
        && let Some(address) = single_header(headers, "x-real-ip")
    {
        return address
            .parse::<IpAddr>()
            .map(|ip| ip.to_canonical())
            .map_err(|_| BAD_REQUEST);
    }
    Ok(peer)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BeginExchange {
    provider: Provider,
    client_id: String,
    redirect_uri: String,
    code_challenge: String,
    code_challenge_method: String,
    state: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Complete {
    transaction_id: String,
    id_token: Option<String>,
    authorization_code: Option<String>,
}

impl Complete {
    fn check(&self) -> Result<(), ApiError> {
        if !(43..=128).contains(&self.transaction_id.len())
            || self
                .id_token
                .as_ref()
                .is_some_and(|value| !(1..=16384).contains(&value.len()))
            || self
                .authorization_code
                .as_ref()
                .is_some_and(|value| !(1..=4096).contains(&value.len()))
        {
            return Err(BAD_REQUEST);
        }
        Ok(())
    }

    async fn verify(
        &self,
        social: &Settings,
        provider: Provider,
        nonce: &str,
    ) -> Result<provider::Identity, ApiError> {
        social
            .providers
            .verify(
                provider,
                self.id_token.as_deref(),
                self.authorization_code.as_deref(),
                nonce,
            )
            .await
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BeginLink {
    provider: Provider,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfirmLink {
    transaction_id: String,
}

#[derive(Serialize)]
struct Started {
    transaction_id: String,
    nonce: String,
    expires_in: u64,
}

async fn begin_exchange(
    Extension(Social(social)): Extension<Social>,
    peer: Peer,
    headers: HeaderMap,
    Json(input): Json<BeginExchange>,
) -> Result<Response, ApiError> {
    if input.code_challenge_method != "S256" || !(1..=2048).contains(&input.state.len()) {
        return Err(BAD_REQUEST);
    }
    let binding = Binding {
        client_id: input.client_id,
        redirect_uri: input.redirect_uri,
        code_challenge: input.code_challenge,
    };
    social.check_binding(&binding)?;
    if let Some(retry_after) = social
        .begin_rate_limit(client_addr(&social, peer, &headers)?)
        .await?
    {
        return Ok((
            StatusCode::TOO_MANY_REQUESTS,
            [(header::RETRY_AFTER, retry_after.to_string())],
            Json(json!({"detail":"native_social_exchange_rate_limited"})),
        )
            .into_response());
    }
    let transaction_id = opaque()?;
    let nonce = opaque()?;
    social
        .save(
            &transaction_id,
            &Transaction::LoginBegin {
                provider: input.provider,
                nonce: nonce.clone(),
                binding,
                state: input.state,
            },
            social.exchange_ttl,
        )
        .await?;
    Ok(Json(Started {
        transaction_id,
        nonce,
        expires_in: social.exchange_ttl,
    })
    .into_response())
}

async fn complete_exchange(
    State(state): State<AppState>,
    Extension(Social(social)): Extension<Social>,
    Json(input): Json<Complete>,
) -> Result<Response, ApiError> {
    input.check()?;
    let Transaction::LoginBegin {
        provider,
        nonce,
        binding,
        state: client_state,
    } = social.consume(&input.transaction_id).await?
    else {
        return Err(EXPIRED);
    };
    social.check_binding(&binding)?;
    let identity = input.verify(&social, provider, &nonce).await?;
    let encoded = URL_SAFE_NO_PAD.encode(identity.subject.as_bytes());
    if let Some((local_user_id, keycloak_subject)) =
        link::email_candidate(&state.db, social.scope, &identity).await?
    {
        let id = opaque()?;
        social
            .save(
                &id,
                &Transaction::EmailLinkConfirmation {
                    provider,
                    provider_subject_b64: encoded,
                    local_user_id,
                    keycloak_subject,
                },
                social.link_ttl,
            )
            .await?;
        return Ok((StatusCode::CONFLICT, Json(json!({"detail":{
            "code":"account_link_required", "link_transaction_id":id, "expires_in":social.link_ttl
        }}))).into_response());
    }
    let mut authorization_url = social.authorization_endpoint.clone();
    authorization_url.query_pairs_mut().extend_pairs([
        ("response_type", "code"),
        ("client_id", &binding.client_id),
        ("redirect_uri", &binding.redirect_uri),
        ("code_challenge", &binding.code_challenge),
        ("code_challenge_method", "S256"),
        ("state", &client_state),
        ("native_social_exchange", &input.transaction_id),
    ]);
    if authorization_url.as_str().len() > 4096 {
        return Err(BAD_REQUEST);
    }
    social
        .save(
            &input.transaction_id,
            &Transaction::LoginRedeem {
                provider,
                provider_subject_b64: encoded,
                binding,
            },
            social.exchange_ttl,
        )
        .await?;
    Ok(Json(
        json!({"authorization_url": authorization_url.as_str(), "expires_in": social.exchange_ttl}),
    )
    .into_response())
}

struct LocalIdentity {
    subject: String,
    session_id: Option<String>,
    user_id: i32,
}

async fn fresh_identity(
    state: &AppState,
    social: &Settings,
    headers: &HeaderMap,
) -> Result<LocalIdentity, ApiError> {
    let access = state.auth.verify_access_token(bearer(headers)?).await?;
    let now = chrono::Utc::now().timestamp_millis() as f64 / 1000.;
    if access.issuer != social.issuer
        || !subject_shape(&access.subject)
        || access.session_id_invalid
        || access.auth_time.is_none_or(|time| {
            !time.is_finite() || time > now + 60. || now - time > social.max_auth_age as f64
        })
    {
        return Err(REAUTHENTICATE);
    }
    let session_id = access.session_id.clone();
    let actor = state
        .auth
        .authorize_verified(headers, Grant::AccountRead, access)
        .await?;
    let account = social.scope.account(&state.db, &actor.subject).await?;
    Ok(LocalIdentity {
        subject: actor.subject,
        session_id,
        user_id: account.id,
    })
}

async fn begin_link(
    State(state): State<AppState>,
    Extension(Social(social)): Extension<Social>,
    headers: HeaderMap,
    Json(input): Json<BeginLink>,
) -> Result<Json<Started>, ApiError> {
    let identity = fresh_identity(&state, &social, &headers).await?;
    let transaction_id = opaque()?;
    let nonce = opaque()?;
    social
        .save(
            &transaction_id,
            &Transaction::LinkBegin {
                provider: input.provider,
                nonce: nonce.clone(),
                keycloak_subject: identity.subject,
                keycloak_session_id: identity.session_id,
            },
            social.link_ttl,
        )
        .await?;
    Ok(Json(Started {
        transaction_id,
        nonce,
        expires_in: social.link_ttl,
    }))
}

async fn complete_link(
    State(state): State<AppState>,
    Extension(Social(social)): Extension<Social>,
    headers: HeaderMap,
    Json(input): Json<Complete>,
) -> Result<Response, ApiError> {
    input.check()?;
    let current = fresh_identity(&state, &social, &headers).await?;
    let Transaction::LinkBegin {
        provider,
        nonce,
        keycloak_subject,
        keycloak_session_id,
    } = social.consume(&input.transaction_id).await?
    else {
        return Err(LINK_EXPIRED);
    };
    if !equal(&keycloak_subject, &current.subject)
        || keycloak_session_id.as_ref().is_some_and(|bound| {
            current
                .session_id
                .as_ref()
                .is_none_or(|session| !equal(bound, session))
        })
    {
        return Err(LINK_EXPIRED);
    }
    let identity = input.verify(&social, provider, &nonce).await?;
    link_identity(state, social, current, identity).await
}

async fn confirm_link(
    State(state): State<AppState>,
    Extension(Social(social)): Extension<Social>,
    headers: HeaderMap,
    Json(input): Json<ConfirmLink>,
) -> Result<Response, ApiError> {
    if !(43..=128).contains(&input.transaction_id.len()) {
        return Err(BAD_REQUEST);
    }
    let current = fresh_identity(&state, &social, &headers).await?;
    let Transaction::EmailLinkConfirmation {
        provider,
        provider_subject_b64,
        local_user_id,
        keycloak_subject,
    } = social.consume(&input.transaction_id).await?
    else {
        return Err(LINK_EXPIRED);
    };
    if local_user_id != current.user_id || !equal(&keycloak_subject, &current.subject) {
        return Err(LINK_EXPIRED);
    }
    let identity = provider::Identity {
        provider,
        subject: decode_subject(&provider_subject_b64)?,
        email: None,
    };
    link_identity(state, social, current, identity).await
}

async fn link_identity(
    state: AppState,
    social: Arc<Settings>,
    current: LocalIdentity,
    identity: provider::Identity,
) -> Result<Response, ApiError> {
    let provider = identity.provider;
    // Complete an accepted remote operation even if the caller disconnects.
    let status = tokio::spawn(async move {
        link::attach(&state.db, social.scope, &social.keycloak, current, identity).await
    })
    .await
    .map_err(|_| UNAVAILABLE)??;
    Ok(Json(json!({"provider":provider, "status":status})).into_response())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Redeem {
    transaction_id: String,
    client_id: String,
    redirect_uri: String,
    code_challenge: String,
}

fn broker_local(peer: Peer, headers: &HeaderMap) -> bool {
    if !peer.0.0.ip().to_canonical().is_loopback() {
        return false;
    }
    single_header(headers, "host")
        .and_then(|value| value.parse::<Authority>().ok())
        .is_some_and(|authority| {
            let host = authority.host().trim_matches(['[', ']']);
            host.eq_ignore_ascii_case("localhost")
                || host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
        })
}

fn verify_signature(key: &Hmac<Sha256>, headers: &HeaderMap, body: &Bytes) -> Result<(), ApiError> {
    let stamp = single_header(headers, "x-visionnaire-timestamp").ok_or(INVALID)?;
    let timestamp = stamp.parse::<i64>().map_err(|_| INVALID)?;
    if !stamp.bytes().all(|value| value.is_ascii_digit())
        || chrono::Utc::now().timestamp().abs_diff(timestamp) > BROKER_TIME_WINDOW
    {
        return Err(INVALID);
    }
    let signature = single_header(headers, "x-visionnaire-signature").ok_or(INVALID)?;
    let supplied = URL_SAFE_NO_PAD.decode(signature).map_err(|_| INVALID)?;
    let mut mac = key.clone();
    mac.update(stamp.as_bytes());
    mac.update(b".");
    mac.update(body);
    mac.verify_slice(&supplied).map_err(|_| INVALID)
}

async fn redeem(
    Extension(Social(social)): Extension<Social>,
    peer: Peer,
    request: Request,
) -> Result<Response, ApiError> {
    if !broker_local(peer, request.headers()) {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "native_social_exchange_forbidden",
        ));
    }
    let (parts, body) = request.into_parts();
    let bytes = to_bytes(body, MAX_BODY).await.map_err(|_| INVALID)?;
    verify_signature(&social.broker_key, &parts.headers, &bytes)?;
    let input: Redeem = serde_json::from_slice(&bytes).map_err(|_| INVALID)?;
    let Transaction::LoginRedeem {
        provider,
        provider_subject_b64,
        binding,
    } = social.consume(&input.transaction_id).await?
    else {
        return Err(INVALID);
    };
    if !equal(&binding.client_id, &input.client_id)
        || !equal(&binding.redirect_uri, &input.redirect_uri)
        || !equal(&binding.code_challenge, &input.code_challenge)
    {
        return Err(INVALID);
    }
    decode_subject(&provider_subject_b64)?;
    Ok(
        Json(json!({"provider":provider, "provider_subject_b64":provider_subject_b64}))
            .into_response(),
    )
}

fn equal(left: &str, right: &str) -> bool {
    verify_slices_are_equal(left.as_bytes(), right.as_bytes()).is_ok()
}

fn decode_subject(encoded: &str) -> Result<String, ApiError> {
    let bytes = URL_SAFE_NO_PAD.decode(encoded).map_err(|_| INVALID)?;
    if bytes.is_empty() || bytes.len() > 512 {
        return Err(INVALID);
    }
    String::from_utf8(bytes).map_err(|_| INVALID)
}
