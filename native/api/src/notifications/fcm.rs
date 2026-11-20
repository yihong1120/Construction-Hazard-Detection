use std::{
    collections::BTreeMap,
    io::Read,
    sync::Mutex as DiagnosticLock,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use chrono::{DateTime, Utc};
use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
use reqwest::{Response, StatusCode, header};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Mutex;

const TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
const SCOPE: &str = "https://www.googleapis.com/auth/firebase.messaging";
const GRANT: &str = "urn:ietf:params:oauth:grant-type:jwt-bearer";
const MAX_RESPONSE_BYTES: usize = 64 * 1024;
const DIAGNOSTIC_INTERVAL: Duration = Duration::from_secs(60);
const CONFIGURATION_RETRY_SECONDS: u64 = 900;
const TRANSIENT_RETRY_SECONDS: u64 = 10;
const MAX_DEFERRED_REGISTRATIONS: usize = 4096;
const MAX_PROVIDER_RETRY_SECONDS: u64 = 86_400;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Config {
    pub service_account_path_env: String,
    pub project_id: Option<String>,
}

#[derive(Deserialize)]
struct ServiceAccount {
    #[serde(rename = "type")]
    kind: String,
    project_id: String,
    client_email: String,
    private_key: String,
    private_key_id: Option<String>,
}

#[derive(Serialize)]
struct Assertion<'a> {
    iss: &'a str,
    aud: &'a str,
    scope: &'static str,
    iat: i64,
    exp: i64,
}

#[derive(Deserialize)]
struct OAuthResponse {
    access_token: String,
    expires_in: u64,
    token_type: String,
}

struct Token {
    value: String,
    valid_until: Instant,
}

struct DeferredRegistration {
    revision: DateTime<Utc>,
    until: Instant,
    failure: Failure,
}

pub(super) struct Sender {
    http: reqwest::Client,
    signer: EncodingKey,
    header: Header,
    email: String,
    oauth_url: String,
    send_url: String,
    token: Mutex<Option<Token>>,
    diagnostic: DiagnosticLock<BTreeMap<(&'static str, &'static str), Instant>>,
    deferred: DiagnosticLock<BTreeMap<String, DeferredRegistration>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum FailureKind {
    Registration,
    Payload,
    Configuration,
    Retryable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Failure {
    pub(super) kind: FailureKind,
    pub(super) reason: &'static str,
    pub(super) retry_after_seconds: Option<u64>,
    phase: &'static str,
    status: Option<StatusCode>,
}

pub(super) type SendResult = std::result::Result<(), Failure>;

impl Failure {
    fn new(kind: FailureKind, reason: &'static str, phase: &'static str) -> Self {
        let retry_after_seconds = match kind {
            FailureKind::Configuration => Some(CONFIGURATION_RETRY_SECONDS),
            FailureKind::Retryable => Some(TRANSIENT_RETRY_SECONDS),
            _ => None,
        };
        Self {
            kind,
            reason,
            retry_after_seconds,
            phase,
            status: None,
        }
    }

    pub(super) fn device_storage() -> Self {
        Self::new(
            FailureKind::Configuration,
            "device_token_decryption_failed",
            "device_store",
        )
    }

    fn transport(error: &reqwest::Error, phase: &'static str) -> Self {
        let reason = if error.is_timeout() {
            "provider_timeout"
        } else {
            "provider_transport_error"
        };
        Self::new(FailureKind::Retryable, reason, phase)
    }

    fn with_response(mut self, status: StatusCode, retry_after: Option<u64>) -> Self {
        self.status = Some(status);
        self.retry_after_seconds = self
            .retry_after_seconds
            .map(|delay| delay.max(retry_after.unwrap_or(0)));
        self
    }
}

impl Sender {
    pub fn new(config: Config) -> Result<Self> {
        let path = std::env::var_os(&config.service_account_path_env)
            .with_context(|| format!("Set {}", config.service_account_path_env))?;
        let file =
            std::fs::File::open(path).context("Open Firebase service-account credentials")?;
        let mut content = Vec::new();
        file.take(65537)
            .read_to_end(&mut content)
            .context("Read Firebase service-account credentials")?;
        ensure!(
            content.len() <= 65536,
            "Firebase service-account file is too large"
        );
        let service: ServiceAccount = serde_json::from_slice(&content)
            .map_err(|_| anyhow::anyhow!("Invalid Firebase credentials JSON"))?;
        let project = config.project_id.unwrap_or(service.project_id);
        ensure!(
            service.kind == "service_account"
                && service.client_email.ends_with(".gserviceaccount.com")
                && (1..=128).contains(&project.len())
                && project
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-'),
            "Invalid Firebase service-account configuration"
        );
        let signer = EncodingKey::from_rsa_pem(service.private_key.as_bytes())
            .map_err(|_| anyhow::anyhow!("Invalid Firebase private key"))?;
        let mut header = Header::new(Algorithm::RS256);
        header.kid = service.private_key_id;
        Ok(Self {
            http: reqwest::Client::builder()
                .https_only(true)
                .redirect(reqwest::redirect::Policy::none())
                .connect_timeout(Duration::from_secs(3))
                .timeout(Duration::from_secs(12))
                .build()?,
            signer,
            header,
            email: service.client_email,
            oauth_url: TOKEN_URL.to_owned(),
            send_url: format!("https://fcm.googleapis.com/v1/projects/{project}/messages:send"),
            token: Mutex::new(None),
            diagnostic: DiagnosticLock::default(),
            deferred: DiagnosticLock::default(),
        })
    }

    async fn access_token(&self) -> std::result::Result<String, Failure> {
        let mut cached = self.token.lock().await;
        if let Some(token) = cached
            .as_ref()
            .filter(|token| token.valid_until > Instant::now())
        {
            return Ok(token.value.clone());
        }
        let now = Utc::now().timestamp();
        let assertion = encode(
            &self.header,
            &Assertion {
                iss: &self.email,
                aud: &self.oauth_url,
                scope: SCOPE,
                iat: now,
                exp: now + 3600,
            },
            &self.signer,
        )
        .map_err(|_| {
            Failure::new(
                FailureKind::Configuration,
                "firebase_assertion_invalid",
                "oauth",
            )
        })?;
        let mut response = self
            .http
            .post(&self.oauth_url)
            .form(&[("grant_type", GRANT), ("assertion", &assertion)])
            .send()
            .await
            .map_err(|error| Failure::transport(&error, "oauth"))?;
        let status = response.status();
        let retry_after = retry_after(&response);
        if !status.is_success() {
            let failure = if retryable_status(status) {
                retryable(status, "oauth")
            } else {
                let body = limited_json::<Value>(&mut response).await;
                let reason = match body.as_ref().and_then(|body| body["error"].as_str()) {
                    Some("invalid_grant") => "firebase_oauth_invalid_grant",
                    Some("invalid_client") => "firebase_oauth_invalid_client",
                    Some("unauthorized_client") => "firebase_oauth_unauthorized_client",
                    _ => "firebase_oauth_rejected",
                };
                Failure::new(FailureKind::Configuration, reason, "oauth")
            };
            return Err(failure.with_response(status, retry_after));
        }
        let token = limited_json::<OAuthResponse>(&mut response)
            .await
            .filter(|token| {
                token.token_type.eq_ignore_ascii_case("bearer")
                    && !token.access_token.is_empty()
                    && token.expires_in > 0
            })
            .ok_or_else(|| {
                Failure::new(
                    FailureKind::Retryable,
                    "firebase_oauth_unconfirmed",
                    "oauth",
                )
                .with_response(status, retry_after)
            })?;
        let value = token.access_token;
        let lifetime = token.expires_in.min(3600).saturating_sub(30);
        *cached = Some(Token {
            value: value.clone(),
            valid_until: Instant::now() + Duration::from_secs(lifetime),
        });
        Ok(value)
    }

    pub(super) async fn send(&self, message: &Value, platform: &str) -> SendResult {
        let result = self.send_provider(message).await;
        if let Err(failure) = result {
            self.observe_failure(platform, failure);
        }
        result
    }

    pub(super) fn deferred_failure(&self, hash: &str, revision: DateTime<Utc>) -> Option<Failure> {
        let mut entries = self.deferred.lock().ok()?;
        let entry = entries.get(hash)?;
        if entry.revision > revision {
            return None;
        }
        let remaining = entry.until.saturating_duration_since(Instant::now());
        if entry.revision != revision || remaining.is_zero() {
            entries.remove(hash);
            return None;
        }
        let retry_after_seconds = Some(remaining.as_secs().saturating_add(1));
        Some(Failure {
            retry_after_seconds,
            ..entry.failure
        })
    }

    pub(super) fn defer_failure(&self, hash: &str, revision: DateTime<Utc>, failure: Failure) {
        if failure.kind != FailureKind::Configuration
            && !matches!(failure.status, Some(StatusCode::TOO_MANY_REQUESTS))
        {
            return;
        }
        let Some(seconds) = failure.retry_after_seconds else {
            return;
        };
        let Ok(mut entries) = self.deferred.lock() else {
            return;
        };
        if entries
            .get(hash)
            .is_some_and(|entry| entry.revision > revision)
        {
            return;
        }
        let now = Instant::now();
        if entries.len() >= MAX_DEFERRED_REGISTRATIONS {
            entries.retain(|_, entry| entry.until > now);
        }
        if entries.len() < MAX_DEFERRED_REGISTRATIONS || entries.contains_key(hash) {
            entries.insert(
                hash.to_owned(),
                DeferredRegistration {
                    revision,
                    until: now + Duration::from_secs(seconds),
                    failure,
                },
            );
        }
    }

    pub(super) fn observe_failure(&self, platform: &str, failure: Failure) {
        let platform = match platform {
            "android" => "android",
            "ios" => "ios",
            "web" => "web",
            _ => "unknown",
        };
        let now = Instant::now();
        let Ok(mut last) = self.diagnostic.lock() else {
            return;
        };
        let key = (platform, failure.reason);
        if last
            .get(&key)
            .is_some_and(|previous| now.duration_since(*previous) < DIAGNOSTIC_INTERVAL)
        {
            return;
        }
        last.insert(key, now);
        drop(last);
        tracing::warn!(
            phase = failure.phase,
            platform,
            class = ?failure.kind,
            reason = failure.reason,
            http_status = failure.status.map(|status| status.as_u16()),
            retry_after_seconds = failure.retry_after_seconds,
            "Firebase delivery not acknowledged"
        );
    }

    async fn send_provider(&self, message: &Value) -> SendResult {
        for attempt in 0..2 {
            let token = self.access_token().await?;
            let mut response = self
                .http
                .post(&self.send_url)
                .bearer_auth(&token)
                .json(message)
                .send()
                .await
                .map_err(|error| Failure::transport(&error, "send"))?;
            let status = response.status();
            let retry_after = retry_after(&response);
            let body = limited_json::<Value>(&mut response)
                .await
                .unwrap_or(Value::Null);
            if status.is_success() {
                return if acknowledged(&body) {
                    Ok(())
                } else {
                    Err(
                        Failure::new(FailureKind::Retryable, "firebase_send_unconfirmed", "send")
                            .with_response(status, retry_after),
                    )
                };
            }
            let failure = classify_error(status, &body).with_response(status, retry_after);
            if failure.reason == "firebase_authentication_failed" && attempt == 0 {
                let mut cached = self.token.lock().await;
                if cached.as_ref().is_some_and(|cached| cached.value == token) {
                    *cached = None;
                }
                continue;
            }
            return Err(failure);
        }
        unreachable!("Firebase auth is retried at most once")
    }
}

fn acknowledged(body: &Value) -> bool {
    body["name"].as_str().is_some_and(|name| {
        name.len() <= 512
            && name.bytes().all(|byte| byte.is_ascii_graphic())
            && name
                .strip_prefix("projects/")
                .and_then(|name| name.split_once("/messages/"))
                .is_some_and(|(project, message)| !project.is_empty() && !message.is_empty())
    })
}

async fn limited_json<T: serde::de::DeserializeOwned>(response: &mut Response) -> Option<T> {
    let mut data = Vec::new();
    while let Some(chunk) = response.chunk().await.ok()? {
        if chunk.len() > MAX_RESPONSE_BYTES - data.len() {
            return None;
        }
        data.extend_from_slice(&chunk);
    }
    serde_json::from_slice(&data).ok()
}

fn retry_after(response: &Response) -> Option<u64> {
    let raw = response.headers().get(header::RETRY_AFTER)?.to_str().ok()?;
    let delay = raw.parse().ok().or_else(|| {
        let until = chrono::DateTime::parse_from_rfc2822(raw).ok()?;
        u64::try_from(
            (until.with_timezone(&Utc) - Utc::now())
                .num_seconds()
                .saturating_add(1),
        )
        .ok()
    })?;
    Some(delay.min(MAX_PROVIDER_RETRY_SECONDS))
}

fn retryable_status(status: StatusCode) -> bool {
    status.is_server_error()
        || matches!(
            status,
            StatusCode::REQUEST_TIMEOUT | StatusCode::TOO_EARLY | StatusCode::TOO_MANY_REQUESTS
        )
}

fn retryable(status: StatusCode, phase: &'static str) -> Failure {
    let mut failure = match status {
        StatusCode::TOO_MANY_REQUESTS => {
            Failure::new(FailureKind::Retryable, "firebase_quota_exceeded", phase)
        }
        StatusCode::SERVICE_UNAVAILABLE => {
            Failure::new(FailureKind::Retryable, "firebase_unavailable", phase)
        }
        StatusCode::REQUEST_TIMEOUT => {
            Failure::new(FailureKind::Retryable, "firebase_request_timeout", phase)
        }
        _ => Failure::new(FailureKind::Retryable, "firebase_temporary_error", phase),
    };
    if status == StatusCode::TOO_MANY_REQUESTS {
        failure.retry_after_seconds = Some(60);
    }
    failure
}

fn classify_error(status: StatusCode, body: &Value) -> Failure {
    use FailureKind::{Configuration, Payload, Registration, Retryable};
    if retryable_status(status) {
        return retryable(status, "send");
    }
    let error = &body["error"];
    let details = error["details"].as_array().into_iter().flatten();
    let fcm = details
        .clone()
        .find(|detail| detail["@type"] == "type.googleapis.com/google.firebase.fcm.v1.FcmError")
        .and_then(|detail| detail["errorCode"].as_str());
    let token_field = details
        .clone()
        .filter(|detail| detail["@type"] == "type.googleapis.com/google.rpc.BadRequest")
        .filter_map(|detail| detail["fieldViolations"].as_array())
        .flatten()
        .collect::<Vec<_>>();
    let exclusively_token = !token_field.is_empty()
        && token_field
            .iter()
            .all(|violation| violation["field"] == "message.token");
    let invalid_registration = exclusively_token
        || (token_field.is_empty()
            && fcm == Some("INVALID_ARGUMENT")
            && error["message"] == "The registration token is not a valid FCM registration token");
    let invalid_apns_credential = error["message"].as_str().is_some_and(|message| {
        message
            .trim()
            .trim_end_matches('.')
            .eq_ignore_ascii_case("Invalid APNS credential")
    });
    let (kind, reason) = match (status, fcm) {
        (StatusCode::NOT_FOUND, Some("UNREGISTERED")) => (Registration, "firebase_unregistered"),
        (StatusCode::BAD_REQUEST, _) if invalid_registration => {
            (Registration, "firebase_invalid_registration")
        }
        (StatusCode::BAD_REQUEST, Some("INVALID_ARGUMENT")) => {
            (Payload, "firebase_invalid_payload")
        }
        (StatusCode::BAD_REQUEST, _)
            if !token_field.is_empty() || error["status"] == "INVALID_ARGUMENT" =>
        {
            (Payload, "firebase_invalid_payload")
        }
        (StatusCode::PAYLOAD_TOO_LARGE, _) => (Payload, "firebase_payload_too_large"),
        (StatusCode::FORBIDDEN, Some("SENDER_ID_MISMATCH")) => {
            (Configuration, "firebase_sender_id_mismatch")
        }
        (StatusCode::UNAUTHORIZED, _)
            if fcm == Some("THIRD_PARTY_AUTH_ERROR") || invalid_apns_credential =>
        {
            (Configuration, "firebase_third_party_auth")
        }
        (StatusCode::FORBIDDEN, Some("THIRD_PARTY_PERMISSION_DENIED")) => {
            (Configuration, "firebase_third_party_permission")
        }
        (StatusCode::UNAUTHORIZED, _) => (Configuration, "firebase_authentication_failed"),
        (StatusCode::FORBIDDEN, _) => (Configuration, "firebase_permission_denied"),
        (StatusCode::NOT_FOUND, _) => (Configuration, "firebase_project_or_route_not_found"),
        _ => (Retryable, "firebase_unclassified_response"),
    };
    Failure::new(kind, reason, "send")
}

pub(super) fn payload(
    token: &str,
    title: &str,
    body: &str,
    data: &Value,
    image: Option<&str>,
) -> Value {
    let mut message = json!({
        "token": token,
        "notification": {"title": title, "body": body},
        "data": data,
        "android": {"priority": "high", "notification": {
            "channel_id": "high_importance_channel", "sound": "default", "default_vibrate_timings": true
        }},
        "apns": {"headers": {"apns-priority": "10"}, "payload": {"aps": {"sound": "default", "badge": 1}}},
        "webpush": {"headers": {"Urgency": "high", "TTL": "3600"}}
    });
    if let Some(image) = image {
        message["notification"]["image"] = image.into();
    }
    json!({"message": message})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_only_explicit_fcm_device_errors_as_invalid() {
        use FailureKind::*;
        let typed = |code, message| {
            json!({"error": {"message": message, "details": [{
                "@type": "type.googleapis.com/google.firebase.fcm.v1.FcmError", "errorCode": code
            }]}})
        };
        let field = |field| {
            json!({"error":{"status":"INVALID_ARGUMENT","details":[{
                "@type":"type.googleapis.com/google.rpc.BadRequest","fieldViolations":[{"field":field}]
            }]}})
        };
        for (status, error, kind, reason) in [
            (
                404,
                typed("UNREGISTERED", "private diagnostic"),
                Registration,
                "firebase_unregistered",
            ),
            (
                404,
                json!({"error":{"status":"NOT_FOUND"}}),
                Configuration,
                "firebase_project_or_route_not_found",
            ),
            (
                503,
                typed("UNREGISTERED", ""),
                Retryable,
                "firebase_unavailable",
            ),
            (
                400,
                typed(
                    "INVALID_ARGUMENT",
                    "The registration token is not a valid FCM registration token",
                ),
                Registration,
                "firebase_invalid_registration",
            ),
            (
                400,
                field("message.token"),
                Registration,
                "firebase_invalid_registration",
            ),
            (
                400,
                field("message.data[0].value"),
                Payload,
                "firebase_invalid_payload",
            ),
            (
                400,
                typed("INVALID_ARGUMENT", "unknown argument"),
                Payload,
                "firebase_invalid_payload",
            ),
            (
                401,
                typed("THIRD_PARTY_AUTH_ERROR", "credential details"),
                Configuration,
                "firebase_third_party_auth",
            ),
            (
                401,
                json!({"error":{"status":"UNAUTHENTICATED","message":"Invalid APNS credential."}}),
                Configuration,
                "firebase_third_party_auth",
            ),
            (
                403,
                typed("THIRD_PARTY_PERMISSION_DENIED", "credential details"),
                Configuration,
                "firebase_third_party_permission",
            ),
            (
                403,
                typed("SENDER_ID_MISMATCH", "wrong sender"),
                Configuration,
                "firebase_sender_id_mismatch",
            ),
            (
                401,
                json!({}),
                Configuration,
                "firebase_authentication_failed",
            ),
            (429, json!({}), Retryable, "firebase_quota_exceeded"),
        ] {
            let observed = classify_error(StatusCode::from_u16(status).unwrap(), &error);
            assert_eq!((observed.kind, observed.reason), (kind, reason));
        }
        let message = payload(
            "fixture",
            "title",
            "body",
            &json!({"violation_id":"14"}),
            None,
        );
        assert_eq!(
            message["message"]["android"]["notification"]["channel_id"],
            "high_importance_channel"
        );
        assert_eq!(message["message"]["data"]["violation_id"], "14");
        assert!(!acknowledged(&json!({"name":"not a provider receipt"})));
        assert!(acknowledged(
            &json!({"name":"projects/test/messages/1:123%abcdef"})
        ));
    }

    #[tokio::test]
    async fn exchanges_signed_assertions_and_refreshes_after_fcm_unauthorised() {
        use aws_lc_rs::{
            encoding::{AsDer, Pkcs8V1Der},
            rsa::{KeyPair, KeySize},
            signature::KeyPair as _,
        };
        use axum::{
            Form, Json, Router,
            extract::State,
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
            routing::post,
        };
        use jsonwebtoken::{DecodingKey, Validation, decode};
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };

        #[derive(Clone)]
        struct Mock {
            audience: String,
            verifier: DecodingKey,
            exchanges: Arc<AtomicUsize>,
        }

        async fn exchange(
            State(mock): State<Mock>,
            Form(form): Form<std::collections::HashMap<String, String>>,
        ) -> Json<Value> {
            assert_eq!(form["grant_type"], GRANT);
            let mut validation = Validation::new(Algorithm::RS256);
            validation.set_audience(&[&mock.audience]);
            let jwt = decode::<Value>(&form["assertion"], &mock.verifier, &validation).unwrap();
            assert_eq!(jwt.claims["scope"], SCOPE);
            assert_eq!(jwt.claims["iss"], "fixture@demo.iam.gserviceaccount.com");
            let attempt = mock.exchanges.fetch_add(1, Ordering::SeqCst) + 1;
            Json(
                json!({"access_token": format!("fixture-{attempt}"), "token_type": "Bearer", "expires_in": 3600}),
            )
        }

        async fn push(headers: HeaderMap, Json(body): Json<Value>) -> impl IntoResponse {
            let token = &body["message"]["token"];
            if token == "third-party" {
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(json!({"error":{"details":[{
                        "@type":"type.googleapis.com/google.firebase.fcm.v1.FcmError",
                        "errorCode":"THIRD_PARTY_AUTH_ERROR"
                    }]}})),
                )
                    .into_response();
            }
            if headers["authorization"] == "Bearer fixture-1" {
                return (StatusCode::UNAUTHORIZED, Json(json!({}))).into_response();
            }
            assert_eq!(headers["authorization"], "Bearer fixture-2");
            if token == "retired" {
                return (
                    StatusCode::NOT_FOUND,
                    Json(json!({"error": {"details": [{
                        "@type": "type.googleapis.com/google.firebase.fcm.v1.FcmError", "errorCode": "UNREGISTERED"
                    }]}})),
                ).into_response();
            }
            if token == "throttled" {
                return (
                    StatusCode::TOO_MANY_REQUESTS,
                    [(header::RETRY_AFTER, "120")],
                    Json(json!({})),
                )
                    .into_response();
            }
            if token == "unconfirmed" {
                return (StatusCode::OK, Json(json!({}))).into_response();
            }
            (
                StatusCode::OK,
                Json(json!({"name": "projects/demo/messages/fixture"})),
            )
                .into_response()
        }

        let key = KeyPair::generate(KeySize::Rsa2048).unwrap();
        let private: Pkcs8V1Der<'static> = key.as_der().unwrap();
        use base64::{Engine, engine::general_purpose::STANDARD};
        let pem = format!(
            "-----BEGIN PRIVATE KEY-----\n{}\n-----END PRIVATE KEY-----",
            STANDARD.encode(private.as_ref())
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let exchanges = Arc::new(AtomicUsize::new(0));
        let mock = Mock {
            audience: format!("{base}/token"),
            verifier: DecodingKey::from_rsa_der(key.public_key().as_ref()),
            exchanges: exchanges.clone(),
        };
        let app = Router::new()
            .route("/token", post(exchange))
            .route("/send", post(push))
            .with_state(mock);
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let sender = Sender {
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(3))
                .build()
                .unwrap(),
            signer: EncodingKey::from_rsa_pem(pem.as_bytes()).unwrap(),
            header: Header::new(Algorithm::RS256),
            email: "fixture@demo.iam.gserviceaccount.com".into(),
            oauth_url: format!("{base}/token"),
            send_url: format!("{base}/send"),
            token: Mutex::new(None),
            diagnostic: DiagnosticLock::default(),
            deferred: DiagnosticLock::default(),
        };
        let message = |device| payload(device, "title", "body", &json!({}), None);
        assert_eq!(sender.access_token().await.unwrap(), "fixture-1");
        let third_party = sender
            .send(&message("third-party"), "ios")
            .await
            .unwrap_err();
        assert_eq!(third_party.reason, "firebase_third_party_auth");
        assert_eq!(
            third_party.retry_after_seconds,
            Some(CONFIGURATION_RETRY_SECONDS)
        );
        assert_eq!(exchanges.load(Ordering::SeqCst), 1);
        let registration = Utc::now();
        sender.defer_failure("fixture-hash", registration, third_party);
        let defer = sender
            .deferred_failure("fixture-hash", registration)
            .unwrap();
        assert_eq!(defer.reason, "firebase_third_party_auth");
        assert!(defer.retry_after_seconds.unwrap() <= CONFIGURATION_RETRY_SECONDS);
        assert!(
            sender
                .deferred_failure("fixture-hash", registration - chrono::Duration::seconds(1))
                .is_none()
        );
        assert!(
            sender
                .deferred_failure("fixture-hash", registration)
                .is_some()
        );
        assert!(
            sender
                .deferred_failure("fixture-hash", registration + chrono::Duration::seconds(1))
                .is_none()
        );
        assert!(sender.send(&message("active"), "android").await.is_ok());
        assert_eq!(
            sender
                .send(&message("retired"), "ios")
                .await
                .unwrap_err()
                .kind,
            FailureKind::Registration
        );
        assert_eq!(
            sender
                .send(&message("throttled"), "web")
                .await
                .unwrap_err()
                .retry_after_seconds,
            Some(120)
        );
        assert_eq!(
            sender
                .send(&message("unconfirmed"), "web")
                .await
                .unwrap_err()
                .reason,
            "firebase_send_unconfirmed"
        );
        assert_eq!(exchanges.load(Ordering::SeqCst), 2);
        task.abort();
    }
}
