use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use percent_encoding::percent_decode_str;
use reqwest::{Client, Method, Response, StatusCode, header::LOCATION};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::Mutex;
use url::Url;

use crate::ApiError;

const UNAVAILABLE: ApiError = ApiError::new(
    StatusCode::SERVICE_UNAVAILABLE,
    "keycloak_identity_service_unavailable",
);
const USER_NOT_FOUND: ApiError = ApiError::new(StatusCode::NOT_FOUND, "keycloak_user_not_found");

#[derive(Clone)]
pub(crate) struct Keycloak(Arc<Settings>);

struct Settings {
    client: Client,
    base: Url,
    realm: String,
    client_id: String,
    client_secret: String,
    token: Mutex<Option<Token>>,
}

struct Token {
    access_token: String,
    expires: Instant,
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    expires_in: u64,
}

impl Keycloak {
    pub(crate) fn from_env() -> Result<Option<Self>> {
        let client_id = env_value("KEYCLOAK_USER_LINKER_CLIENT_ID");
        let client_secret = env_value("KEYCLOAK_USER_LINKER_CLIENT_SECRET");
        let Some(client_secret) = client_secret else {
            return Ok(None);
        };
        let client_id = client_id.unwrap_or_else(|| "visionnaire-user-linker".to_owned());
        let realm = env_value("KEYCLOAK_REALM").unwrap_or_else(|| "visionnaire".to_owned());
        ensure!(valid_segment(&realm), "Invalid Keycloak realm");
        let base = env_value("KEYCLOAK_ADMIN_BASE_URL")
            .or_else(|| {
                env_value("OIDC_ISSUER_URL")
                    .and_then(|issuer| Some(issuer.split_once("/realms/")?.0.to_owned()))
            })
            .context("Set KEYCLOAK_ADMIN_BASE_URL or OIDC_ISSUER_URL")?;
        let base = Url::parse(&base).context("Invalid Keycloak Admin base URL")?;
        ensure!(
            matches!(base.scheme(), "http" | "https")
                && base.has_host()
                && base.username().is_empty()
                && base.password().is_none()
                && base.query().is_none()
                && base.fragment().is_none(),
            "Invalid Keycloak Admin base URL"
        );
        let client = Client::builder()
            .timeout(Duration::from_secs(5))
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .build()?;
        Ok(Some(Self(Arc::new(Settings {
            client,
            base,
            realm,
            client_id,
            client_secret,
            token: Mutex::new(None),
        }))))
    }

    fn endpoint(&self, segments: &[&str]) -> Url {
        let mut url = self.0.base.clone();
        url.path_segments_mut()
            .expect("hierarchical Keycloak endpoint")
            .pop_if_empty()
            .extend(segments);
        url
    }

    async fn token(&self) -> Result<String, ApiError> {
        let mut cache = self.0.token.lock().await;
        if let Some(token) = &*cache
            && token.expires > Instant::now()
        {
            return Ok(token.access_token.clone());
        }
        let endpoint = self.endpoint(&[
            "realms",
            &self.0.realm,
            "protocol",
            "openid-connect",
            "token",
        ]);
        let response = self
            .0
            .client
            .post(endpoint)
            .form(&[
                ("grant_type", "client_credentials"),
                ("client_id", self.0.client_id.as_str()),
                ("client_secret", self.0.client_secret.as_str()),
            ])
            .send()
            .await
            .map_err(|_| UNAVAILABLE)?;
        if !response.status().is_success() {
            return Err(UNAVAILABLE);
        }
        let token: TokenResponse = response.json().await.map_err(|_| UNAVAILABLE)?;
        if token.access_token.is_empty()
            || token.access_token.len() > 16 * 1024
            || token.expires_in == 0
        {
            return Err(UNAVAILABLE);
        }
        let expires =
            Instant::now() + Duration::from_secs(token.expires_in.min(86_400).saturating_sub(10));
        let access = token.access_token.clone();
        *cache = Some(Token {
            access_token: token.access_token,
            expires,
        });
        Ok(access)
    }

    pub(crate) async fn request(
        &self,
        method: Method,
        segments: &[&str],
        body: Option<&Value>,
    ) -> Result<Response, ApiError> {
        if segments.iter().any(|segment| !valid_segment(segment)) {
            return Err(UNAVAILABLE);
        }
        let mut path = vec!["admin", "realms", self.0.realm.as_str()];
        path.extend_from_slice(segments);
        let endpoint = self.endpoint(&path);
        let mut token = self.token().await?;
        let response = self.send(method.clone(), &endpoint, &token, body).await?;
        if response.status() != StatusCode::UNAUTHORIZED {
            return Ok(response);
        }
        {
            let mut cache = self.0.token.lock().await;
            if cache
                .as_ref()
                .is_some_and(|cached| cached.access_token == token)
            {
                *cache = None;
            }
        }
        token = self.token().await?;
        self.send(method, &endpoint, &token, body).await
    }

    async fn send(
        &self,
        method: Method,
        url: &Url,
        token: &str,
        body: Option<&Value>,
    ) -> Result<Response, ApiError> {
        let mut request = self
            .0
            .client
            .request(method, url.clone())
            .bearer_auth(token);
        if let Some(body) = body {
            request = request.json(body);
        }
        request.send().await.map_err(|_| UNAVAILABLE)
    }

    pub(super) async fn update(&self, subject: &str, payload: Value) -> Result<(), ApiError> {
        match self
            .request(Method::PUT, &["users", subject], Some(&payload))
            .await?
            .status()
        {
            StatusCode::NO_CONTENT => Ok(()),
            StatusCode::NOT_FOUND => Err(USER_NOT_FOUND),
            _ => Err(UNAVAILABLE),
        }
    }

    pub(super) async fn create(&self, payload: Value) -> Result<String, ApiError> {
        let response = self
            .request(Method::POST, &["users"], Some(&payload))
            .await?;
        if response.status() == StatusCode::CONFLICT {
            return Err(ApiError::new(
                StatusCode::CONFLICT,
                "keycloak_username_or_email_already_exists",
            ));
        }
        if response.status() != StatusCode::CREATED {
            return Err(UNAVAILABLE);
        }
        let location = response
            .headers()
            .get(LOCATION)
            .and_then(|value| value.to_str().ok())
            .ok_or(UNAVAILABLE)?;
        let url = self.0.base.join(location).map_err(|_| UNAVAILABLE)?;
        let mut path = url.path_segments().ok_or(UNAVAILABLE)?;
        let raw = path.next_back().ok_or(UNAVAILABLE)?;
        if path.next_back() != Some("users") {
            return Err(UNAVAILABLE);
        }
        let subject = percent_decode_str(raw)
            .decode_utf8()
            .map_err(|_| UNAVAILABLE)?;
        if !valid_segment(&subject) || subject.len() > 255 {
            return Err(UNAVAILABLE);
        }
        Ok(subject.into_owned())
    }

    pub(super) async fn delete(&self, subject: &str) -> Result<(), ApiError> {
        match self
            .request(Method::DELETE, &["users", subject], None)
            .await?
            .status()
        {
            StatusCode::NO_CONTENT | StatusCode::NOT_FOUND => Ok(()),
            _ => Err(UNAVAILABLE),
        }
    }

    pub(super) async fn reset_password(
        &self,
        subject: &str,
        password: &str,
        temporary: bool,
    ) -> Result<(), ApiError> {
        let payload = json!({"type":"password","value":password,"temporary":temporary});
        match self
            .request(
                Method::PUT,
                &["users", subject, "reset-password"],
                Some(&payload),
            )
            .await?
            .status()
        {
            StatusCode::NO_CONTENT => Ok(()),
            StatusCode::NOT_FOUND => Err(USER_NOT_FOUND),
            StatusCode::BAD_REQUEST => Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                "keycloak_password_policy_rejected",
            )),
            _ => Err(UNAVAILABLE),
        }
    }
}

fn env_value(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

fn valid_segment(segment: &str) -> bool {
    !segment.trim().is_empty()
        && !matches!(segment, "." | "..")
        && !segment.chars().any(char::is_control)
}
