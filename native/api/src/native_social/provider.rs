use std::{path::PathBuf, time::Duration};

use anyhow::{Context, Result, ensure};
use axum::http::StatusCode;
use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
use serde::{Deserialize, Serialize};
use url::Url;

use crate::{ApiError, auth::jwks};

use super::Provider;

const GOOGLE_KEYS: &str = "https://www.googleapis.com/oauth2/v3/certs";
const GOOGLE_ISSUERS: &[&str] = &["accounts.google.com", "https://accounts.google.com"];
const APPLE_ISSUER: &str = "https://appleid.apple.com";
const APPLE_KEYS: &str = "https://appleid.apple.com/auth/keys";
const APPLE_TOKENS: &str = "https://appleid.apple.com/auth/token";
const MAX_RESPONSE: usize = 64 * 1024;

pub(super) const INVALID_TOKEN: ApiError =
    ApiError::new(StatusCode::UNAUTHORIZED, "Invalid provider token");
pub(super) const INVALID_CREDENTIAL: ApiError =
    ApiError::new(StatusCode::UNAUTHORIZED, "invalid_provider_credential");
const PROVIDER_UNAVAILABLE: ApiError = ApiError::new(
    StatusCode::SERVICE_UNAVAILABLE,
    "native_social_provider_unavailable",
);
const APPLE_UNCONFIGURED: ApiError = ApiError::new(
    StatusCode::INTERNAL_SERVER_ERROR,
    "Apple client secret is not configured",
);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub google_client_ids: Vec<String>,
    pub apple_client_ids: Vec<String>,
    pub apple: Option<AppleConfig>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AppleConfig {
    pub team_id: String,
    pub key_id: String,
    pub private_key_env: Option<String>,
    pub private_key_path: Option<PathBuf>,
    pub service_id: Option<String>,
    pub bundle_id: Option<String>,
    pub redirect_uri: Option<String>,
}

impl Config {
    pub(super) fn validate(&self) -> Result<()> {
        for ids in [&self.google_client_ids, &self.apple_client_ids] {
            ensure!(
                !ids.is_empty()
                    && ids.len() <= 32
                    && ids.iter().all(|id| {
                        (1..=256).contains(&id.len())
                            && id.bytes().all(|c| {
                                c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.')
                            })
                    }),
                "Configure the exact Google and Apple token audiences"
            );
        }
        if let Some(apple) = &self.apple {
            ensure!(
                [apple.team_id.as_str(), apple.key_id.as_str()]
                    .iter()
                    .all(|id| {
                        (1..=80).contains(&id.len())
                            && id.bytes().all(|c| c.is_ascii_alphanumeric())
                    }),
                "Invalid Apple team or signing key ID"
            );
            ensure!(
                apple.private_key_env.is_some() || apple.private_key_path.is_some(),
                "Specify the Apple signing key"
            );
            if let Some(name) = &apple.private_key_env {
                ensure!(
                    super::environment_name(name),
                    "Invalid Apple signing key environment name"
                );
            }
            if let Some(path) = &apple.private_key_path {
                ensure!(path.is_absolute(), "Use an absolute Apple signing key path");
            }
            if let Some(id) = &apple.service_id {
                ensure!(
                    self.apple_client_ids.contains(id),
                    "Apple service ID must be an allowed audience"
                );
                let uri = apple
                    .redirect_uri
                    .as_deref()
                    .context("Set the Apple service redirect URI")?;
                let url = Url::parse(uri).context("Invalid Apple service redirect URI")?;
                ensure!(
                    url.scheme() == "https"
                        && url.has_host()
                        && url.username().is_empty()
                        && url.password().is_none()
                        && url.fragment().is_none(),
                    "Apple service redirect URI must use HTTPS"
                );
            }
            if let Some(id) = &apple.bundle_id {
                ensure!(
                    self.apple_client_ids.contains(id),
                    "Apple bundle ID must be an allowed audience"
                );
            }
        }
        Ok(())
    }
}

pub(super) struct Providers {
    google: jwks::Verifier,
    apple_jwks: jwks::Verifier,
    apple: Option<Apple>,
}

struct Apple {
    config: AppleConfig,
    client: reqwest::Client,
    signing_key: EncodingKey,
    code_clients: Vec<String>,
    token_url: String,
}

#[derive(Deserialize)]
struct Claims {
    sub: String,
    aud: String,
    iat: Option<i64>,
    nonce: Option<String>,
    email: Option<String>,
    #[serde(default)]
    email_verified: EmailVerified,
}

#[derive(Default, Deserialize)]
#[serde(untagged)]
enum EmailVerified {
    Boolean(bool),
    AppleString(String),
    #[default]
    Missing,
}

impl EmailVerified {
    fn verified(&self, provider: Provider) -> bool {
        match self {
            Self::Boolean(value) => *value,
            Self::AppleString(value) if provider == Provider::Apple => value == "true",
            _ => false,
        }
    }
}

pub(super) struct Identity {
    pub provider: Provider,
    pub subject: String,
    pub email: Option<String>,
}

impl Providers {
    pub(super) fn new(config: Config) -> Result<Self> {
        Self::with_endpoints(config, GOOGLE_KEYS, APPLE_KEYS, APPLE_TOKENS)
    }

    pub(super) fn with_endpoints(
        config: Config,
        google_keys: &str,
        apple_keys: &str,
        apple_tokens: &str,
    ) -> Result<Self> {
        config.validate()?;
        let google = jwks::Verifier::new(
            google_keys.into(),
            GOOGLE_ISSUERS,
            &config.google_client_ids,
        )?;
        let apple_jwks =
            jwks::Verifier::new(apple_keys.into(), &[APPLE_ISSUER], &config.apple_client_ids)?;
        let apple = config
            .apple
            .map(|apple| Apple::new(apple, config.apple_client_ids, apple_tokens.into()))
            .transpose()?;
        Ok(Self {
            google,
            apple_jwks,
            apple,
        })
    }

    pub(super) async fn verify(
        &self,
        provider: Provider,
        token: Option<&str>,
        code: Option<&str>,
        nonce: &str,
    ) -> Result<Identity, ApiError> {
        let claims = match provider {
            Provider::Google => {
                let token = token.filter(|_| code.is_none()).ok_or(INVALID_CREDENTIAL)?;
                verified(&self.google, token).await?
            }
            Provider::Apple => {
                let code = code.ok_or(INVALID_CREDENTIAL)?;
                self.apple_identity(token, code, nonce).await?
            }
        };
        validate_nonce(&claims, nonce)?;
        if claims.sub.is_empty() || claims.sub.len() > 512 {
            return Err(INVALID_CREDENTIAL);
        }
        let email = claims
            .email
            .filter(|_| claims.email_verified.verified(provider))
            .map(|mail| mail.trim().to_lowercase())
            .filter(|mail| !mail.is_empty());
        Ok(Identity {
            provider,
            subject: claims.sub,
            email,
        })
    }

    async fn apple_identity(
        &self,
        token: Option<&str>,
        code: &str,
        nonce: &str,
    ) -> Result<Claims, ApiError> {
        let apple = self.apple.as_ref().ok_or(APPLE_UNCONFIGURED)?;
        if let Some(token) = token {
            let original = verified(&self.apple_jwks, token).await?;
            validate_nonce(&original, nonce)?;
            if let Some(exchanged) = apple.exchange(code, &original.aud).await? {
                let matched = verified(&self.apple_jwks, &exchanged).await?;
                if matched.sub != original.sub
                    || matched.aud != original.aud
                    || matched.nonce.as_deref().is_some_and(|value| value != nonce)
                {
                    return Err(INVALID_TOKEN);
                }
            }
            return Ok(original);
        }

        for id in &apple.code_clients {
            match apple.exchange(code, id).await {
                Ok(Some(token)) => {
                    let claims = verified(&self.apple_jwks, &token).await?;
                    if &claims.aud != id {
                        return Err(INVALID_TOKEN);
                    }
                    return Ok(claims);
                }
                Ok(None) => return Err(INVALID_TOKEN),
                Err(error) if error.status() == StatusCode::UNAUTHORIZED => {}
                Err(error) => return Err(error),
            }
        }
        Err(INVALID_TOKEN)
    }
}

fn validate_nonce(claims: &Claims, nonce: &str) -> Result<(), ApiError> {
    if claims.nonce.as_deref() != Some(nonce) {
        return Err(INVALID_TOKEN);
    }
    Ok(())
}

async fn verified(verifier: &jwks::Verifier, token: &str) -> Result<Claims, ApiError> {
    let claims: Claims = verifier.verify(token).await.map_err(|error| match error {
        jwks::Error::Invalid => INVALID_TOKEN,
        jwks::Error::Unavailable => PROVIDER_UNAVAILABLE,
    })?;
    if claims
        .iat
        .is_some_and(|iat| iat < 0 || iat > chrono::Utc::now().timestamp())
    {
        return Err(INVALID_TOKEN);
    }
    Ok(claims)
}

impl Apple {
    fn new(config: AppleConfig, audiences: Vec<String>, token_url: String) -> Result<Self> {
        let pem = config
            .private_key_env
            .as_ref()
            .and_then(|name| std::env::var(name).ok())
            .filter(|value| !value.trim().is_empty())
            .map(|value| value.replace("\\n", "\n"))
            .map(Ok)
            .or_else(|| {
                config
                    .private_key_path
                    .as_ref()
                    .map(std::fs::read_to_string)
            })
            .context("Apple signing key is missing")?
            .context("Read Apple signing key")?;
        let signing_key =
            EncodingKey::from_ec_pem(pem.as_bytes()).context("Invalid Apple signing key")?;
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        let mut code_clients = Vec::new();
        for id in config
            .service_id
            .iter()
            .chain(config.bundle_id.iter())
            .chain(&audiences)
        {
            if !code_clients.contains(id) {
                code_clients.push(id.clone());
            }
        }
        Ok(Self {
            config,
            client,
            signing_key,
            code_clients,
            token_url,
        })
    }

    async fn exchange(&self, code: &str, client_id: &str) -> Result<Option<String>, ApiError> {
        #[derive(Serialize)]
        struct Secret<'a> {
            iss: &'a str,
            aud: &'static str,
            sub: &'a str,
            iat: i64,
            exp: i64,
        }
        #[derive(Deserialize)]
        struct TokenResponse {
            id_token: Option<String>,
        }
        let now = chrono::Utc::now().timestamp();
        let mut header = Header::new(Algorithm::ES256);
        header.kid = Some(self.config.key_id.clone());
        let secret = encode(
            &header,
            &Secret {
                iss: &self.config.team_id,
                aud: APPLE_ISSUER,
                sub: client_id,
                iat: now,
                exp: now + 300,
            },
            &self.signing_key,
        )
        .map_err(|_| APPLE_UNCONFIGURED)?;
        let mut form = vec![
            ("grant_type", "authorization_code"),
            ("code", code),
            ("client_id", client_id),
            ("client_secret", &secret),
        ];
        if self.config.service_id.as_deref() == Some(client_id)
            && let Some(redirect) = &self.config.redirect_uri
        {
            form.push(("redirect_uri", redirect));
        }
        let mut response = self
            .client
            .post(&self.token_url)
            .form(&form)
            .send()
            .await
            .map_err(|_| PROVIDER_UNAVAILABLE)?;
        if matches!(
            response.status(),
            StatusCode::BAD_REQUEST | StatusCode::UNAUTHORIZED
        ) {
            return Err(INVALID_TOKEN);
        }
        if !response.status().is_success() {
            return Err(PROVIDER_UNAVAILABLE);
        }
        let mut body = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(|_| PROVIDER_UNAVAILABLE)? {
            if body.len() + chunk.len() > MAX_RESPONSE {
                return Err(INVALID_TOKEN);
            }
            body.extend_from_slice(&chunk);
        }
        serde_json::from_slice::<TokenResponse>(&body)
            .map(|tokens| tokens.id_token)
            .map_err(|_| INVALID_TOKEN)
    }
}
