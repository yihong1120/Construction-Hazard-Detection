mod deployment;
pub(crate) mod jwks;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{Result, ensure};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use redis::aio::ConnectionManager;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use url::Url;

use crate::ApiError;
pub(crate) use deployment::DatabaseAccess;
pub use deployment::DatabaseAccessConfig;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OidcSettings {
    pub issuer: String,
    pub jwks_url: String,
    pub audiences: Vec<String>,
}

impl OidcSettings {
    pub fn validate(&self) -> Result<()> {
        for value in [&self.issuer, &self.jwks_url] {
            let url = Url::parse(value).map_err(|_| anyhow::anyhow!("Invalid OIDC URL"))?;
            let local_http = url.scheme() == "http"
                && url.host().is_some_and(|host| match host {
                    url::Host::Ipv4(address) => address.is_loopback(),
                    url::Host::Ipv6(address) => address.is_loopback(),
                    url::Host::Domain(_) => false,
                });
            ensure!(
                (url.scheme() == "https" || local_http)
                    && url.has_host()
                    && url.username().is_empty()
                    && url.password().is_none()
                    && url.fragment().is_none(),
                "OIDC URLs require HTTPS or loopback HTTP"
            );
        }
        ensure!(
            !self.audiences.is_empty() && self.audiences.iter().all(|a| !a.trim().is_empty()),
            "Configure at least one OIDC audience"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Grant {
    AccountRead,
    AccountManage,
    EvidenceRead,
    EvidenceWrite,
    ImageDetection,
    ModelDownload,
    ModelManage,
    NotificationRead,
    NotificationDispatch,
    StreamRead,
    ViolationFeedback,
    ViolationRead,
    ViolationReview,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SiteGrant {
    id: i64,
    name: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceGrant {
    subject: String,
    client_id: String,
    permissions: HashSet<Grant>,
    sites: Vec<SiteGrant>,
}

pub fn validate_grants(grants: &[ServiceGrant]) -> Result<()> {
    ensure!(!grants.is_empty(), "Configure at least one API grant");
    let mut identities = HashSet::new();
    let mut site_ids = HashMap::new();
    let mut site_names = HashMap::new();
    for grant in grants {
        ensure!(
            !grant.subject.trim().is_empty()
                && !grant.client_id.trim().is_empty()
                && !grant.permissions.is_empty()
                && !grant.sites.is_empty(),
            "Each API grant needs a subject, client ID, permissions and sites"
        );
        ensure!(
            identities.insert((&grant.subject, &grant.client_id)),
            "Duplicate API identity"
        );
        let mut sites = HashSet::new();
        for site in &grant.sites {
            ensure!(
                site.id > 0 && !site.name.trim().is_empty() && sites.insert(site.id),
                "Invalid or duplicate site in API grant"
            );
            ensure!(
                *site_ids.entry(site.name.as_str()).or_insert(site.id) == site.id
                    && *site_names.entry(site.id).or_insert(site.name.as_str()) == site.name,
                "Site names and IDs must map uniquely across API grants"
            );
        }
    }
    Ok(())
}

pub struct Actor {
    pub subject: String,
    pub user_id: Option<i32>,
    pub username: Option<String>,
    pub role: Option<String>,
    pub access_expires_at: u64,
    pub access_revocation_key: String,
    pub oidc_revocation_key: String,
    sites: Arc<HashMap<i64, String>>,
}

impl Actor {
    pub fn allows_site(&self, id: i64, name: &str) -> bool {
        self.sites.get(&id).is_some_and(|site| site == name)
    }

    pub fn sites(&self) -> Vec<(i64, String)> {
        self.sites
            .iter()
            .map(|(&id, name)| (id, name.clone()))
            .collect()
    }
}

struct PermissionSet {
    permissions: HashSet<Grant>,
    sites: Arc<HashMap<i64, String>>,
}

#[derive(Deserialize)]
struct Claims {
    iss: String,
    sub: String,
    azp: String,
    jti: String,
    iat: i64,
    exp: i64,
    typ: String,
    auth_time: Option<serde_json::Value>,
    sid: Option<serde_json::Value>,
    session_state: Option<serde_json::Value>,
}

pub(crate) struct VerifiedAccess {
    pub subject: String,
    pub client_id: String,
    pub issuer: String,
    pub expires_at: i64,
    pub token_id: String,
    pub auth_time: Option<f64>,
    pub session_id: Option<String>,
    pub session_id_invalid: bool,
}

pub(crate) fn access_revocation_key(token_id: &str) -> String {
    format!(
        "auth:access-revoked:{:x}",
        Sha256::digest(token_id.as_bytes())
    )
}

impl VerifiedAccess {
    pub fn oidc_revocation_key(&self) -> String {
        access_revocation_key(&format!("oidc:{}:{}", self.issuer, self.token_id))
    }
}

struct Inner {
    verifier: jwks::Verifier,
    grants: HashMap<(String, String), PermissionSet>,
    redis: ConnectionManager,
    database: Option<DatabaseAccess>,
}

#[derive(Clone)]
pub struct Authorizer(Arc<Inner>);

fn invalid_token() -> ApiError {
    ApiError::new(StatusCode::UNAUTHORIZED, "Could not validate credentials")
}

fn auth_unavailable() -> ApiError {
    ApiError::new(
        StatusCode::SERVICE_UNAVAILABLE,
        "Authentication service unavailable",
    )
}

pub(crate) fn bearer(headers: &HeaderMap) -> Result<&str, ApiError> {
    let mut values = headers.get_all(header::AUTHORIZATION).iter();
    let value = values
        .next()
        .and_then(|value| value.to_str().ok())
        .ok_or_else(invalid_token)?;
    let (scheme, token) = value.split_once(' ').ok_or_else(invalid_token)?;
    if values.next().is_some()
        || !scheme.eq_ignore_ascii_case("Bearer")
        || token.is_empty()
        || token.len() > 16 * 1024
        || token.bytes().any(|byte| byte.is_ascii_whitespace())
    {
        return Err(invalid_token());
    }
    Ok(token)
}

pub fn websocket_credentials(
    headers: &HeaderMap,
    query: Option<&str>,
) -> Result<HeaderMap, ApiError> {
    let mut credentials = HeaderMap::new();
    let mut value = if let Some(value) = headers.get(header::AUTHORIZATION) {
        bearer(headers)?;
        value.clone()
    } else {
        let mut tokens = url::form_urlencoded::parse(query.unwrap_or_default().as_bytes())
            .filter_map(|(key, value)| (key == "token").then_some(value));
        let token = tokens.next().ok_or_else(invalid_token)?;
        if tokens.next().is_some() || token.len() > 16 * 1024 {
            return Err(invalid_token());
        }
        HeaderValue::from_str(&format!("Bearer {token}")).map_err(|_| invalid_token())?
    };
    value.set_sensitive(true);
    credentials.insert(header::AUTHORIZATION, value);
    for host in headers.get_all(header::HOST) {
        credentials.append(header::HOST, host.clone());
    }
    bearer(&credentials)?;
    Ok(credentials)
}

impl Authorizer {
    #[cfg(feature = "tensorrt")]
    pub(crate) fn redis(&self) -> ConnectionManager {
        self.0.redis.clone()
    }

    pub fn new(
        settings: OidcSettings,
        grants: Vec<ServiceGrant>,
        redis: ConnectionManager,
    ) -> Result<Self> {
        Self::configured(settings, grants, redis, None)
    }

    pub(crate) fn configured(
        settings: OidcSettings,
        grants: Vec<ServiceGrant>,
        redis: ConnectionManager,
        database: Option<DatabaseAccess>,
    ) -> Result<Self> {
        settings.validate()?;
        if database.is_none() || !grants.is_empty() {
            validate_grants(&grants)?;
        }
        let verifier = jwks::Verifier::new(
            settings.jwks_url,
            &[settings.issuer.trim_end_matches('/')],
            &settings.audiences,
        )?;
        let grants = grants
            .into_iter()
            .map(|entry| {
                (
                    (entry.subject, entry.client_id),
                    PermissionSet {
                        permissions: entry.permissions,
                        sites: Arc::new(
                            entry
                                .sites
                                .into_iter()
                                .map(|site| (site.id, site.name))
                                .collect(),
                        ),
                    },
                )
            })
            .collect();
        Ok(Self(Arc::new(Inner {
            verifier,
            grants,
            redis,
            database,
        })))
    }

    pub async fn authorize(&self, headers: &HeaderMap, grant: Grant) -> Result<Actor, ApiError> {
        let access = self.verify_access_token(bearer(headers)?).await?;
        self.authorize_verified(headers, grant, access).await
    }

    pub(crate) async fn authorize_verified(
        &self,
        headers: &HeaderMap,
        grant: Grant,
        access: VerifiedAccess,
    ) -> Result<Actor, ApiError> {
        let oidc_revocation_key = access.oidc_revocation_key();
        let configured = self
            .0
            .grants
            .get(&(access.subject.clone(), access.client_id.clone()));
        let (sites, account) = match &self.0.database {
            Some(database) => {
                database
                    .authorize(headers, &access, grant, configured)
                    .await?
            }
            None => {
                let entry = configured
                    .filter(|permissions| permissions.permissions.contains(&grant))
                    .ok_or_else(permission_denied)?;
                (entry.sites.clone(), None)
            }
        };
        let (user_id, username, role) = account.map_or((None, None, None), |user| {
            (Some(user.id), Some(user.username), Some(user.role))
        });
        Ok(Actor {
            access_expires_at: access.expires_at as u64,
            access_revocation_key: access_revocation_key(&access.token_id),
            oidc_revocation_key,
            subject: access.subject,
            user_id,
            username,
            role,
            sites,
        })
    }

    pub(crate) async fn verify_access_token(
        &self,
        token: &str,
    ) -> Result<VerifiedAccess, ApiError> {
        let claims: Claims = self
            .0
            .verifier
            .verify(token)
            .await
            .map_err(verification_error)?;
        if claims.typ != "Bearer"
            || claims.sub.trim().is_empty()
            || claims.azp.trim().is_empty()
            || claims.jti.trim().is_empty()
            || claims.iat < 0
            || claims.iat > chrono::Utc::now().timestamp()
        {
            return Err(invalid_token());
        }
        let mut redis = self.0.redis.clone();
        let session = claims
            .sid
            .filter(|value| !value.is_null() && value.as_str() != Some(""))
            .or(claims.session_state);
        let access = VerifiedAccess {
            auth_time: claims.auth_time.and_then(|value| value.as_f64()),
            session_id_invalid: session
                .as_ref()
                .is_some_and(|value| !value.is_null() && !value.is_string()),
            session_id: session.and_then(|value| value.as_str().map(str::to_owned)),
            subject: claims.sub,
            client_id: claims.azp,
            issuer: claims.iss,
            expires_at: claims.exp,
            token_id: claims.jti,
        };
        let revoked: u8 = redis::cmd("EXISTS")
            .arg(access_revocation_key(&access.token_id))
            .arg(access.oidc_revocation_key())
            .query_async(&mut redis)
            .await
            .map_err(|_| auth_unavailable())?;
        if revoked != 0 {
            return Err(invalid_token());
        }
        Ok(access)
    }

    pub async fn ready(&self) -> Result<(), ApiError> {
        let mut redis = self.0.redis.clone();
        redis::cmd("PING")
            .query_async::<String>(&mut redis)
            .await
            .map_err(|_| auth_unavailable())?;
        self.0.verifier.ready().await.map_err(verification_error)
    }
}

fn verification_error(error: jwks::Error) -> ApiError {
    match error {
        jwks::Error::Invalid => invalid_token(),
        jwks::Error::Unavailable => auth_unavailable(),
    }
}

fn permission_denied() -> ApiError {
    ApiError::new(
        StatusCode::FORBIDDEN,
        "Account has no grant for this operation",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;
    use serde_json::json;

    #[test]
    fn requires_one_bearer_authorization_header() {
        let mut headers = HeaderMap::new();
        assert!(bearer(&headers).is_err());
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Basic test"),
        );
        assert!(bearer(&headers).is_err());
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer  test"),
        );
        assert!(bearer(&headers).is_err());
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("bearer test"),
        );
        assert_eq!(bearer(&headers).unwrap(), "test");
        headers.append(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer another"),
        );
        assert!(bearer(&headers).is_err());
    }

    #[test]
    fn accepts_only_authenticated_transport_or_literal_loopback() {
        let mut settings = OidcSettings {
            issuer: "https://id.example/realm".into(),
            jwks_url: "http://127.0.0.1:8123/jwks".into(),
            audiences: vec!["example".into()],
        };
        assert!(settings.validate().is_ok());
        settings.jwks_url = "http://id.example/jwks".into();
        assert!(settings.validate().is_err());
        settings.jwks_url = "https://credentials@id.example/jwks".into();
        assert!(settings.validate().is_err());
    }

    #[test]
    fn rejects_ambiguous_site_labels_across_identities() {
        let grants: Vec<ServiceGrant> = serde_json::from_value(json!([
            {"subject":"one","client_id":"client","permissions":["stream_read"],"sites":[{"id":1,"name":"demo"}]},
            {"subject":"two","client_id":"client","permissions":["evidence_read"],"sites":[{"id":2,"name":"demo"}]}
        ])).unwrap();
        assert!(validate_grants(&grants).is_err());
    }
}
