use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    ApiError, Deployment, MAX_SESSION_RECORD, Revocation, SESSION_MAXIMUM, Session, UserSummary,
    VerifiedAccess, authentication_required, now, unavailable, valid_id,
};

const PROVIDER: &str = "oidc";
const MODE: &str = "online_sso_v1";

pub(super) struct Metadata {
    digest: String,
    user: UserSummary,
    features: Vec<String>,
    created_at: i64,
    pub revision: i64,
}

impl Metadata {
    pub(super) fn new(id: &str, user: UserSummary, features: Vec<String>) -> Self {
        Self {
            digest: digest(id),
            user,
            features,
            created_at: now(),
            revision: 1,
        }
    }
}

fn digest(secret: &str) -> String {
    format!("{:x}", Sha256::digest(secret.as_bytes()))
}

pub(super) fn key(scope: &str, secret: &str) -> String {
    format!("bff:{scope}:{}", digest(secret))
}

#[derive(Deserialize)]
struct Record {
    auth_provider: String,
    oidc_session_mode: String,
    session_id_hash: String,
    user: UserSummary,
    feature_names: Vec<String>,
    deployment: Deployment,
    access_token_encrypted: String,
    access_expires_at: i64,
    refresh_token_encrypted: String,
    csrf_secret: String,
    revoked: bool,
    created_at: i64,
    revision: i64,
    #[serde(default)]
    rust_retired: Vec<Revocation>,
}

#[derive(Serialize)]
struct Encoded<'a> {
    auth_provider: &'static str,
    oidc_session_mode: &'static str,
    session_id_hash: &'a str,
    user: &'a UserSummary,
    feature_names: &'a [String],
    deployment: &'a Deployment,
    access_token_encrypted: String,
    access_expires_at: i64,
    refresh_token_encrypted: String,
    refresh_expires_at: i64,
    csrf_secret: &'a str,
    revoked: bool,
    created_at: i64,
    updated_at: i64,
    revision: i64,
    rust_retired: &'a [Revocation],
}

#[derive(Deserialize)]
struct TokenClaims {
    iss: String,
    sub: String,
    azp: String,
    jti: String,
    exp: i64,
}

fn token_claims<T: serde::de::DeserializeOwned>(token: &str) -> Option<T> {
    if token.len() > 16 * 1024 {
        return None;
    }
    let mut parts = token.split('.');
    parts.next()?;
    let claims = URL_SAFE_NO_PAD.decode(parts.next()?).ok()?;
    parts.next()?;
    if parts.next().is_some() {
        return None;
    }
    serde_json::from_slice(&claims).ok()
}

fn decrypt(cipher: &fernet::Fernet, record: &str) -> Result<String, ApiError> {
    cipher
        .decrypt(record)
        .ok()
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .filter(|token| !token.is_empty() && token.len() <= 16 * 1024)
        .ok_or_else(authentication_required)
}

pub(super) fn decode(cipher: &fernet::Fernet, key: &str, raw: &[u8]) -> Result<Session, ApiError> {
    if raw.len() > MAX_SESSION_RECORD {
        return Err(authentication_required());
    }
    let data: Record = serde_json::from_slice(raw).map_err(|_| authentication_required())?;
    if data.revoked
        || data.auth_provider != PROVIDER
        || data.oidc_session_mode != MODE
        || key.strip_prefix("bff:session:") != Some(&data.session_id_hash)
        || !valid_id(&data.csrf_secret)
        || data.created_at <= 0
        || data.created_at > now()
        || data.revision <= 0
    {
        return Err(authentication_required());
    }

    let access_token = decrypt(cipher, &data.access_token_encrypted)?;
    let refresh_token = decrypt(cipher, &data.refresh_token_encrypted)?;
    // Recover the revocation keys of the stored credential even after access
    // expiry. The active and refresh paths verify a live token before use.
    let claims: TokenClaims = token_claims(&access_token).ok_or_else(authentication_required)?;
    if claims.exp != data.access_expires_at
        || claims.sub.is_empty()
        || claims.jti.is_empty()
        || claims.iss.is_empty()
    {
        return Err(authentication_required());
    }
    let access = VerifiedAccess {
        auth_time: None,
        session_id: None,
        session_id_invalid: false,
        subject: claims.sub,
        issuer: claims.iss,
        client_id: claims.azp,
        expires_at: claims.exp,
        token_id: claims.jti,
    };
    Ok(Session {
        access_token,
        refresh_token,
        subject: access.subject.clone(),
        csrf: data.csrf_secret,
        deployment: data.deployment,
        absolute_expiry: data.created_at.saturating_add(SESSION_MAXIMUM as i64),
        revocation: Revocation::from(&access),
        retired: data.rust_retired,
        shared: Some(Metadata {
            digest: data.session_id_hash,
            user: data.user,
            features: data.feature_names,
            created_at: data.created_at,
            revision: data.revision,
        }),
    })
}

pub(super) fn encode(cipher: &fernet::Fernet, session: &Session) -> Result<Vec<u8>, ApiError> {
    #[derive(Deserialize)]
    struct Expiry {
        exp: i64,
    }

    let metadata = session.shared.as_ref().ok_or_else(unavailable)?;
    let data = Encoded {
        auth_provider: PROVIDER,
        oidc_session_mode: MODE,
        session_id_hash: &metadata.digest,
        user: &metadata.user,
        feature_names: &metadata.features,
        deployment: &session.deployment,
        access_token_encrypted: cipher.encrypt(session.access_token.as_bytes()),
        access_expires_at: session.revocation.expires_at,
        refresh_token_encrypted: cipher.encrypt(session.refresh_token.as_bytes()),
        refresh_expires_at: token_claims::<Expiry>(&session.refresh_token)
            .map_or(0, |body| body.exp),
        csrf_secret: &session.csrf,
        revoked: false,
        created_at: metadata.created_at,
        updated_at: now(),
        revision: metadata.revision,
        rust_retired: &session.retired,
    };
    let encoded = serde_json::to_vec(&data).map_err(|_| unavailable())?;
    if encoded.len() > MAX_SESSION_RECORD {
        return Err(unavailable());
    }
    Ok(encoded)
}
