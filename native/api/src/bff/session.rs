mod shared;

use std::time::Duration;

use aws_lc_rs::{
    aead::{Aad, NONCE_LEN, Nonce, RandomizedNonceKey},
    constant_time::verify_slices_are_equal,
};
use axum::{
    Json,
    extract::State,
    http::{HeaderMap, StatusCode, header},
    response::Response,
};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::json;
use sha2::{Digest, Sha256};
use sqlx::{PgPool, Row};
use uuid::Uuid;

use super::{
    Bff, MAX_TOKEN_RESPONSE, SESSION_MAXIMUM, STATE_TTL, authentication_required,
    configured_deployment_changed, no_store, oidc, single_header, state_bff, unavailable,
};
use crate::{
    ApiError, AppState,
    auth::{Authorizer, VerifiedAccess, access_revocation_key},
};

const MAX_SESSION_RECORD: usize = 2 * MAX_TOKEN_RESPONSE;
const REFRESH_AHEAD: i64 = 30;
const MAX_OLD_TOKENS: usize = 32;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(super) struct Deployment {
    pub deployment_id: Uuid,
    pub tenant_id: Uuid,
    pub config_revision: i32,
}

pub(super) async fn deployment(db: &PgPool, bff: &Bff) -> Result<Deployment, ApiError> {
    let row = sqlx::query(
        "SELECT d.id, d.tenant_id, d.config_revision
         FROM public.deployments d JOIN public.tenants t ON t.id = d.tenant_id
         WHERE d.api_base_url = $1 AND d.status = 'active' AND t.status = 'active'",
    )
    .bind(&bff.0.urls.public_api_base_url)
    .fetch_optional(db)
    .await
    .map_err(|_| unavailable())?
    .ok_or_else(configured_deployment_changed)?;
    Ok(Deployment {
        deployment_id: row.try_get("id").map_err(|_| unavailable())?,
        tenant_id: row.try_get("tenant_id").map_err(|_| unavailable())?,
        config_revision: row.try_get("config_revision").map_err(|_| unavailable())?,
    })
}

#[derive(Deserialize, Serialize)]
pub(super) struct UserSummary {
    id: i32,
    username: String,
    display_name: String,
    role: String,
    group_id: Option<i32>,
    status: String,
}

pub(super) async fn identity(
    db: &PgPool,
    bff: &Bff,
    deployment: &Deployment,
    subject: &str,
) -> Result<(UserSummary, Vec<String>), ApiError> {
    let row = sqlx::query(
        "SELECT u.id, u.username, u.role, u.group_id, u.status,
            COALESCE(NULLIF(concat_ws(' ', NULLIF(p.given_name, ''), NULLIF(p.family_name, '')), ''), u.username)
                AS display_name,
            ARRAY(SELECT f.feature_name::text
                  FROM group_features gf JOIN features f ON gf.feature_id = f.id
                  WHERE gf.group_id = u.group_id ORDER BY f.feature_name) AS feature_names
         FROM public.user_identities i
         JOIN users u ON u.id = i.user_id
         LEFT JOIN user_profiles p ON p.user_id = u.id
         WHERE i.provider = $1 AND i.provider_user_id = $2 AND u.tenant_id = $3
               AND u.status = 'active' AND NOT u.deletion_pending",
    )
    .bind(&bff.0.identity_provider)
    .bind(subject)
    .bind(deployment.tenant_id)
    .fetch_optional(db)
    .await
    .map_err(|_| unavailable())?
    .ok_or_else(authentication_required)?;
    Ok((
        UserSummary {
            id: row.try_get("id").map_err(|_| unavailable())?,
            username: row.try_get("username").map_err(|_| unavailable())?,
            display_name: row.try_get("display_name").map_err(|_| unavailable())?,
            role: row.try_get("role").map_err(|_| unavailable())?,
            group_id: row.try_get("group_id").map_err(|_| unavailable())?,
            status: row.try_get("status").map_err(|_| unavailable())?,
        },
        row.try_get("feature_names").map_err(|_| unavailable())?,
    ))
}

pub(super) fn opaque_id() -> Result<String, ApiError> {
    let mut bytes = [0u8; 32];
    aws_lc_rs::rand::fill(&mut bytes).map_err(|_| unavailable())?;
    Ok(URL_SAFE_NO_PAD.encode(bytes))
}

pub(super) fn valid_id(value: &str) -> bool {
    value.len() == 43
        && URL_SAFE_NO_PAD
            .decode(value)
            .is_ok_and(|bytes| bytes.len() == 32)
}

pub(super) fn key(scope: &str, secret: &str) -> String {
    format!("bff:{scope}:rust:{:x}", Sha256::digest(secret.as_bytes()))
}

pub(super) fn cookie<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    let mut result = None;
    for header in headers.get_all(header::COOKIE) {
        for attribute in header.to_str().ok()?.split(';') {
            let Some((cookie_name, value)) = attribute.trim().split_once('=') else {
                continue;
            };
            if cookie_name == name {
                if result.is_some() || !valid_id(value) {
                    return None;
                }
                result = Some(value);
            }
        }
    }
    result
}

fn seal<T: Serialize>(
    cipher: &RandomizedNonceKey,
    key: &str,
    value: &T,
) -> Result<Vec<u8>, ApiError> {
    let mut bytes = serde_json::to_vec(value).map_err(|_| unavailable())?;
    if bytes.len() > MAX_SESSION_RECORD - NONCE_LEN - 16 {
        return Err(unavailable());
    }
    let nonce = cipher
        .seal_in_place_append_tag(Aad::from(key.as_bytes()), &mut bytes)
        .map_err(|_| unavailable())?;
    let mut record = Vec::with_capacity(bytes.len() + NONCE_LEN);
    record.extend_from_slice(nonce.as_ref());
    record.extend_from_slice(&bytes);
    Ok(record)
}

fn open<T: DeserializeOwned>(
    cipher: &RandomizedNonceKey,
    key: &str,
    data: &[u8],
) -> Result<T, ApiError> {
    if data.len() > MAX_SESSION_RECORD {
        return Err(authentication_required());
    }
    let (nonce, encrypted) = data
        .split_at_checked(NONCE_LEN)
        .ok_or_else(authentication_required)?;
    let nonce = Nonce::try_assume_unique_for_key(nonce).map_err(|_| authentication_required())?;
    let mut encrypted = encrypted.to_vec();
    let plaintext = cipher
        .open_in_place(nonce, Aad::from(key.as_bytes()), &mut encrypted)
        .map_err(|_| authentication_required())?;
    serde_json::from_slice(plaintext).map_err(|_| authentication_required())
}

#[derive(Deserialize, Serialize)]
struct Revocation {
    raw_key: String,
    issuer_key: String,
    expires_at: i64,
}

impl From<&VerifiedAccess> for Revocation {
    fn from(access: &VerifiedAccess) -> Self {
        Self {
            raw_key: access_revocation_key(&access.token_id),
            issuer_key: access.oidc_revocation_key(),
            expires_at: access.expires_at,
        }
    }
}

#[derive(Deserialize, Serialize)]
pub(super) struct Session {
    pub access_token: String,
    refresh_token: String,
    subject: String,
    csrf: String,
    deployment: Deployment,
    absolute_expiry: i64,
    revocation: Revocation,
    retired: Vec<Revocation>,
    #[serde(skip)]
    shared: Option<shared::Metadata>,
}

fn now() -> i64 {
    chrono::Utc::now().timestamp()
}

impl Session {
    pub(super) fn ttl(&self, bff: &Bff) -> Result<u64, ApiError> {
        let remaining = self.absolute_expiry.saturating_sub(now());
        if remaining <= 0 {
            return Err(authentication_required());
        }
        Ok(bff.0.session_ttl.min(remaining as u64))
    }
}

impl Bff {
    fn session_key(&self, id: &str) -> String {
        if self.0.shared_session_key.is_some() {
            shared::key("session", id)
        } else {
            key("session", id)
        }
    }

    pub(super) fn logout_key(&self, state: &str) -> String {
        if self.0.shared_session_key.is_some() {
            shared::key("oidc:logout", state)
        } else {
            key("oidc:logout", state)
        }
    }

    fn encode_session(&self, key: &str, session: &Session) -> Result<Vec<u8>, ApiError> {
        match &self.0.shared_session_key {
            Some(cipher) => shared::encode(cipher, session),
            None => seal(&self.0.session_key, key, session),
        }
    }

    pub(super) async fn insert<T: Serialize>(
        &self,
        key: &str,
        record: &T,
        ttl: u64,
    ) -> Result<(), ApiError> {
        let encrypted = seal(&self.0.session_key, key, record)?;
        self.insert_encoded(key, encrypted, ttl).await
    }

    async fn insert_encoded(
        &self,
        key: &str,
        encrypted: Vec<u8>,
        ttl: u64,
    ) -> Result<(), ApiError> {
        let result = redis::cmd("SET")
            .arg(key)
            .arg(encrypted)
            .arg("EX")
            .arg(ttl)
            .arg("NX")
            .query_async::<Option<String>>(&mut self.0.redis.clone())
            .await
            .map_err(|_| unavailable())?;
        result.ok_or_else(unavailable)?;
        Ok(())
    }

    pub(super) async fn consume<T: DeserializeOwned>(
        &self,
        key: &str,
    ) -> Result<Option<T>, ApiError> {
        let raw: Option<Vec<u8>> = redis::cmd("GETDEL")
            .arg(key)
            .query_async(&mut self.0.redis.clone())
            .await
            .map_err(|_| unavailable())?;
        raw.map(|raw| open(&self.0.session_key, key, &raw))
            .transpose()
    }

    async fn snapshot(&self, key: &str) -> Result<(Vec<u8>, Session), ApiError> {
        let raw: Option<Vec<u8>> = redis::cmd("GET")
            .arg(key)
            .query_async(&mut self.0.redis.clone())
            .await
            .map_err(|_| unavailable())?;
        let raw = raw.ok_or_else(authentication_required)?;
        let session: Session = match &self.0.shared_session_key {
            Some(cipher) => shared::decode(cipher, key, &raw)?,
            None => open(&self.0.session_key, key, &raw)?,
        };
        session.ttl(self)?;
        Ok((raw, session))
    }
}

pub(super) async fn establish(
    bff: &Bff,
    tokens: oidc::Tokens,
    access: &VerifiedAccess,
    deployment: Deployment,
    user: UserSummary,
    feature_names: Vec<String>,
) -> Result<String, ApiError> {
    let id = opaque_id()?;
    let key = bff.session_key(&id);
    let session = Session {
        access_token: tokens.access_token,
        refresh_token: tokens.refresh_token,
        csrf: opaque_id()?,
        subject: access.subject.clone(),
        deployment,
        absolute_expiry: now() + SESSION_MAXIMUM as i64,
        revocation: Revocation::from(access),
        retired: Vec::new(),
        shared: bff
            .0
            .shared_session_key
            .as_ref()
            .map(|_| shared::Metadata::new(&id, user, feature_names)),
    };
    bff.insert_encoded(&key, bff.encode_session(&key, &session)?, session.ttl(bff)?)
        .await?;
    Ok(id)
}

fn require_csrf(headers: &HeaderMap, bff: &Bff, expected: &str) -> Result<(), ApiError> {
    if !single_header(headers, header::ORIGIN).is_some_and(|origin| {
        origin == bff.0.urls.browser_origin
            || bff.0.cors_origins.iter().any(|entry| entry == origin)
    }) {
        return Err(ApiError::new(StatusCode::FORBIDDEN, "invalid_origin"));
    }
    let provided = single_header(headers, header::HeaderName::from_static("x-csrf-token"))
        .filter(|value| valid_id(value));
    if provided
        .is_none_or(|value| verify_slices_are_equal(value.as_bytes(), expected.as_bytes()).is_err())
    {
        return Err(ApiError::new(StatusCode::FORBIDDEN, "invalid_csrf_token"));
    }
    Ok(())
}

pub(super) struct Active {
    pub id: String,
    pub session: Session,
    user: UserSummary,
    feature_names: Vec<String>,
}

pub(super) async fn active(
    state: &AppState,
    bff: &Bff,
    headers: &HeaderMap,
    mutation: bool,
) -> Result<Active, ApiError> {
    let id = cookie(headers, bff.session_cookie()).ok_or_else(authentication_required)?;
    let key = bff.session_key(id);
    let (raw, mut session) = bff.snapshot(&key).await?;
    if mutation {
        require_csrf(headers, bff, &session.csrf)?;
    }
    let deployed = deployment(&state.db, bff).await?;
    if session.deployment != deployed {
        return Err(configured_deployment_changed());
    }
    let (user, feature_names) = identity(&state.db, bff, &deployed, &session.subject).await?;
    if session.revocation.expires_at <= now() + REFRESH_AHEAD {
        session = refresh(bff, &state.auth, key.clone(), raw, session).await?;
    } else {
        let access = oidc::verify(bff, &state.auth, &session.access_token).await?;
        if access.subject != session.subject {
            return Err(authentication_required());
        }
    }
    let renewed = redis::cmd("EXPIRE")
        .arg(&key)
        .arg(session.ttl(bff)?)
        .query_async::<bool>(&mut bff.0.redis.clone())
        .await
        .map_err(|_| unavailable())?;
    if !renewed {
        return Err(authentication_required());
    }
    Ok(Active {
        id: id.to_owned(),
        session,
        user,
        feature_names,
    })
}

pub(super) async fn refresh_for_proxy_retry(
    state: &AppState,
    bff: &Bff,
    active: &mut Active,
) -> Result<(), ApiError> {
    let key = bff.session_key(&active.id);
    let (raw, latest) = bff.snapshot(&key).await?;
    if latest.subject != active.session.subject || latest.deployment != active.session.deployment {
        return Err(authentication_required());
    }
    active.session = if latest.access_token == active.session.access_token {
        refresh(bff, &state.auth, key, raw, latest).await?
    } else {
        let verified = oidc::verify(bff, &state.auth, &latest.access_token).await?;
        if verified.subject != latest.subject {
            return Err(authentication_required());
        }
        latest
    };
    Ok(())
}

async fn refresh(
    bff: &Bff,
    auth: &Authorizer,
    key: String,
    mut raw: Vec<u8>,
    mut session: Session,
) -> Result<Session, ApiError> {
    let lock = if session.shared.is_some() {
        format!("{key}:refresh-lock")
    } else {
        format!("{key}:refresh")
    };
    let owner = opaque_id()?;
    let mut connection = bff.0.redis.clone();
    let acquired = redis::cmd("SET")
        .arg(&lock)
        .arg(&owner)
        .arg("EX")
        .arg(30)
        .arg("NX")
        .query_async::<Option<String>>(&mut connection)
        .await
        .map_err(|_| unavailable())?;
    if acquired.is_none() {
        for _ in 0..20 {
            tokio::time::sleep(Duration::from_millis(75)).await;
            let (latest, refreshed) = bff.snapshot(&key).await?;
            if latest != raw {
                let verified = oidc::verify(bff, auth, &refreshed.access_token).await?;
                if verified.subject != session.subject || session.deployment != refreshed.deployment
                {
                    return Err(authentication_required());
                }
                return Ok(refreshed);
            }
        }
        return Err(ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "session_refresh_busy",
        ));
    }
    let bff = bff.clone();
    let auth = auth.clone();
    // Finish an accepted provider rotation even if the browser disconnects.
    tokio::spawn(async move {
        let result =
            async {
                let (latest, latest_session) = bff.snapshot(&key).await?;
                if latest != raw {
                    let verified = oidc::verify(&bff, &auth, &latest_session.access_token).await?;
                    if verified.subject != session.subject
                        || session.deployment != latest_session.deployment
                    {
                        return Err(authentication_required());
                    }
                    return Ok(latest_session);
                }
                raw = latest;
                let revoked = redis::cmd("EXISTS")
                    .arg(&session.revocation.raw_key)
                    .arg(&session.revocation.issuer_key)
                    .query_async::<u8>(&mut connection)
                    .await
                    .map_err(|_| unavailable())?;
                if revoked != 0 {
                    return Err(authentication_required());
                }
                session.retired.retain(|entry| now() < entry.expires_at);
                if session.retired.len() >= MAX_OLD_TOKENS {
                    return Err(unavailable());
                }
                let tokens = match oidc::token_exchange(
                    &bff,
                    &[
                        ("grant_type", "refresh_token"),
                        ("refresh_token", &session.refresh_token),
                    ],
                )
                .await
                {
                    Ok(tokens) => tokens,
                    Err(error) if error.status() == StatusCode::UNAUTHORIZED => {
                        let _ = revoke(&bff, &key, &raw, &session, None).await?;
                        return Err(authentication_required());
                    }
                    Err(error) => return Err(error),
                };
                let access = oidc::verify(&bff, &auth, &tokens.access_token).await?;
                if access.subject != session.subject {
                    return Err(authentication_required());
                }
                let old_raw_key = session.revocation.raw_key.clone();
                let old_issuer_key = session.revocation.issuer_key.clone();
                let revocation = Revocation::from(&access);
                if revocation.raw_key != old_raw_key {
                    session
                        .retired
                        .push(std::mem::replace(&mut session.revocation, revocation));
                } else {
                    session.revocation = revocation;
                }
                session.access_token = tokens.access_token;
                session.refresh_token = tokens.refresh_token;
                if let Some(metadata) = &mut session.shared {
                    metadata.revision = metadata.revision.checked_add(1).ok_or_else(unavailable)?;
                }
                let value = bff.encode_session(&key, &session)?;
                let updated = redis::Script::new(
                "if redis.call('GET', KEYS[1]) ~= ARGV[1] or redis.call('GET', KEYS[2]) ~= ARGV[2]
                   or redis.call('EXISTS', KEYS[3], KEYS[4]) > 0 then return 0 end
                 redis.call('SET', KEYS[1], ARGV[3], 'EX', ARGV[4])
                 return 1",
            )
            .key(&key).key(&lock).key(&old_raw_key).key(&old_issuer_key)
            .arg(&raw).arg(&owner).arg(value).arg(session.ttl(&bff)?)
            .invoke_async::<bool>(&mut connection).await.map_err(|_| unavailable())?;
                if !updated {
                    return Err(authentication_required());
                }
                Ok(session)
            }
            .await;
        let _ = redis::Script::new(
            "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) end
             return 0",
        )
        .key(&lock)
        .arg(&owner)
        .invoke_async::<u8>(&mut connection)
        .await;
        result
    })
    .await
    .map_err(|_| unavailable())?
}

async fn revoke(
    bff: &Bff,
    key: &str,
    raw: &[u8],
    session: &Session,
    bridge: Option<&str>,
) -> Result<bool, ApiError> {
    let script = redis::Script::new(
        "if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
         local now = tonumber(redis.call('TIME')[1])
         for i = 3, #KEYS, 2 do
            local exp = tonumber(ARGV[(i + 1) / 2 + 1])
            if exp > now then
              redis.call('SET', KEYS[i], '1', 'EXAT', exp)
              redis.call('SET', KEYS[i+1], '1', 'EXAT', exp)
            end
         end
         if #KEYS[2] > 0 then redis.call('SET', KEYS[2], '1', 'EX', ARGV[2]) end
         redis.call('DEL', KEYS[1])
         return 1",
    );
    let mut args = script.prepare_invoke();
    let bridge_key = bridge.map_or_else(String::new, |bridge| bff.logout_key(bridge));
    let logout_ttl = if session.shared.is_some() {
        120
    } else {
        STATE_TTL
    };
    args.key(key).key(bridge_key).arg(raw).arg(logout_ttl);
    for entry in std::iter::once(&session.revocation).chain(&session.retired) {
        args.key(&entry.raw_key)
            .key(&entry.issuer_key)
            .arg(entry.expires_at);
    }
    args.invoke_async(&mut bff.0.redis.clone())
        .await
        .map_err(|_| unavailable())
}

pub(super) async fn current(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    let bff = state_bff(&state, &headers)?;
    let active = active(&state, &bff, &headers, false).await?;
    let ttl = active.session.ttl(&bff)?;
    let mut response = no_store(Json(json!({
        "authenticated": true, "user": active.user,
        "feature_names": active.feature_names, "deployment": active.session.deployment,
    })));
    bff.set_cookie(&mut response, bff.session_cookie(), &active.id, ttl);
    Ok(response)
}

pub(super) async fn csrf(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    let bff = state_bff(&state, &headers)?;
    let active = active(&state, &bff, &headers, false).await?;
    let ttl = active.session.ttl(&bff)?;
    let mut response = no_store(Json(json!({"csrf_token": active.session.csrf})));
    bff.set_cookie(&mut response, bff.session_cookie(), &active.id, ttl);
    Ok(response)
}

pub(super) async fn logout(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    let bff = state_bff(&state, &headers)?;
    let id = cookie(&headers, bff.session_cookie()).ok_or_else(authentication_required)?;
    let key = bff.session_key(id);
    let bridge = opaque_id()?;
    for _ in 0..3 {
        let (raw, session) = bff.snapshot(&key).await?;
        require_csrf(&headers, &bff, &session.csrf)?;
        if revoke(&bff, &key, &raw, &session, Some(&bridge)).await? {
            let mut response = no_store(Json(json!({
                "global_logout_url": format!("/bff/auth/oidc/logout?state={bridge}"),
            })));
            bff.set_cookie(&mut response, bff.session_cookie(), "", 0);
            return Ok(response);
        }
    }
    Err(unavailable())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aws_lc_rs::aead::AES_256_GCM;

    #[test]
    fn session_cipher_rejects_swapping_keys_and_tampering() {
        let key = RandomizedNonceKey::new(&AES_256_GCM, &[0x21; 32]).unwrap();
        let token = "fixture-access-token";
        let mut encrypted = seal(&key, "session-a", &token).unwrap();
        assert!(
            !encrypted
                .windows(token.len())
                .any(|window| window == token.as_bytes())
        );
        assert_eq!(
            open::<String>(&key, "session-a", &encrypted).unwrap(),
            token
        );
        assert!(open::<String>(&key, "session-b", &encrypted).is_err());
        let last = encrypted.len() - 1;
        encrypted[last] ^= 1;
        assert!(open::<String>(&key, "session-a", &encrypted).is_err());
    }

    #[test]
    fn access_cookies_reject_duplicates() {
        let id = opaque_id().unwrap();
        let mut headers = HeaderMap::new();
        headers.append(header::COOKIE, format!("session={id}").parse().unwrap());
        assert_eq!(cookie(&headers, "session"), Some(id.as_str()));
        headers.append(header::COOKIE, format!("session={id}").parse().unwrap());
        assert_eq!(cookie(&headers, "session"), None);
    }
}
