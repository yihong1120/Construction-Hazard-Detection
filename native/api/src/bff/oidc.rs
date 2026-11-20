use axum::{
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    response::Response,
};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use percent_encoding::percent_decode_str;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    ApiError, AppState,
    auth::{Authorizer, VerifiedAccess},
};

use super::{
    Bff, MAX_TOKEN_RESPONSE, STATE_TTL, authentication_required, configured_deployment_changed,
    redirect, session, state_bff, unavailable,
};

#[derive(Deserialize)]
pub(super) struct LoginQuery {
    return_to: Option<String>,
    idp_hint: Option<String>,
}

#[derive(Deserialize)]
pub(super) struct CallbackQuery {
    state: Option<String>,
    code: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct LoginState {
    verifier: String,
    return_to: String,
    deployment: session::Deployment,
}

#[derive(Deserialize)]
pub(super) struct Tokens {
    pub access_token: String,
    pub refresh_token: String,
}

fn return_path(candidate: Option<&str>) -> &str {
    let Some(candidate) = candidate.map(str::trim) else {
        return "/";
    };
    let decoded = percent_decode_str(candidate).decode_utf8_lossy();
    if candidate.len() > 2048
        || !candidate.starts_with('/')
        || decoded.starts_with("//")
        || decoded.contains('\\')
        || decoded.chars().any(char::is_control)
        || !candidate
            .bytes()
            .all(|character| character.is_ascii_graphic())
    {
        "/"
    } else {
        candidate
    }
}

fn login_hint(value: Option<&str>) -> Result<Option<&str>, ApiError> {
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    match () {
        () if value.eq_ignore_ascii_case("google") => Ok(Some("google")),
        () if value.eq_ignore_ascii_case("apple") => Ok(Some("apple")),
        () => Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "oidc_identity_provider_not_supported",
        )),
    }
}

fn flow_key(state: &str, browser: &str) -> String {
    session::key("oidc:state", &format!("{state}.{browser}"))
}

pub(super) async fn account(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    let bff = state_bff(&state, &headers)?;
    redirect(StatusCode::TEMPORARY_REDIRECT, bff.0.urls.account.as_str())
}

pub(super) async fn login(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<LoginQuery>,
) -> Result<Response, ApiError> {
    let bff = state_bff(&state, &headers)?;
    let deployment = session::deployment(&state.db, &bff).await?;
    let hint = login_hint(query.idp_hint.as_deref())?;
    let state = session::opaque_id()?;
    let verifier = session::opaque_id()?;
    let browser = session::cookie(&headers, bff.flow_cookie())
        .map_or_else(session::opaque_id, |cookie| Ok(cookie.to_owned()))?;
    let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(&verifier));
    let record = LoginState {
        verifier,
        return_to: return_path(query.return_to.as_deref()).to_owned(),
        deployment,
    };
    bff.insert(&flow_key(&state, &browser), &record, STATE_TTL)
        .await?;
    let mut target = bff.0.urls.authorization.clone();
    {
        let mut params = target.query_pairs_mut();
        params.extend_pairs([
            ("client_id", bff.0.client_id.as_str()),
            ("code_challenge", &challenge),
            ("code_challenge_method", "S256"),
            ("redirect_uri", &bff.0.urls.redirect_uri),
            ("response_type", "code"),
            ("scope", "openid profile email"),
            ("state", &state),
        ]);
        if let Some(hint) = hint {
            params.append_pair("kc_idp_hint", hint);
        }
    }
    let mut response = redirect(StatusCode::TEMPORARY_REDIRECT, target.as_str())?;
    bff.set_cookie(&mut response, bff.flow_cookie(), &browser, STATE_TTL);
    Ok(response)
}

pub(super) async fn callback(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<CallbackQuery>,
) -> Result<Response, ApiError> {
    let bff = state_bff(&state, &headers)?;
    let invalid_state = || ApiError::new(StatusCode::BAD_REQUEST, "oidc_state_invalid");
    let state_secret = query
        .state
        .as_deref()
        .filter(|value| session::valid_id(value))
        .ok_or_else(invalid_state)?;
    let browser = session::cookie(&headers, bff.flow_cookie()).ok_or_else(invalid_state)?;
    let record: LoginState = bff
        .consume(&flow_key(state_secret, browser))
        .await?
        .ok_or_else(invalid_state)?;
    let code = query
        .code
        .as_deref()
        .filter(|code| !code.is_empty() && code.len() <= 4096)
        .ok_or_else(|| ApiError::new(StatusCode::BAD_REQUEST, "oidc_authorization_failed"))?;
    let deployment = session::deployment(&state.db, &bff).await?;
    if deployment != record.deployment {
        return Err(configured_deployment_changed());
    }
    let tokens = token_exchange(
        &bff,
        &[
            ("code", code),
            ("code_verifier", &record.verifier),
            ("grant_type", "authorization_code"),
            ("redirect_uri", &bff.0.urls.redirect_uri),
        ],
    )
    .await?;
    let access = verify(&bff, &state.auth, &tokens.access_token).await?;
    let (user, features) = session::identity(&state.db, &bff, &deployment, &access.subject).await?;
    let id = session::establish(&bff, tokens, &access, deployment, user, features).await?;
    let mut response = redirect(StatusCode::SEE_OTHER, return_path(Some(&record.return_to)))?;
    bff.set_cookie(&mut response, bff.session_cookie(), &id, bff.0.session_ttl);
    bff.set_cookie(&mut response, bff.flow_cookie(), "", 0);
    Ok(response)
}

pub(super) async fn global_logout(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<CallbackQuery>,
) -> Result<Response, ApiError> {
    let bff = state_bff(&state, &headers)?;
    let invalid = || ApiError::new(StatusCode::BAD_REQUEST, "oidc_logout_state_invalid");
    let proof = query
        .state
        .as_deref()
        .filter(|value| session::valid_id(value))
        .ok_or_else(invalid)?;
    let consumed: Option<String> = redis::cmd("GETDEL")
        .arg(bff.logout_key(proof))
        .query_async(&mut bff.0.redis.clone())
        .await
        .map_err(|_| unavailable())?;
    if consumed.as_deref() != Some("1") {
        return Err(invalid());
    }
    let mut url = bff.0.urls.end_session.clone();
    url.query_pairs_mut()
        .append_pair("client_id", &bff.0.client_id)
        .append_pair("post_logout_redirect_uri", bff.0.urls.post_logout.as_str());
    redirect(StatusCode::SEE_OTHER, url.as_str())
}

pub(super) async fn verify(
    bff: &Bff,
    auth: &Authorizer,
    token: &str,
) -> Result<VerifiedAccess, ApiError> {
    let access = auth.verify_access_token(token).await?;
    if access.client_id != bff.0.client_id {
        return Err(authentication_required());
    }
    Ok(access)
}

pub(super) async fn token_exchange(bff: &Bff, form: &[(&str, &str)]) -> Result<Tokens, ApiError> {
    let failure = |status| ApiError::new(status, "oidc_token_exchange_failed");
    let unavailable = || failure(StatusCode::SERVICE_UNAVAILABLE);
    let mut response = bff
        .0
        .oidc_http
        .post(bff.0.urls.token.clone())
        .basic_auth(&bff.0.client_id, Some(&bff.0.client_secret))
        .form(form)
        .send()
        .await
        .map_err(|_| unavailable())?;
    let status = response.status();
    let mut payload = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|_| unavailable())? {
        if payload.len() + chunk.len() > MAX_TOKEN_RESPONSE {
            return Err(unavailable());
        }
        payload.extend_from_slice(&chunk);
    }
    if !status.is_success() {
        #[derive(Deserialize)]
        struct ProviderError {
            error: String,
        }
        if matches!(status.as_u16(), 400 | 401 | 403)
            && serde_json::from_slice::<ProviderError>(&payload)
                .is_ok_and(|error| error.error == "invalid_grant")
        {
            return Err(failure(StatusCode::UNAUTHORIZED));
        }
        if status == StatusCode::TOO_MANY_REQUESTS {
            return Err(failure(status));
        }
        return Err(unavailable());
    }
    let tokens: Tokens = serde_json::from_slice(&payload).map_err(|_| unavailable())?;
    if tokens.access_token.is_empty() || tokens.refresh_token.is_empty() {
        return Err(unavailable());
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn return_target_cannot_escape_the_origin() {
        for target in [
            "/\\other.example",
            "//other.example",
            "https://other.example",
            "/%2fother.example",
            "/%5Cother.example",
            "/\t/other.example",
        ] {
            assert_eq!(return_path(Some(target)), "/");
        }
        assert_eq!(
            return_path(Some("/wall?site=demo#camera")),
            "/wall?site=demo#camera"
        );
        assert_eq!(login_hint(Some("GOOGLE")).unwrap(), Some("google"));
        assert!(login_hint(Some("external")).is_err());
    }
}
