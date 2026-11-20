use axum::{body::Body, extract::RawQuery};
use percent_encoding::percent_decode_str;
use url::form_urlencoded;

use super::*;

const MAX_PLAYLIST_BYTES: usize = 1024 * 1024;

pub(super) fn same_playlist(initial: &Url, redirect: &Url) -> bool {
    redirect.origin() == initial.origin()
        && redirect.path() == initial.path()
        && redirect.username().is_empty()
        && redirect.password().is_none()
        && redirect.fragment().is_none()
}

struct HlsRequest {
    token: String,
    media_query: String,
}

impl HlsRequest {
    fn parse(query: Option<&str>) -> Result<Self, super::super::media_auth::Failure> {
        use super::super::media_auth::Failure as MediaFailure;
        let mut token = None;
        let mut media_query = form_urlencoded::Serializer::new(String::new());
        for (key, value) in form_urlencoded::parse(query.unwrap_or_default().as_bytes()) {
            if key == "mt" || key == "media_token" {
                if token.is_some() || !valid_session_id(&value) {
                    return Err(MediaFailure::Missing);
                }
                token = Some(value.into_owned());
            } else if matches!(key.as_ref(), "_HLS_msn" | "_HLS_part" | "_HLS_skip") {
                media_query.append_pair(&key, &value);
            }
        }
        Ok(Self {
            token: token.ok_or(MediaFailure::Missing)?,
            media_query: media_query.finish(),
        })
    }
}

pub(super) async fn stable(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    RawQuery(query): RawQuery,
) -> Result<Response, Response> {
    let runtime = runtime(&state).map_err(Failure::into_response)?;
    if !valid_session_id(&session_id) {
        return Err(Failure::new(StatusCode::NOT_FOUND, "session_not_found").into_response());
    }
    let request = HlsRequest::parse(query.as_deref()).map_err(IntoResponse::into_response)?;
    let session = runtime
        .load(&session_id)
        .await
        .map_err(Failure::into_response)?;
    let media_path = session
        .selected_path()
        .ok_or_else(|| Failure::unavailable().into_response())?;
    let public_path = format!("{}/{media_path}/", runtime.settings.public_hls);
    let uri = format!("{public_path}index.m3u8");
    state
        .media_auth
        .authorise_playback(
            request.token.clone(),
            &uri,
            &session_id,
            &session.revision,
            runtime.demand_prefix(),
        )
        .await
        .map_err(IntoResponse::into_response)?;
    let age = (chrono::Utc::now() - session.created_at)
        .to_std()
        .unwrap_or_default();
    if let Some(wait) = runtime.settings.startup_wait.checked_sub(age)
        && !wait.is_zero()
    {
        tokio::time::sleep(wait).await;
        state
            .media_auth
            .authorise_playback(
                request.token.clone(),
                &uri,
                &session_id,
                &session.revision,
                runtime.demand_prefix(),
            )
            .await
            .map_err(IntoResponse::into_response)?;
    }
    let (playlist, cookie) =
        fetch_playlist(&runtime.settings, &request.media_query, media_path).await?;
    let rewritten = rewrite_playlist(&playlist, &public_path, media_path, &request.token)
        .ok_or_else(invalid_playlist)?;
    let has_media = rewritten
        .lines()
        .any(|line| !line.is_empty() && !line.starts_with('#') || line.contains("URI=\"/"));
    let profile = match session.profile {
        Profile::Clean => "clean",
        Profile::Overlay => "overlay",
    };
    let mut response = Response::builder()
        .header(header::CONTENT_TYPE, "application/vnd.apple.mpegurl")
        .header(header::CACHE_CONTROL, "private, no-store")
        .header("x-playback-session", session_id)
        .header("x-playback-profile", profile)
        .header("x-hls-media-path", media_path)
        .header("x-hls-playlist-lines", rewritten.lines().count())
        .header(
            "x-hls-playlist-has-media-uri",
            if has_media { "true" } else { "false" },
        );
    if let Some(cookie) = cookie {
        response = response.header(header::SET_COOKIE, format!(
            "hlsSession={cookie}; Path={public_path}; Secure; HttpOnly; SameSite=None; Partitioned"
        ));
    }
    response
        .body(Body::from(rewritten))
        .map_err(|_| invalid_playlist())
}

fn invalid_playlist() -> Response {
    Failure::new(StatusCode::BAD_GATEWAY, "invalid_media_playlist").into_response()
}

fn unavailable() -> Response {
    Failure::new(StatusCode::BAD_GATEWAY, "media_playlist_unavailable").into_response()
}

fn not_ready() -> Response {
    let mut response =
        Failure::new(StatusCode::SERVICE_UNAVAILABLE, "media_playlist_not_ready").into_response();
    response
        .headers_mut()
        .insert(header::RETRY_AFTER, "2".parse().unwrap());
    response
}

async fn fetch_playlist(
    settings: &PlaybackSettings,
    query: &str,
    path: &str,
) -> Result<(String, Option<String>), Response> {
    let mut url = settings.upstream.clone();
    url.path_segments_mut()
        .map_err(|_| unavailable())?
        .pop_if_empty()
        .push(path)
        .push("index.m3u8");
    if !query.is_empty() {
        url.set_query(Some(query));
    }
    let mut response = settings
        .http
        .get(url)
        .send()
        .await
        .map_err(|_| unavailable())?;
    match response.status() {
        StatusCode::NOT_FOUND | StatusCode::SERVICE_UNAVAILABLE => return Err(not_ready()),
        StatusCode::OK => (),
        _ => return Err(unavailable()),
    }
    let cookie = response
        .headers()
        .get_all(header::SET_COOKIE)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .find_map(|cookie| cookie.split(';').next()?.strip_prefix("hlsSession="))
        .map(str::to_owned);
    if cookie.as_deref().is_some_and(|value| {
        value.is_empty()
            || value.len() > 4096
            || !value.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'~' | b'-')
            })
    }) {
        return Err(invalid_playlist());
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|_| unavailable())? {
        if bytes.len() + chunk.len() > MAX_PLAYLIST_BYTES {
            return Err(invalid_playlist());
        }
        bytes.extend_from_slice(&chunk);
    }
    if bytes.is_empty() {
        return Err(not_ready());
    }
    let playlist = String::from_utf8(bytes).map_err(|_| invalid_playlist())?;
    Ok((playlist, cookie))
}

fn rewrite_playlist(
    input: &str,
    public_path: &str,
    media_path: &str,
    token: &str,
) -> Option<String> {
    if input.lines().next()? != "#EXTM3U" {
        return None;
    }
    let base = Url::parse(&format!("https://preview.invalid{public_path}")).ok()?;
    let mut rewritten = String::with_capacity(input.len());
    for line in input.lines() {
        match line.strip_prefix('#') {
            Some(_) => {
                let mut rest = line;
                while let Some((before, after)) = rest.split_once("URI=\"") {
                    let (child, remaining) = after.split_once('"')?;
                    rewritten.push_str(before);
                    rewritten.push_str("URI=\"");
                    rewritten.push_str(&rewrite_uri(&base, media_path, child, token)?);
                    rewritten.push('"');
                    rest = remaining;
                }
                rewritten.push_str(rest);
            }
            None if !line.is_empty() => {
                rewritten.push_str(&rewrite_uri(&base, media_path, line, token)?)
            }
            None => (),
        }
        rewritten.push('\n');
        if rewritten.len() > MAX_PLAYLIST_BYTES {
            return None;
        }
    }
    Some(rewritten)
}

fn rewrite_uri(base: &Url, media_path: &str, child: &str, token: &str) -> Option<String> {
    if child.is_empty() || child.chars().any(|c| c.is_control() || c == '\\') {
        return None;
    }
    let target = base.join(child).ok()?;
    if !matches!(target.scheme(), "http" | "https") || target.fragment().is_some() {
        return None;
    }
    let direct_base = format!("/{media_path}/");
    let suffix = target
        .path()
        .strip_prefix(base.path())
        .or_else(|| target.path().strip_prefix(&direct_base))?;
    if suffix.is_empty() {
        return None;
    }
    for segment in suffix.split('/') {
        let decoded = percent_decode_str(segment).decode_utf8().ok()?;
        if decoded.is_empty()
            || decoded == "."
            || decoded == ".."
            || decoded
                .chars()
                .any(|c| c.is_control() || matches!(c, '/' | '\\' | '"' | '?' | '#'))
        {
            return None;
        }
    }
    let mut query = form_urlencoded::Serializer::new(String::new());
    for (key, value) in target.query_pairs() {
        if key != "mt" && key != "media_token" {
            query.append_pair(&key, &value);
        }
    }
    query.append_pair("mt", token);
    Some(format!("{}{suffix}?{}", base.path(), query.finish()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redirects_stay_on_the_configured_playlist() {
        let initial = Url::parse("http://127.0.0.1:8888/camera/index.m3u8").unwrap();
        assert!(same_playlist(
            &initial,
            &initial.join("?cookieCheck=1").unwrap()
        ));
        assert!(!same_playlist(
            &initial,
            &initial.join("/other/index.m3u8").unwrap()
        ));
        assert!(!same_playlist(
            &initial,
            &initial
                .join("http://127.0.0.1:9000/camera/index.m3u8")
                .unwrap()
        ));
    }

    #[test]
    fn rewrites_playlist_tag_and_segment_uris_with_one_local_capability() {
        let media = "hazard_c2l0ZQ_Y2Ft_preview";
        let base = format!("/hazard/media/{media}/");
        let input = format!(
            "#EXTM3U\r\n#EXT-X-MAP:URI=\"init.mp4?mt=old\"\r\n#EXT-X-MEDIA:URI=\"http://internal/{media}/a.m3u8?x=y\"\npart%201.m4s\n"
        );
        let playlist = rewrite_playlist(&input, &base, media, "fixture").unwrap();
        assert!(playlist.contains(&format!("#EXT-X-MAP:URI=\"{base}init.mp4?mt=fixture\"")));
        assert!(playlist.contains(&format!("#EXT-X-MEDIA:URI=\"{base}a.m3u8?x=y&mt=fixture\"")));
        assert!(playlist.contains(&format!("{base}part%201.m4s?mt=fixture")));
        assert!(rewrite_playlist("#EXTM3U\n../../private", &base, media, "fixture").is_none());
        assert!(rewrite_playlist("#EXTM3U\npart%2Fsecret.m4s", &base, media, "fixture").is_none());
        assert!(
            rewrite_playlist(
                "#EXTM3U\n#EXT-X-MAP:URI=\"//evil/track.aac\"",
                &base,
                media,
                "fixture"
            )
            .is_none()
        );
    }
}
