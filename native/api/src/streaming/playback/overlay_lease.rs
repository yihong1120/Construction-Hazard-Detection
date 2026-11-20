use std::time::Duration;

use axum::body::Bytes;
use tokio::time::Instant;

use super::*;

// Count distinct active languages across both renditions before touching a key.
const LEASE: &str = r#"
local now_active, requested_is_active = 0, false
for idx = 3, #KEYS, 2 do
    if redis.call('EXISTS', KEYS[idx], KEYS[idx + 1]) > 0 then
        now_active = now_active + 1
        if (idx - 3) / 2 + 1 == tonumber(ARGV[2]) then requested_is_active = true end
    end
end
if not requested_is_active and now_active >= tonumber(ARGV[3]) then return -1 end
redis.call('SET', KEYS[1], '1', 'EX', math.max(tonumber(ARGV[1]), redis.call('TTL', KEYS[1])))
return redis.call('EXISTS', KEYS[2])
"#;

pub(in crate::streaming) struct OverlayLease {
    redis: ConnectionManager,
    demand_key: String,
    ready_key: String,
    sibling_keys: Vec<String>,
    language_position: usize,
    ttl: u32,
    limit: u32,
    next_refresh: Instant,
    next_ready: Instant,
    ready_payload: Option<Bytes>,
    pending: Option<Bytes>,
}

pub(in crate::streaming) fn backend_mode(mode: Option<&str>) -> bool {
    mode.is_some_and(|mode| {
        matches!(
            mode.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on" | "backend" | "annotated"
        )
    })
}

impl Playback {
    pub(in crate::streaming) async fn metadata_overlay(
        &self,
        db: &sqlx::PgPool,
        media: &Settings,
        site_id: i64,
        site: &str,
        stream_id: &str,
        language: Option<&str>,
    ) -> Result<OverlayLease, ApiError> {
        let camera = decode_camera(stream_id)
            .filter(|camera| valid_camera(camera))
            .ok_or_else(|| ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, "invalid_stream_id"))?;
        verify_cameras(
            db,
            &[Plan {
                label: site.to_owned(),
                site_id,
                key: camera.clone(),
                session_id: None,
                profile: Profile::Overlay,
                rendition: Rendition::Detail,
                language: None,
            }],
        )
        .await
        .map_err(|error| match error.0 {
            StatusCode::NOT_FOUND => ApiError::new(StatusCode::NOT_FOUND, "stream_not_found"),
            _ => overlay_unavailable(),
        })?;
        let language = normalise_label_language(language.unwrap_or("en"));
        let language_position = media
            .languages
            .iter()
            .position(|code| *code == language)
            .ok_or_else(|| {
                ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, "unsupported_language")
            })?
            + 1;
        let base = base_path(site, &camera);
        let prefix = self.demand_prefix();
        let demand = |path: &str, code: &str| {
            format!(
                "{prefix}media_overlay_demand:{path}:{}",
                URL_SAFE_NO_PAD.encode(code)
            )
        };
        let selected = format!("{base}_annotated_{}", URL_SAFE_NO_PAD.encode(language));
        let url = format!("{}/{selected}/index.m3u8", self.settings.public_hls);
        let preview = format!("{base}_preview");
        let sibling_keys = media
            .languages
            .iter()
            .flat_map(|code| [demand(&base, code), demand(&preview, code)])
            .collect();
        let payload = json!({
            "profile": "overlay", "status": "ready", "state": "ready",
            "playback_ready": true, "overlay_ready": true, "language": language,
            "media_path": selected, "playback_url": url, "media_hls_url": url,
        });
        let now = Instant::now();
        let mut lease = OverlayLease {
            redis: self.redis.clone(),
            demand_key: demand(&base, language),
            ready_key: format!("{prefix}media_overlay_ready:{selected}"),
            sibling_keys,
            language_position,
            ttl: self.settings.overlay_demand_ttl,
            limit: media.overlay_limit,
            next_refresh: now,
            next_ready: now,
            ready_payload: Some(Bytes::from(format!(
                "event: overlay_ready\ndata: {payload}\n\n"
            ))),
            pending: None,
        };
        lease.pending = lease.tick().await?;
        Ok(lease)
    }
}

fn overlay_unavailable() -> ApiError {
    ApiError::new(
        StatusCode::SERVICE_UNAVAILABLE,
        "overlay_storage_unavailable",
    )
}

impl OverlayLease {
    pub(in crate::streaming) fn take_pending(&mut self) -> Option<Bytes> {
        self.pending.take()
    }

    pub(in crate::streaming) fn next_tick(&self) -> Instant {
        if self.ready_payload.is_some() {
            self.next_ready.min(self.next_refresh)
        } else {
            self.next_refresh
        }
    }

    pub(in crate::streaming) async fn tick(&mut self) -> Result<Option<Bytes>, ApiError> {
        let now = Instant::now();
        let ready = if now >= self.next_refresh {
            let state: i64 = redis::Script::new(LEASE)
                .key(&self.demand_key)
                .key(&self.ready_key)
                .key(&self.sibling_keys)
                .arg(self.ttl)
                .arg(self.language_position)
                .arg(self.limit)
                .invoke_async(&mut self.redis)
                .await
                .map_err(|_| overlay_unavailable())?;
            if state < 0 {
                return Err(ApiError::new(
                    StatusCode::TOO_MANY_REQUESTS,
                    "overlay_language_limit_reached",
                ));
            }
            self.next_refresh = now + Duration::from_secs_f64((f64::from(self.ttl) / 2.).max(0.25));
            state == 1
        } else {
            redis::cmd("EXISTS")
                .arg(&self.ready_key)
                .query_async(&mut self.redis)
                .await
                .map_err(|_| overlay_unavailable())?
        };
        self.next_ready = now + Duration::from_secs(1);
        Ok(if ready {
            self.ready_payload.take()
        } else {
            None
        })
    }
}
