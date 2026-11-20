mod support;
use axum::{
    Form, Json, Router,
    extract::{Multipart, State},
    http::{HeaderMap, StatusCode},
    routing::post,
};
use construction_pipeline::{
    config::Config,
    outbox::{Event, Outbox},
    runtime,
};
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};
use tokio_util::sync::CancellationToken;

#[derive(Default)]
struct Spy {
    uploads: Mutex<HashMap<String, i64>>,
    notified: Mutex<HashSet<i64>>,
    upload_warnings: Mutex<HashMap<String, Value>>,
    notification_warnings: Mutex<HashMap<String, Value>>,
    upload_requests: AtomicUsize,
    notification_requests: AtomicUsize,
    tokens: AtomicUsize,
    reject_first_upload: AtomicBool,
    retry_first_notification: AtomicBool,
}
async fn upload(
    State(spy): State<Arc<Spy>>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> (StatusCode, Json<Value>) {
    spy.upload_requests.fetch_add(1, Ordering::SeqCst);
    let mut fields = HashMap::new();
    let mut image = vec![];
    while let Some(field) = multipart.next_field().await.unwrap() {
        let name = field.name().unwrap().to_string();
        if name == "image" {
            image = field.bytes().await.unwrap().to_vec()
        } else {
            fields.insert(name, field.text().await.unwrap());
        }
    }
    // Consume the request before replying so this tests HTTP 401 refresh,
    // rather than an early response racing a still-streaming multipart body.
    if spy.reject_first_upload.swap(false, Ordering::SeqCst) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({"error":"fixture auth refresh"})),
        );
    }
    assert!(image::load_from_memory(&image).is_ok());
    for key in [
        "site_id",
        "model_id",
        "model_version",
        "detection_time",
        "warnings_json",
        "detections_json",
    ] {
        assert!(!fields[key].is_empty());
    }
    assert_eq!(
        headers["idempotency-key"].to_str().unwrap(),
        fields["event_id"]
    );
    spy.upload_warnings.lock().unwrap().insert(
        fields["stream_name"].clone(),
        serde_json::from_str(&fields["warnings_json"]).unwrap(),
    );
    let mut uploads = spy.uploads.lock().unwrap();
    let next = uploads.len() as i64 + 1;
    let id = *uploads.entry(fields["event_id"].clone()).or_insert(next);
    (StatusCode::OK, Json(json!({"violation_id":id})))
}
async fn notify(State(spy): State<Arc<Spy>>, Json(body): Json<Value>) -> (StatusCode, Json<Value>) {
    spy.notification_requests.fetch_add(1, Ordering::SeqCst);
    if spy.retry_first_notification.swap(false, Ordering::SeqCst) {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error":"fixture unavailable"})),
        );
    }
    assert_eq!(body["type"], "violation");
    assert!(body["site_id"].as_i64().unwrap() > 0);
    spy.notification_warnings.lock().unwrap().insert(
        body["stream_name"].as_str().unwrap().to_owned(),
        body["body"].clone(),
    );
    spy.notified
        .lock()
        .unwrap()
        .insert(body["violation_id"].as_i64().unwrap());
    (StatusCode::OK, Json(json!({"outcome":"completed"})))
}
async fn token(
    State(spy): State<Arc<Spy>>,
    Form(body): Form<HashMap<String, String>>,
) -> Json<Value> {
    assert_eq!(body["grant_type"], "client_credentials");
    assert_eq!(body["client_id"], "fixture-client");
    assert_eq!(body["client_secret"], "fixture-secret");
    let count = spy.tokens.fetch_add(1, Ordering::SeqCst) + 1;
    Json(json!({"access_token":format!("fixture-token-{count}"),"expires_in":3600}))
}
async fn server(spy: Arc<Spy>) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = Router::new()
        .route("/upload", post(upload))
        .route("/notify", post(notify))
        .route("/token", post(token))
        .with_state(spy);
    (
        format!("http://{address}"),
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        }),
    )
}
fn config(root: &Path, cameras: usize) -> Config {
    support::check_postgres_if_configured();
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/videos/test.mp4")
        .canonicalize()
        .unwrap();
    let replay = root.join("replay.json");
    std::fs::write(
        &replay,
        serde_json::to_vec(&json!([[
            [10, 10, 40, 50, 0.9, 2],
            [50, 10, 80, 50, 0.9, 4]
        ]]))
        .unwrap(),
    )
    .unwrap();
    let streams:Vec<_>=(0..cameras).map(|i|json!({"id":format!("camera-{i}"),"site_id":1,"site":"fixture","name":format!("camera-{i}"),
        "source":source,"model":"fixture","width":640,"height":352,"work_start_hour":0,"work_end_hour":24,
        "evidence_cooldown_seconds":0,"languages":["en"],"output":{"clean":root.join(format!("clean-{i}.mp4")),"annotated":root.join(format!("overlay-{i}.mp4"))}})).collect();
    serde_json::from_value(json!({"state_dir":root.join("state"),"outbox":{"url_env":"CHD_RUST_TEST_DATABASE_URL"},"experimental_replay":true,"lossless":true,"max_frames":4,
        "capture":{"max_convert_fps":1},
        "batch_size":8,"models":[{"key":"fixture","artifact":"","sha256":"0".repeat(64),"version":"fixture-v1","replay":replay}],"streams":streams})).unwrap()
}

#[test]
fn work_hours_follow_the_cameras_local_same_day_window() {
    let root = tempfile::tempdir().unwrap();
    let mut cfg = config(root.path(), 1);
    let spec = &mut cfg.streams[0];
    spec.timezone = "Asia/Taipei".into();
    spec.work_start_hour = 7;
    spec.work_end_hour = 18;
    let stamp = |utc: &str| {
        chrono::DateTime::parse_from_rfc3339(utc)
            .unwrap()
            .timestamp_millis()
    };
    for (utc, expected) in [
        ("2026-11-17T22:59:59Z", false),
        ("2026-11-17T23:00:00Z", true),
        ("2026-11-18T09:59:59Z", true),
        ("2026-11-18T10:00:00Z", false),
    ] {
        assert_eq!(runtime::working(spec, stamp(utc)), expected, "{utc}");
    }
    spec.work_start_hour = 0;
    spec.work_end_hour = 0;
    assert!(!runtime::working(spec, stamp("2026-11-17T16:00:00Z")));
    // Inverted bounds are not an overnight schedule in the original camera contract.
    spec.work_start_hour = 22;
    spec.work_end_hour = 6;
    assert!(!runtime::working(spec, stamp("2026-11-18T15:00:00Z")));
    assert!(!runtime::working(spec, stamp("2026-11-18T17:00:00Z")));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn after_hours_only_personnel_zone_entries_are_saved_and_notified() {
    let root = tempfile::tempdir().unwrap();
    let spy = Arc::new(Spy::default());
    let (url, server) = server(spy.clone()).await;
    let mut cfg = config(root.path(), 3);
    // The replay generates missing PPE, a person inside the pole zone, and a
    // machine near the pole. Repeatable all-day/no-work settings avoid using
    // the wall clock to choose the policy under test.
    std::fs::write(
        cfg.models[0].replay.as_ref().unwrap(),
        serde_json::to_vec(&json!([[
            [0, 0, 10, 20, 0.9, 2],
            [0, 0, 10, 20, 0.9, 4],
            [95, 90, 105, 100, 0.9, 5],
            [95, 0, 105, 100, 0.9, 9],
            [90, 50, 110, 100, 0.9, 8]
        ]]))
        .unwrap(),
    )
    .unwrap();
    for (i, stream) in cfg.streams.iter_mut().enumerate() {
        stream.output = Default::default();
        stream.rules.detect_in_utility_pole_restricted_area = i != 2;
        stream.rules.detect_machinery_close_to_pole = true;
        if i != 0 {
            stream.work_end_hour = 0;
        }
    }
    cfg.max_frames = Some(1);
    cfg.delivery.enabled = true;
    cfg.delivery.upload_url = Some(format!("{url}/upload"));
    cfg.delivery.notification_url = Some(format!("{url}/notify"));
    cfg.validate().unwrap();

    let report = runtime::run(cfg, CancellationToken::new()).await.unwrap();
    assert_eq!(report.outbox["done"], 2);
    for camera in &report.cameras {
        assert_eq!(camera.processed, 1, "{}", camera.id);
        assert_eq!(camera.stored_events, u64::from(camera.id != "camera-2"));
    }
    let uploads = spy.upload_warnings.lock().unwrap();
    let notifications = spy.notification_warnings.lock().unwrap();
    assert_eq!(*uploads, *notifications);
    assert_eq!(uploads.len(), 2);
    let on_duty = &uploads["camera-0"];
    assert_eq!(on_duty["warning_no_hardhat"]["count"], 1);
    assert_eq!(on_duty["detect_machinery_close_to_pole"]["count"], 1);
    let off_duty = &uploads["camera-1"];
    assert_eq!(
        off_duty,
        &json!({"warning_people_in_utility_pole_controlled_area":{"count":1}})
    );
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn full_pipeline_32_cameras_store_notify_and_encode_without_drops() {
    let root = tempfile::tempdir().unwrap();
    let spy = Arc::new(Spy::default());
    let (url, server) = server(spy.clone()).await;
    let mut config = config(root.path(), 32);
    config.delivery.enabled = true;
    config.delivery.upload_url = Some(format!("{url}/upload"));
    config.delivery.notification_url = Some(format!("{url}/notify"));
    config.validate().unwrap();
    let report = tokio::time::timeout(
        Duration::from_secs(90),
        runtime::run(config, CancellationToken::new()),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(report.cameras.len(), 32);
    assert_eq!(report.outbox["done"], 128);
    assert!(
        !root.path().join("state/outbox.sqlite3").exists(),
        "Runtime created a SQLite outbox"
    );
    for camera in &report.cameras {
        assert_eq!(camera.captured, 4);
        assert_eq!(camera.processed, 4);
        assert_eq!(camera.skipped, 0);
        assert_eq!(camera.capture_rate_dropped, 0);
        assert_eq!(camera.encoded, 8);
        assert_eq!(camera.encoded_unique, 8);
        assert_eq!(camera.encoded_repeated, 0);
        assert_eq!(camera.failures, 0);
    }
    assert_eq!(spy.uploads.lock().unwrap().len(), 128);
    assert_eq!(spy.notified.lock().unwrap().len(), 128);
    for i in 0..32 {
        for kind in ["clean", "overlay"] {
            let output = tokio::process::Command::new("ffprobe")
                .args([
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_read_frames,width,height",
                    "-of",
                    "json",
                ])
                .arg(root.path().join(format!("{kind}-{i}.mp4")))
                .output()
                .await
                .unwrap();
            assert!(output.status.success());
            let value: Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(
                value["streams"][0]["nb_read_frames"], "4",
                "{kind}-{i}: {value}"
            );
            assert_eq!(value["streams"][0]["width"], 640);
        }
    }
    server.abort();
}

#[tokio::test]
async fn live_conversion_budget_drains_decode_and_honours_the_frame_limit() {
    use std::sync::atomic::Ordering;
    let root = tempfile::tempdir().unwrap();
    let mut cfg = config(root.path(), 1);
    cfg.capture.max_convert_fps = 8;
    let capture = construction_pipeline::capture::spawn(
        cfg.streams.remove(0),
        false,
        Some(16),
        cfg.capture,
        None,
        None,
        CancellationToken::new(),
    );
    let mut frames = capture.frames;
    let stop = CancellationToken::new();
    let (mut seen, mut received) = (0, 0);
    while let Some(frame) = frames.next(seen, &stop).await {
        assert!(frame.sequence > seen);
        seen = frame.sequence;
        received += 1;
    }
    capture.task.await.unwrap().unwrap();
    let converted = capture.stats.converted.load(Ordering::Relaxed);
    let dropped = capture.stats.rate_dropped.load(Ordering::Relaxed);
    assert_eq!(capture.produced.load(Ordering::Relaxed), 16);
    assert_eq!(converted + dropped, 16);
    // A slow test host may decode below the cap and legitimately drop none.
    // The simulated-clock unit test checks the exact admission count.
    assert!((1..=16).contains(&received));
}

#[tokio::test]
async fn automatic_cpu_budget_shrinks_after_a_camera_expires() {
    let root = tempfile::tempdir().unwrap();
    let mut cfg = config(root.path(), 3);
    cfg.cpu_workers = 2;
    cfg.cpu_workers_auto = Some(
        serde_json::from_value(json!({
            "cameras_per_worker": 2,
            "min_workers": 1,
            "reserve_cpus": 0,
        }))
        .unwrap(),
    );
    cfg.lossless = false;
    cfg.max_frames = None;
    cfg.record_detections = true;
    cfg.capture.max_convert_fps = 15;
    for stream in &mut cfg.streams {
        stream.output = Default::default();
        stream.work_end_hour = 0;
    }
    let expires = chrono::Utc::now() + chrono::Duration::seconds(5);
    cfg.streams[2].expires = Some(expires);
    cfg.validate().unwrap();

    let expected_initial = std::thread::available_parallelism().map_or(1, |n| n.get().min(2));
    let survivors = [
        cfg.state_dir.join("detections-camera-0.jsonl"),
        cfg.state_dir.join("detections-camera-1.jsonl"),
    ];
    let after_expiry = expires.timestamp_millis() + 500;
    let stop = CancellationToken::new();
    let stop_after_survivors_advance = async {
        loop {
            let advanced = survivors.iter().all(|trace| {
                std::fs::read_to_string(trace).is_ok_and(|records| {
                    records
                        .lines()
                        .rev()
                        .find_map(|line| {
                            let frame: Value = serde_json::from_str(line).ok()?;
                            frame["timestamp_ms"].as_i64()
                        })
                        .is_some_and(|stamp| stamp >= after_expiry)
                })
            });
            if advanced {
                stop.cancel();
                return;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    };
    let report = tokio::time::timeout(Duration::from_secs(20), async {
        let run = runtime::run(cfg, stop.clone());
        tokio::pin!(run);
        tokio::select! {
            report = &mut run => report,
            _ = stop_after_survivors_advance => run.await,
        }
    })
    .await
    .expect("Both surviving streams must keep analyzing after expiry")
    .unwrap();
    assert!(report.cpu_workers.automatic);
    assert_eq!(report.cpu_workers.initial, expected_initial);
    assert_eq!(report.cpu_workers.limit, expected_initial);
    assert_eq!(report.cpu_workers.current, 1);
    assert_eq!(
        report.cpu_workers.adjustments,
        if expected_initial == 2 { 1 } else { 0 }
    );
    assert_eq!(report.cameras.len(), 3, "No camera should restart");
    assert!(report.cameras.iter().all(|camera| camera.processed > 0));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn notification_retry_preserves_upload_receipt_across_dispatchers() {
    let root = tempfile::tempdir().unwrap();
    let spy = Arc::new(Spy::default());
    spy.retry_first_notification.store(true, Ordering::SeqCst);
    let (url, server) = server(spy.clone()).await;
    let mut config = config(root.path(), 1);
    config.streams[0].enabled = false;
    config.delivery.enabled = true;
    config.delivery.upload_url = Some(format!("{url}/upload"));
    config.delivery.notification_url = Some(format!("{url}/notify"));
    let event = Event {
        event_id: uuid::Uuid::new_v4().to_string(),
        site_id: 1,
        site: "fixture".into(),
        stream_name: "fixture".into(),
        detection_time: "2026-11-17T00:00:00Z".into(),
        model_id: "fixture".into(),
        model_version: "v1".into(),
        warnings: Default::default(),
        detections: vec![],
        cone_polygon: Default::default(),
        pole_polygon: Default::default(),
    };
    let mut image = vec![];
    image::codecs::jpeg::JpegEncoder::new(&mut image)
        .encode(&vec![0; 12], 2, 2, image::ExtendedColorType::Rgb8)
        .unwrap();
    let store = Outbox::open(&config).await.unwrap();
    store.put(&event, &image).await.unwrap();
    let mut pending = store
        .claim(chrono::Utc::now().timestamp())
        .await
        .unwrap()
        .unwrap();
    let first = construction_pipeline::delivery::Transport::new(config.delivery.clone()).unwrap();
    first.deliver(&store, &mut pending).await.unwrap();
    drop(first);
    let report = runtime::run(config, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(report.outbox["done"], 1);
    assert_eq!(spy.upload_requests.load(Ordering::SeqCst), 1);
    assert_eq!(spy.notification_requests.load(Ordering::SeqCst), 2);
    server.abort();
}

#[test]
fn replay_cannot_send_to_real_endpoints_and_outputs_cannot_collide() {
    let root = tempfile::tempdir().unwrap();
    let mut cfg = config(root.path(), 1);
    cfg.delivery.enabled = true;
    cfg.delivery.upload_url = Some("https://example.com/upload".into());
    assert!(cfg.validate().is_err());
    cfg.delivery.enabled = false;
    cfg.streams[0].output.clean = cfg.streams[0].output.annotated.clone();
    assert!(cfg.validate().is_err());
}

#[test]
fn gpu_overlap_requires_async_and_respects_global_admission() {
    let root = tempfile::tempdir().unwrap();
    let mut cfg = config(root.path(), 1);
    cfg.gpu.exact_resize = true;
    assert!(cfg.validate().is_err());
    cfg.gpu.asynchronous = true;
    assert!(cfg.validate().is_ok());
    cfg.gpu.exact_resize = false;
    cfg.gpu.asynchronous = false;
    cfg.gpu.pipeline_depth = 2;
    assert!(cfg.validate().is_err());
    cfg.gpu.asynchronous = true;
    assert!(cfg.validate().is_ok());
    cfg.gpu.inference_slots = 1;
    assert!(cfg.validate().is_err());
    cfg.gpu.pipeline_depth = 1;
    assert!(cfg.validate().is_ok());
    cfg.preprocess_workers = 0;
    assert!(cfg.validate().is_err());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancellation_joins_native_media_owners_without_child_processes() {
    let root = tempfile::tempdir().unwrap();
    let mut config = config(root.path(), 1);
    config.max_frames = None;
    config.lossless = true;
    config.record_detections = true;
    let trace = config.state_dir.join("detections-camera-0.jsonl");
    let source = root.path().join("unique-source.mp4");
    std::fs::copy(&config.streams[0].source, &source).unwrap();
    config.streams[0].source = source.to_str().unwrap().into();
    let stop = CancellationToken::new();
    let cancel = stop.clone();
    let run = tokio::spawn(runtime::run(config, stop));
    let mut started = false;
    for _ in 0..500 {
        started = std::fs::metadata(&trace).is_ok_and(|m| m.len() > 0);
        if started {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(
        started,
        "Capture and analysis must start before cancellation"
    );
    let children: Vec<_> = std::fs::read_dir("/proc")
        .unwrap()
        .flatten()
        .filter_map(|entry| {
            let pid = entry.file_name().to_str()?.parse::<u32>().ok()?;
            let cmd = std::fs::read(entry.path().join("cmdline")).ok()?;
            if cmd.starts_with(b"ffmpeg\0")
                && String::from_utf8_lossy(&cmd).contains(root.path().to_str().unwrap())
            {
                Some(pid)
            } else {
                None
            }
        })
        .collect();
    assert!(
        children.is_empty(),
        "Native media must not start FFmpeg processes"
    );
    cancel.cancel();
    let report = tokio::time::timeout(Duration::from_secs(10), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(report.elapsed_seconds < 10.);
    assert!(report.cameras[0].converted > 0);
    assert!(report.cameras[0].encoded > 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn oauth_refreshes_once_after_401_without_global_environment_changes() {
    let root = tempfile::tempdir().unwrap();
    let spy = Arc::new(Spy::default());
    spy.reject_first_upload.store(true, Ordering::SeqCst);
    let (url, server) = server(spy.clone()).await;
    let mut cfg = config(root.path(), 1);
    cfg.max_frames = Some(1);
    cfg.delivery.enabled = true;
    cfg.delivery.upload_url = Some(format!("{url}/upload"));
    cfg.delivery.notification_url = Some(format!("{url}/notify"));
    cfg.delivery.token_url = Some(format!("{url}/token"));
    cfg.delivery.client_id_env = Some("CHD_FIXTURE_CLIENT".into());
    cfg.delivery.client_secret_env = Some("CHD_FIXTURE_SECRET".into());
    cfg.validate().unwrap();
    let path = root.path().join("config.json");
    std::fs::write(&path, serde_json::to_vec(&cfg).unwrap()).unwrap();
    let output = tokio::process::Command::new(env!("CARGO_BIN_EXE_construction-pipeline"))
        .args(["run", "--config"])
        .arg(&path)
        .env("CHD_FIXTURE_CLIENT", "fixture-client")
        .env("CHD_FIXTURE_SECRET", "fixture-secret")
        .output()
        .await
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        spy.tokens.load(Ordering::SeqCst),
        2,
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert_eq!(spy.upload_requests.load(Ordering::SeqCst), 2);
    assert_eq!(spy.notification_requests.load(Ordering::SeqCst), 1);
    assert!(!String::from_utf8_lossy(&output.stderr).contains("fixture-secret"));
    server.abort();
}

#[tokio::test]
async fn inactive_lossless_live_output_joins_when_its_sender_closes() {
    use construction_pipeline::{demand::Gate, publish, render::Text};
    let root = tempfile::tempdir().unwrap();
    let config = Arc::new(config(root.path(), 1));
    let text = Text::new(&config.fonts).unwrap();
    let publisher = publish::spawn_gated(
        config,
        "rtsp://127.0.0.1:1/unused".into(),
        640,
        352,
        15,
        false,
        text.clone(),
        Gate::new(false),
        CancellationToken::new(),
    );
    // Enter the inactive await before closing; an already closed receiver
    // would exercise only the initial check and miss this shutdown regression.
    tokio::task::yield_now().await;
    drop(publisher.sender);
    tokio::time::timeout(Duration::from_secs(1), publisher.task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(!text.initialized());
    assert_eq!(publisher.encoded.load(Ordering::Relaxed), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn full_and_preview_outputs_keep_frames_and_language_paths() {
    let root = tempfile::tempdir().unwrap();
    let mut cfg = config(root.path(), 1);
    cfg.capture.host_yuv = true;
    let stream = &mut cfg.streams[0];
    stream.languages = vec!["en".into(), "zh-TW".into()];
    stream.output.annotated = Some(
        root.path()
            .join("overlay_{language_b64}.mp4")
            .to_string_lossy()
            .into(),
    );
    stream.output.preview = Some(
        serde_json::from_value(json!({
            "clean":root.path().join("preview.mp4"),
            "annotated":root.path().join("preview_{language_b64}.mp4"),
            "width":320,"height":176,"fps":15
        }))
        .unwrap(),
    );
    cfg.validate().unwrap();
    let report = runtime::run(cfg, CancellationToken::new()).await.unwrap();
    assert_eq!(report.cameras[0].encoded, 24);
    assert_eq!(report.cameras[0].encoder_annotated_host_frames, 16);
    assert_eq!(report.cameras[0].host_materialized, 0);
    for (name, width, height) in [
        ("clean-0.mp4", 640, 352),
        ("overlay_ZW4.mp4", 640, 352),
        ("overlay_emgtVFc.mp4", 640, 352),
        ("preview.mp4", 320, 176),
        ("preview_ZW4.mp4", 320, 176),
        ("preview_emgtVFc.mp4", 320, 176),
    ] {
        let output = tokio::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_read_frames",
                "-of",
                "json",
            ])
            .arg(root.path().join(name))
            .output()
            .await
            .unwrap();
        assert!(output.status.success(), "{name}");
        let data: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(data["streams"][0]["width"], width, "{name}");
        assert_eq!(data["streams"][0]["height"], height, "{name}");
        assert_eq!(data["streams"][0]["nb_read_frames"], "4", "{name}");
    }
    let read_frame = |name| {
        let output = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-threads", "1", "-i"])
            .arg(root.path().join(name))
            .args([
                "-frames:v",
                "1",
                "-pix_fmt",
                "yuv420p",
                "-threads",
                "1",
                "-f",
                "rawvideo",
                "pipe:1",
            ])
            .output()
            .unwrap();
        assert!(output.status.success());
        output.stdout
    };
    let preview = read_frame("preview.mp4");
    let english = read_frame("preview_ZW4.mp4");
    let taiwanese = read_frame("preview_emgtVFc.mp4");
    assert_eq!(preview.len(), 320 * 176 * 3 / 2);
    assert_ne!(preview, english);
    assert_ne!(english, taiwanese);
}

#[test]
fn scaled_encoder_counts_muxed_packets_and_rejects_invalid_input() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("scaled.mp4");
    let mut encoder =
        construction_media::Encoder::open(path.to_str().unwrap(), 320, 176, 15, None).unwrap();
    assert_eq!(encoder.packets_written(), 0);
    assert!(encoder.write_scaled(&[0; 64], 640, 352).is_err());
    assert_eq!(encoder.packets_written(), 0);
    encoder
        .write_scaled(&vec![30; 640 * 352 * 3 + 64], 640, 352)
        .unwrap();
    assert_eq!(encoder.packets_written(), 1);
    encoder.finish().unwrap();
    assert_eq!(encoder.packets_written(), 1);
    let out = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=width,height,nb_read_frames",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .unwrap();
    assert!(out.status.success());
    let data: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(data["streams"][0]["width"], 320);
    assert_eq!(data["streams"][0]["height"], 176);
    assert_eq!(data["streams"][0]["nb_read_frames"], "1");
}

#[test]
fn scaled_encoder_repeats_the_same_pixels_at_the_next_timestamp() {
    let root = tempfile::tempdir().unwrap();
    let first = vec![30; 640 * 352 * 3 + 64];
    let second = vec![220; first.len()];
    let encode = |name: &str, use_cache: bool| {
        let path = root.path().join(name);
        let mut encoder =
            construction_media::Encoder::open(path.to_str().unwrap(), 320, 176, 15, None).unwrap();
        assert!(encoder.repeat_frame().is_err());
        encoder.write_scaled(&first, 640, 352).unwrap();
        if use_cache {
            encoder.repeat_frame().unwrap();
        } else {
            encoder.write_scaled(&first, 640, 352).unwrap();
        }
        encoder.write_scaled(&second, 640, 352).unwrap();
        encoder.finish().unwrap();
        assert_eq!(encoder.packets_written(), 3);
        assert!(encoder.repeat_frame().is_err());
        let output = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(path)
            .args(["-f", "framehash", "-"])
            .output()
            .unwrap();
        assert!(output.status.success());
        output.stdout
    };

    // Framehash includes decoded pixel hashes and PTS, including the change
    // after the repeated image.
    assert_eq!(encode("reference.mp4", false), encode("repeat.mp4", true));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn live_http_source_recovers_after_unavailable_response() {
    use axum::{body::Bytes, routing::get};

    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/videos/test.mp4");
    let video = tokio::process::Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args([
            "-t",
            "1",
            "-an",
            "-c:v",
            "mpeg2video",
            "-threads",
            "1",
            "-f",
            "mpegts",
            "pipe:1",
        ])
        .output()
        .await
        .unwrap();
    assert!(video.status.success());
    let bytes = Bytes::from(video.stdout);
    let calls = Arc::new(AtomicUsize::new(0));
    let server_calls = calls.clone();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = Router::new().route(
        "/camera.ts",
        get(move || {
            let video = bytes.clone();
            let call = server_calls.fetch_add(1, Ordering::Relaxed);
            async move {
                if call == 0 {
                    (StatusCode::SERVICE_UNAVAILABLE, Bytes::new())
                } else {
                    (StatusCode::OK, video)
                }
            }
        }),
    );
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let root = tempfile::tempdir().unwrap();
    let mut cfg = config(root.path(), 1);
    let mut spec = cfg.streams.remove(0);
    spec.source = format!("http://{address}/camera.ts");
    let stop = CancellationToken::new();
    let mut capture = construction_pipeline::capture::spawn(
        spec,
        false,
        Some(3),
        cfg.capture,
        None,
        None,
        stop.clone(),
    );
    let received = tokio::time::timeout(Duration::from_secs(10), capture.frames.next(0, &stop))
        .await
        .unwrap()
        .expect("HTTP source should reconnect and recover");
    assert!(received.generation > 1);
    assert!(calls.load(Ordering::Relaxed) >= 2);
    stop.cancel();
    capture.task.await.unwrap().unwrap();
    server.abort();
}
