//! Optional isolated-service tests. Never point these at a deployed application.
mod support;
use construction_pipeline::{
    catalog::Catalog, config::Config, hazard::HazardResult, metadata::Sink, runtime,
};
use serde_json::{Value, json};
use sqlx::{Connection, Executor};
use std::{path::Path, time::Duration};
use tokio_util::sync::CancellationToken;

#[tokio::test]
#[ignore = "Requires isolated CHD_RUST_TEST_REDIS_URL"]
async fn shared_demand_keeps_camera_owners_and_reload_independent() {
    use construction_pipeline::demand::{Demand, Gate};
    use redis::AsyncCommands;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    let url_env = "CHD_RUST_TEST_REDIS_URL";
    let url = std::env::var(url_env).unwrap();
    let host = reqwest::Url::parse(&url).unwrap();
    assert!(matches!(host.host_str(), Some("127.0.0.1" | "localhost")));
    let mut redis = redis::Client::open(url)
        .unwrap()
        .get_multiplexed_async_connection()
        .await
        .unwrap();
    let prefix = format!("demand-test:{}:", uuid::Uuid::new_v4());
    let config = serde_json::from_value(json!({"url_env":url_env,"prefix":prefix})).unwrap();
    let stop = CancellationToken::new();
    let (demand, worker) = Demand::spawn(config, stop.clone());
    let mut cameras = Vec::new();
    for id in 0..32 {
        let gate = Gate::new(false);
        let ready = Arc::new(AtomicBool::new(false));
        let registration = demand.register(
            vec![(format!("demand:{id}"), gate.clone())],
            vec![(format!("ready:{id}"), ready.clone())],
        );
        cameras.push((registration, gate, ready));
    }
    let demand_a = format!("{prefix}demand:31");
    let demand_b = format!("{prefix}demand:30");
    let ready_a = format!("{prefix}ready:31");
    let ready_b = format!("{prefix}ready:30");
    let _: () = redis.set_ex(&demand_a, "1", 12).await.unwrap();
    let _: () = redis.set_ex(&demand_b, "1", 12).await.unwrap();
    let (a, old_gate, a_flag) = cameras.pop().unwrap();
    let (_, b_gate, b_flag) = cameras.last().unwrap();
    a_flag.store(true, Ordering::Release);
    b_flag.store(true, Ordering::Release);
    let (a_owner, b_owner) = tokio::time::timeout(Duration::from_secs(4), async {
        loop {
            let first = redis.get::<_, Option<String>>(&ready_a).await.unwrap();
            let second = redis.get::<_, Option<String>>(&ready_b).await.unwrap();
            if let (Some(first), Some(second)) = (first, second) {
                break (first, second);
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .unwrap();
    assert_ne!(a_owner, b_owner);
    assert!(old_gate.active() && b_gate.active());
    assert!(cameras[..30].iter().all(|(_, gate, _)| !gate.active()));

    drop(a);
    assert!(!old_gate.active());
    let new_gate = Gate::new(false);
    let _replacement = demand.register(
        vec![("demand:31".into(), new_gate.clone())],
        vec![("ready:31".into(), Arc::new(AtomicBool::new(true)))],
    );
    let replacement_owner = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let current = redis.get::<_, Option<String>>(&ready_a).await.unwrap();
            if let Some(owner) = current
                && owner != a_owner
            {
                break owner;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .unwrap();
    assert_ne!(replacement_owner, b_owner);
    assert!(new_gate.active() && b_gate.active() && !old_gate.active());

    let _: () = redis.del(&demand_b).await.unwrap();
    b_flag.store(false, Ordering::Release);
    tokio::time::timeout(Duration::from_secs(4), async {
        while b_gate.active() || redis.exists::<_, bool>(&ready_b).await.unwrap() {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .unwrap();
    assert!(new_gate.active());
    stop.cancel();
    worker.await.unwrap();
    assert!(!new_gate.active());
}

fn fixture(root: &Path) -> Config {
    support::check_postgres_if_configured();
    let video = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/videos/test.mp4")
        .canonicalize()
        .unwrap();
    let replay = root.join("replay.json");
    std::fs::write(&replay, "[[[0,0,10,20,0.9,2]]]").unwrap();
    serde_json::from_value(json!({"state_dir":root.join("state"),"outbox":{"url_env":"CHD_RUST_TEST_DATABASE_URL"},"experimental_replay":true,"lossless":true,"max_frames":4,
        "models":[{"key":"fixture","artifact":"unused","sha256":"0".repeat(64),"version":"fixture","replay":replay}],
        "streams":[{"id":"1","site_id":1,"site":"fixture","name":"fixture","source":video,"model":"fixture","width":640,"height":352,"work_start_hour":0,"work_end_hour":24}]})).unwrap()
}

#[tokio::test]
#[ignore = "Requires CHD_RUST_TEST_DATABASE_URL naming chd_rust_pipeline_test"]
async fn postgres_catalog_refresh_expiry_and_manifest_fence() {
    let url = std::env::var("CHD_RUST_TEST_DATABASE_URL").unwrap();
    let parsed = reqwest::Url::parse(&url).unwrap();
    assert_eq!(parsed.path(), "/chd_rust_pipeline_test");
    assert!(matches!(parsed.host_str(), Some("127.0.0.1" | "localhost")));
    let mut db = sqlx::PgConnection::connect(&url).await.unwrap();
    db.execute("DROP TABLE IF EXISTS stream_configs,detection_model_catalog,sites;
        CREATE TABLE sites(id bigint PRIMARY KEY,name text NOT NULL);
        CREATE TABLE detection_model_catalog(model_key text PRIMARY KEY,definition jsonb NOT NULL);
        CREATE TABLE stream_configs(id bigint PRIMARY KEY,site_id bigint,stream_name text,video_url text,model_key text,
        recognition_enabled bool,expire_date timestamptz,work_start_hour int,work_end_hour int,
        detect_no_safety_vest_or_helmet bool,detect_near_machinery_or_vehicle bool,detect_in_restricted_area bool,
        detect_in_utility_pole_restricted_area bool,detect_machinery_close_to_pole bool);
        INSERT INTO sites VALUES(1,'fixture');").await.unwrap();
    let root = tempfile::tempdir().unwrap();
    let mut config = fixture(root.path());
    let manifest = root.path().join("manifest.json");
    std::fs::write(&manifest,serde_json::to_vec(&json!({"format":1,"models":{"fixture":{"artifact":"unused.engine","sha256":"1".repeat(64),"source_sha256":"0".repeat(64),"version":"fixture"}}})).unwrap()).unwrap();
    let definition = json!({"display_name":"fixture","version":"fixture","artifact":"unused.pt","sha256":"0".repeat(64),"capabilities":["stream"],"classes":[{"id":2,"code":"no_helmet","display_name":"未戴安全帽"}]});
    sqlx::query("INSERT INTO detection_model_catalog VALUES('fixture',$1)")
        .bind(definition)
        .execute(&mut db)
        .await
        .unwrap();
    sqlx::query(
        "INSERT INTO stream_configs VALUES
         (1,1,'fixture',$1,'fixture',true,NULL,0,24,true,true,true,false,false),
         (2,1,'second-shift',$1,'fixture',true,NULL,12,20,true,true,true,false,false)",
    )
    .bind(&config.streams[0].source)
    .execute(&mut db)
    .await
    .unwrap();
    config.database=Some(serde_json::from_value(json!({"url_env":"CHD_RUST_TEST_DATABASE_URL","engine_manifest":manifest,"width":640,"height":352})).unwrap());
    config.database.as_mut().unwrap().media = Some(
        serde_json::from_value(json!({
            "publish_base":"rtsp://127.0.0.1:8554","languages":["en","zh-TW"],
            "on_demand":false,"preview":{"width":640,"height":360,"fps":15}
        }))
        .unwrap(),
    );
    let mut catalog = Catalog::new(&config).await.unwrap();
    let loaded = catalog.load(&config).await.unwrap();
    assert_eq!(loaded.streams.len(), 2);
    assert_eq!(
        loaded.streams[0].output.clean.as_deref(),
        Some("rtsp://127.0.0.1:8554/hazard_Zml4dHVyZQ_Zml4dHVyZQ")
    );
    assert_eq!(
        loaded.streams[0].output.preview.as_ref().unwrap().height,
        360
    );
    db.execute("UPDATE stream_configs SET stream_name='renamed' WHERE id=1")
        .await
        .unwrap();
    let renamed = catalog.load(&config).await.unwrap();
    assert!(
        renamed.streams[0]
            .output
            .clean
            .as_ref()
            .unwrap()
            .ends_with("_cmVuYW1lZA")
    );

    assert_eq!(loaded.models[0].class_metadata[0]["color"], "#F44336");
    assert_eq!(
        loaded.models[0].class_metadata[0]["display_name"],
        "未戴安全帽"
    );
    assert!(
        loaded.models[0]
            .labels
            .get(&2)
            .is_none_or(|labels| !labels.contains_key("en"))
    );
    assert!(loaded.models[0].catalog_revision.is_some());
    db.execute(
        "UPDATE stream_configs SET work_start_hour=8,work_end_hour=17 WHERE id=1;
         NOTIFY stream_config_changed",
    )
    .await
    .unwrap();
    tokio::time::timeout(Duration::from_secs(2), catalog.changed(30))
        .await
        .unwrap();
    let refreshed = catalog.load(&config).await.unwrap();
    assert_eq!(
        refreshed
            .streams
            .iter()
            .map(|s| (s.work_start_hour, s.work_end_hour))
            .collect::<Vec<_>>(),
        vec![(8, 17), (12, 20)],
    );
    let taipei_morning = chrono::DateTime::parse_from_rfc3339("2026-11-18T08:00:00+08:00")
        .unwrap()
        .timestamp_millis();
    assert!(runtime::working(&refreshed.streams[0], taipei_morning));
    assert!(!runtime::working(&refreshed.streams[1], taipei_morning));
    sqlx::query("UPDATE detection_model_catalog SET definition=jsonb_set(definition,'{version}','\"changed\"')").execute(&mut db).await.unwrap();
    assert!(catalog.load(&config).await.is_err());
    sqlx::query("UPDATE stream_configs SET expire_date=NOW()-INTERVAL '1 second'")
        .execute(&mut db)
        .await
        .unwrap();
    assert!(catalog.load(&config).await.unwrap().streams.is_empty());
}

#[tokio::test]
#[ignore = "Requires isolated CHD_RUST_TEST_REDIS_URL"]
async fn redis_generation_and_owner_fences_preserve_new_owner() {
    let root = tempfile::tempdir().unwrap();
    let mut cfg = fixture(root.path());
    let prefix = format!("rust-test:{}:", uuid::Uuid::new_v4());
    cfg.redis = Some(
        serde_json::from_value(json!({"url_env":"CHD_RUST_TEST_REDIS_URL","prefix":prefix}))
            .unwrap(),
    );
    let url = std::env::var("CHD_RUST_TEST_REDIS_URL").unwrap();
    let parsed = reqwest::Url::parse(&url).unwrap();
    assert!(matches!(parsed.host_str(), Some("127.0.0.1" | "localhost")));
    let mut connection = redis::Client::open(url)
        .unwrap()
        .get_multiplexed_async_connection()
        .await
        .unwrap();
    let spec = &cfg.streams[0];
    let mut model = cfg.models[0].clone();
    model.catalog_revision = Some("revision".into());
    model
        .labels
        .insert(12, [("en".into(), "fixture".into())].into());
    let mut first = Sink::new(cfg.redis.as_ref().unwrap(), spec).await.unwrap();
    let cone = vec![[4., 4.], [6., 4.], [6., 6.], [4., 4.]];
    let warning = HazardResult {
        cone_polys: vec![cone.clone()].into(),
        ..HazardResult::default()
    };
    let now = chrono::Utc::now().timestamp_millis();
    assert!(
        first
            .publish(spec, 1, 1, now, &[], &warning, true, &model)
            .await
            .unwrap()
    );
    let mut second = Sink::new(cfg.redis.as_ref().unwrap(), spec).await.unwrap();
    assert!(
        !second
            .publish(spec, 2, 1, now, &[], &warning, true, &model)
            .await
            .unwrap()
    );
    let _: i64 = redis::cmd("DEL")
        .arg(format!("{prefix}rust_owner:1:1"))
        .query_async(&mut connection)
        .await
        .unwrap();
    assert!(
        second
            .publish(spec, 2, 1, now, &[], &warning, true, &model)
            .await
            .unwrap()
    );
    first.close().await.unwrap();
    let latest: String = redis::cmd("GET")
        .arg(format!("{prefix}rust_latest:1:1"))
        .query_async(&mut connection)
        .await
        .unwrap();
    let latest: Value = serde_json::from_str(&latest).unwrap();
    assert_eq!(latest["sequence"], 2);
    assert_eq!(latest["labels"]["12"]["en"], "fixture");
    assert_eq!(latest["cone_polys"], json!([cone]));
    assert_eq!(latest["pole_polys"], json!([]));
    assert_eq!(latest.get("source_pts"), Some(&Value::Null));
    let _: () = redis::cmd("SET")
        .arg(format!("{prefix}stream_metadata_generation:Zml4dHVyZQ=="))
        .arg(1)
        .query_async(&mut connection)
        .await
        .unwrap();
    assert!(
        !second
            .publish(spec, 3, 1, now, &[], &warning, true, &model)
            .await
            .unwrap()
    );
    second.close().await.unwrap();
    let keys: Vec<String> = redis::cmd("KEYS")
        .arg(format!("{prefix}*"))
        .query_async(&mut connection)
        .await
        .unwrap();
    if !keys.is_empty() {
        let _: i64 = redis::cmd("DEL")
            .arg(keys)
            .query_async(&mut connection)
            .await
            .unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "Requires loopback CHD_RUST_TEST_RTSP_URL with a running publisher"]
async fn thirty_two_rtsp_sessions_decode_infer_and_store() {
    for rust_rtsp in [false, true] {
        for host_yuv in [false, true] {
            run_rtsp_sessions(rust_rtsp, host_yuv).await;
        }
    }
}
async fn run_rtsp_sessions(rust_rtsp: bool, host_yuv: bool) {
    let root = tempfile::tempdir().unwrap();
    let mut cfg = fixture(root.path());
    cfg.capture.rust_rtsp = rust_rtsp;
    cfg.capture.host_yuv = host_yuv;
    let url = std::env::var("CHD_RUST_TEST_RTSP_URL").unwrap();
    let parsed = reqwest::Url::parse(&url).unwrap();
    assert!(matches!(parsed.host_str(), Some("127.0.0.1" | "localhost")));
    let template = cfg.streams[0].clone();
    cfg.streams = (0..32)
        .map(|i| {
            let mut spec = template.clone();
            spec.id = format!("camera-{i}");
            spec.source = url.clone();
            spec.evidence_cooldown_seconds = 0;
            spec
        })
        .collect();
    cfg.validate().unwrap();
    let report = tokio::time::timeout(
        Duration::from_secs(60),
        runtime::run(cfg, CancellationToken::new()),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(report.cameras.len(), 32);
    assert_eq!(report.outbox["pending"], 128);
    for cam in report.cameras {
        assert_eq!(cam.processed, 4);
        assert_eq!(cam.skipped, 0);
        assert_eq!(cam.failures, 0);
        if rust_rtsp {
            assert!(cam.rust_rtsp_frames >= 4);
        } else {
            assert_eq!(cam.rust_rtsp_frames, 0);
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "Requires loopback CHD_RUST_TEST_RTSP_URL and a running publisher"]
async fn rust_receiver_drop_closes_tcp_connections() {
    use construction_pipeline::rtsp::{Source, Stats};
    use std::sync::Arc;
    fn sockets() -> usize {
        std::fs::read_dir("/proc/self/fd")
            .unwrap()
            .filter_map(Result::ok)
            .filter_map(|e| std::fs::read_link(e.path()).ok())
            .filter(|p| p.to_string_lossy().starts_with("socket:["))
            .count()
    }
    let url = std::env::var("CHD_RUST_TEST_RTSP_URL").unwrap();
    let parsed = reqwest::Url::parse(&url).unwrap();
    assert!(matches!(parsed.host_str(), Some("127.0.0.1" | "localhost")));
    let baseline = sockets();
    let handle = tokio::runtime::Handle::current();
    for _ in 0..3 {
        let address = url.clone();
        let runtime = handle.clone();
        tokio::task::spawn_blocking(move || {
            let mut source = Source::open(
                &runtime,
                true,
                &address,
                -1,
                CancellationToken::new(),
                Duration::from_secs(5),
                Arc::default(),
                Arc::new(Stats::default()),
            )
            .unwrap();
            assert!(source.next_frame().unwrap().is_some());
            drop(source);
        })
        .await
        .unwrap();
    }
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    while sockets() > baseline && std::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(
        sockets() <= baseline,
        "Cancelled Rust RTSP readers leaked sockets"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "Requires isolated Redis, RTSP and CHD_RUST_TEST_MEDIA_API_URL"]
async fn demand_passthrough_and_browser_metadata_preserve_inference() {
    use redis::AsyncCommands;
    let url = std::env::var("CHD_RUST_TEST_RTSP_URL").unwrap();
    let api = std::env::var("CHD_RUST_TEST_MEDIA_API_URL").unwrap();
    let parsed = reqwest::Url::parse(&url).unwrap();
    assert!(matches!(parsed.host_str(), Some("127.0.0.1" | "localhost")));
    assert!(matches!(
        reqwest::Url::parse(&api).unwrap().host_str(),
        Some("127.0.0.1" | "localhost")
    ));
    let probe = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-rtsp_transport",
            "tcp",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            "-of",
            "json",
            &url,
        ])
        .output()
        .unwrap();
    assert!(probe.status.success());
    let details: Value = serde_json::from_slice(&probe.stdout).unwrap();
    let video = &details["streams"][0];
    let rate: Vec<f64> = video["r_frame_rate"]
        .as_str()
        .unwrap()
        .split('/')
        .map(|v| v.parse().unwrap())
        .collect();
    let root = tempfile::tempdir().unwrap();
    let mut config = fixture(root.path());
    config.max_frames = None;
    config.lossless = false;
    config.capture.host_yuv = true;
    config.metadata_async = true;
    let prefix = format!("demand-test:{}:", uuid::Uuid::new_v4());
    config.redis = Some(
        serde_json::from_value(json!({"url_env":"CHD_RUST_TEST_REDIS_URL","prefix":prefix}))
            .unwrap(),
    );
    let name = format!("hazard_test_{}", uuid::Uuid::new_v4().simple());
    let mut target = parsed.clone();
    target.set_path(&format!("/{name}"));
    let spec = &mut config.streams[0];
    spec.source = url;
    spec.width = video["width"].as_u64().unwrap() as u32;
    spec.height = video["height"].as_u64().unwrap() as u32;
    spec.fps = (rate[0] / rate[1]).round() as u32;
    spec.output=serde_json::from_value(json!({"clean":target.as_str(),"annotated":format!("{target}_annotated_ZW4"),"clean_passthrough":true,"client_overlay":true,"on_demand":true})).unwrap();
    let mut db = redis::Client::open(std::env::var("CHD_RUST_TEST_REDIS_URL").unwrap())
        .unwrap()
        .get_multiplexed_async_connection()
        .await
        .unwrap();
    let latest = format!("{prefix}rust_latest:1:1");
    let demand = format!("{prefix}media_clean_demand:{name}");
    let stop = CancellationToken::new();
    let mut fallback_config = config.clone();
    let task = tokio::spawn(runtime::run(config, stop.clone()));
    let client = reqwest::Client::new();
    let ready = async |name: &str| -> bool {
        client
            .get(format!("{api}/v3/paths/get/{name}"))
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap()["ready"]
            .as_bool()
            .unwrap_or(false)
    };
    tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            if db.exists::<_, bool>(&latest).await.unwrap() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .unwrap();
    fn owners() -> usize {
        std::fs::read_dir("/proc/self/task")
            .unwrap()
            .flatten()
            .filter(|entry| {
                std::fs::read_to_string(entry.path().join("comm"))
                    .is_ok_and(|name| matches!(name.trim(), "encode" | "packet-remux"))
            })
            .count()
    }
    assert!(!ready(&name).await, "No lease must not start a publisher");
    assert_eq!(owners(), 0, "Unwatched output created native owner threads");
    let before: Value = serde_json::from_str(&db.get::<_, String>(&latest).await.unwrap()).unwrap();
    let _: () = db.set_ex(&demand, "1", 15).await.unwrap();
    tokio::time::timeout(Duration::from_secs(10), async {
        while !ready(&name).await {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .unwrap();
    // The passthrough shares capture's input session; it must not add a second
    // camera reader. No other test clients are running in this serial suite.
    let source = parsed.path().trim_start_matches('/');
    let status: Value = client
        .get(format!("{api}/v3/paths/get/{source}"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(status["readers"].as_array().unwrap().len(), 1);
    let _: () = db.expire(&demand, 1).await.unwrap();
    tokio::time::timeout(Duration::from_secs(6), async {
        while ready(&name).await {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .unwrap();
    tokio::time::timeout(Duration::from_secs(3), async {
        while owners() != 0 {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    let after: Value = serde_json::from_str(&db.get::<_, String>(&latest).await.unwrap()).unwrap();
    assert!(after["sequence"].as_u64() > before["sequence"].as_u64());
    assert!(after["source_pts"].is_number());
    assert_eq!(after["frame_width"], video["width"]);
    assert!(after["owner"].is_string());
    use base64::Engine;
    let site = base64::engine::general_purpose::URL_SAFE.encode("fixture");
    let frames: Vec<(String, std::collections::HashMap<String, String>)> = redis::cmd("XREVRANGE")
        .arg(format!("{prefix}stream_metadata:{site}|{site}"))
        .arg("+")
        .arg("-")
        .arg("COUNT")
        .arg(1)
        .query_async(&mut db)
        .await
        .unwrap();
    let overlay: Value = serde_json::from_str(&frames[0].1["overlay"]).unwrap();
    assert_eq!(overlay["owner"], after["owner"]);
    assert_eq!(overlay["frame_width"], video["width"]);
    assert!(overlay["track_data"].is_array());
    println!("rust_client_overlay_fixture={overlay}");

    let _: () = db.set_ex(&demand, "1", 10).await.unwrap();
    tokio::time::timeout(Duration::from_secs(10), async {
        while !ready(&name).await {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .unwrap();
    stop.cancel();
    let report = tokio::time::timeout(Duration::from_secs(10), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let cam = &report.cameras[0];
    assert!(cam.processed > 10);
    assert!(
        cam.relayed_packets > 0,
        "{}",
        serde_json::to_string(cam).unwrap()
    );
    assert_eq!(cam.encoded, 0);
    assert_eq!(cam.rendered, 0);
    assert!(
        !cam.fonts_initialized,
        "Client overlay eagerly initialized fonts"
    );
    assert_eq!(cam.metadata_errors, 0);
    assert!(cam.metadata_sent > 0);
    assert_eq!(cam.relay_errors, 0);
    assert_eq!(cam.relay_fallback, 0);
    assert!(cam.host_materialized < cam.processed);
    fallback_config.max_frames = Some(12);
    let spec = &mut fallback_config.streams[0];
    spec.width /= 2;
    spec.height /= 2;
    target.set_path(&format!("/{name}-fallback"));
    spec.output =
        serde_json::from_value(json!({"clean":target.as_str(),"clean_passthrough":true})).unwrap();
    let mut recording_config = fallback_config.clone();
    let fallback = tokio::time::timeout(
        Duration::from_secs(15),
        runtime::run(fallback_config, CancellationToken::new()),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(fallback.cameras[0].relay_fallback > 0);
    assert!(fallback.cameras[0].encoded > 0);
    assert_eq!(fallback.cameras[0].relayed_packets, 0);
    // A file recorder remains active even without a playback lease.
    target.set_path(&format!("/{name}-recording"));
    let recording = root.path().join("recorded.mp4");
    recording_config.streams[0].output=serde_json::from_value(json!({"clean":target.as_str(),"annotated":recording,"clean_passthrough":true,"client_overlay":true,"on_demand":true})).unwrap();
    let recording_report = tokio::time::timeout(
        Duration::from_secs(15),
        runtime::run(recording_config, CancellationToken::new()),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(recording_report.cameras[0].relayed_packets, 0);
    assert!(recording_report.cameras[0].encoded > 0);
    let check = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(recording)
        .args(["-f", "null", "-"])
        .output()
        .unwrap();
    assert!(check.status.success());
}

#[test]
#[ignore = "Requires loopback CHD_RUST_TEST_RTSP_URL and a running RTSP server"]
fn rtsp_encoder_drop_closes_muxer_sockets_without_explicit_finish() {
    fn sockets() -> usize {
        std::fs::read_dir("/proc/self/fd")
            .unwrap()
            .flatten()
            .filter(|entry| {
                std::fs::read_link(entry.path())
                    .is_ok_and(|link| link.to_string_lossy().starts_with("socket:["))
            })
            .count()
    }
    let mut target =
        reqwest::Url::parse(&std::env::var("CHD_RUST_TEST_RTSP_URL").unwrap()).unwrap();
    assert!(matches!(target.host_str(), Some("127.0.0.1" | "localhost")));
    target.set_path(&format!("/drop-check-{}", uuid::Uuid::new_v4()));
    let pixels = vec![20; 640 * 352 * 3 + 64];
    let before = sockets();
    for _ in 0..8 {
        let mut encoder =
            construction_media::Encoder::open(target.as_str(), 640, 352, 15, None).unwrap();
        encoder.write(&pixels).unwrap();
        drop(encoder);
        assert_eq!(sockets(), before, "RTSP encoder Drop leaked a muxer socket");
    }
    assert_eq!(construction_media::live_handles(), (0, 0));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "Requires isolated loopback Redis; pauses only that test server"]
async fn metadata_latest_mailbox_does_not_wait_for_slow_redis() {
    use construction_pipeline::metadata::{Actor, Snapshot};
    use std::sync::{Arc, atomic::Ordering};
    let url = std::env::var("CHD_RUST_TEST_REDIS_URL").unwrap();
    let parsed = reqwest::Url::parse(&url).unwrap();
    assert!(matches!(parsed.host_str(), Some("127.0.0.1" | "localhost")));
    assert_eq!(
        parsed.port(),
        Some(26380),
        "Pause only the isolated benchmark Redis"
    );
    let mut db = redis::Client::open(url)
        .unwrap()
        .get_multiplexed_async_connection()
        .await
        .unwrap();
    let root = tempfile::tempdir().unwrap();
    let config = fixture(root.path());
    let prefix = format!("mailbox-test:{}:", uuid::Uuid::new_v4());
    let redis =
        serde_json::from_value(json!({"url_env":"CHD_RUST_TEST_REDIS_URL","prefix":prefix}))
            .unwrap();
    let actor = Actor::new(redis, config.streams[0].clone(), config.models[0].clone());
    let snapshot = |sequence| Snapshot {
        sequence,
        generation: 1,
        timestamp: chrono::Utc::now().timestamp_millis(),
        tracks: Arc::new(vec![]),
        hazard: Arc::new(HazardResult::default()),
        working: true,
        source_pts: None,
    };
    actor.offer(snapshot(1));
    let key = format!("{prefix}rust_latest:1:1");
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            let value: Option<String> = redis::cmd("GET")
                .arg(&key)
                .query_async(&mut db)
                .await
                .unwrap();
            if value.is_some() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let _: () = redis::cmd("CLIENT")
        .arg("PAUSE")
        .arg(1200)
        .arg("ALL")
        .query_async(&mut db)
        .await
        .unwrap();
    actor.offer(snapshot(2));
    tokio::time::sleep(Duration::from_millis(30)).await;
    let began = std::time::Instant::now();
    for i in 3..=1000 {
        actor.offer(snapshot(i));
    }
    let offer_ms = began.elapsed().as_secs_f64() * 1000.;
    assert!(offer_ms < 100., "Inference-side offer blocked on Redis");
    tokio::time::timeout(Duration::from_secs(4), async {
        loop {
            let value: Option<String> = redis::cmd("GET")
                .arg(&key)
                .query_async(&mut db)
                .await
                .unwrap();
            if value.is_some_and(|s| serde_json::from_str::<Value>(&s).unwrap()["sequence"] == 1000)
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let stats = actor.stats.clone();
    actor.finish().await;
    assert_eq!(stats.offered.load(Ordering::Relaxed), 1000);
    assert!(stats.superseded.load(Ordering::Relaxed) >= 997);
    assert_eq!(stats.errors.load(Ordering::Relaxed), 0);
    assert_eq!(
        stats.sent.load(Ordering::Relaxed) + stats.superseded.load(Ordering::Relaxed),
        1000
    );
    println!(
        "metadata_slow_fixture={}",
        json!({"redis_pause_ms":1200,"offered":1000,"producer_ms":offer_ms,
        "sent":stats.sent.load(Ordering::Relaxed),"superseded":stats.superseded.load(Ordering::Relaxed),"last_sequence":1000})
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "Requires isolated Redis, RTSP and CHD_RUST_TEST_MEDIA_API_URL"]
async fn preview_language_demand_starts_only_requested_variant() {
    use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
    use redis::AsyncCommands;
    let root = tempfile::tempdir().unwrap();
    let mut cfg = fixture(root.path());
    let source = std::env::var("CHD_RUST_TEST_RTSP_URL").unwrap();
    let api = std::env::var("CHD_RUST_TEST_MEDIA_API_URL").unwrap();
    assert_eq!(
        reqwest::Url::parse(&source).unwrap().host_str(),
        Some("127.0.0.1")
    );
    let prefix = format!("preview-test:{}:", uuid::Uuid::new_v4());
    let name = format!("hazard_preview_test_{}", uuid::Uuid::new_v4().simple());
    let mut target = reqwest::Url::parse(&source).unwrap();
    target.set_path(&format!("/{name}"));
    cfg.lossless = false;
    cfg.max_frames = None;
    cfg.capture.host_yuv = true;
    cfg.redis = Some(
        serde_json::from_value(json!({"url_env":"CHD_RUST_TEST_REDIS_URL","prefix":prefix}))
            .unwrap(),
    );
    let s = &mut cfg.streams[0];
    s.source = source;
    let languages = [
        ("zh-TW", "zh-TW"),
        ("en", "en"),
        ("zh_CN", "zh-CN"),
        ("ja", "ja"),
        ("vi", "vi"),
        ("id", "id"),
        ("fr-FR", "fr"),
        ("th_TH", "th"),
    ];
    s.languages = languages
        .iter()
        .map(|(configured, _)| (*configured).into())
        .collect();
    s.output=serde_json::from_value(json!({"clean":target.as_str(),"annotated":format!("{target}_annotated_{{language_b64}}"),"on_demand":true,
        "preview":{"clean":format!("{target}_preview"),"annotated":format!("{target}_preview_annotated_{{language_b64}}"),"width":320,"height":176,"fps":15}})).unwrap();
    cfg.validate().unwrap();
    let mut db = redis::Client::open(std::env::var("CHD_RUST_TEST_REDIS_URL").unwrap())
        .unwrap()
        .get_multiplexed_async_connection()
        .await
        .unwrap();
    let stop = CancellationToken::new();
    let task = tokio::spawn(runtime::run(cfg, stop.clone()));
    let latest = format!("{prefix}rust_latest:1:1");
    tokio::time::timeout(Duration::from_secs(15), async {
        while !db.exists::<_, bool>(&latest).await.unwrap() {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .unwrap();
    let client = reqwest::Client::new();
    let ready = async |path: &str| -> bool {
        client
            .get(format!("{api}/v3/paths/get/{path}"))
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap()["ready"]
            .as_bool()
            .unwrap_or(false)
    };
    let paths = languages
        .iter()
        .flat_map(|(_, canonical)| {
            let code = URL_SAFE_NO_PAD.encode(canonical);
            [
                format!("{name}_annotated_{code}"),
                format!("{name}_preview_annotated_{code}"),
            ]
        })
        .chain([name.clone(), format!("{name}_preview")])
        .collect::<Vec<_>>();
    for path in &paths {
        assert!(!ready(path).await, "Unleased output {path} started");
    }
    for language in ["zh-TW", "en", "fr", "th"] {
        let code = URL_SAFE_NO_PAD.encode(language);
        let demand = format!("{prefix}media_overlay_demand:{name}_preview:{code}");
        let requested = format!("{name}_preview_annotated_{code}");
        let marker = format!("{prefix}media_overlay_ready:{requested}");
        let _: () = db.set_ex(&demand, "1", 30).await.unwrap();
        tokio::time::timeout(Duration::from_secs(12), async {
            while !ready(&requested).await || !db.exists::<_, bool>(&marker).await.unwrap() {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("Requested {language} preview did not start"));
        for other in paths.iter().filter(|other| *other != &requested) {
            assert!(!ready(other).await, "Unleased output {other} started");
            let marker = format!("{prefix}media_overlay_ready:{other}");
            assert!(!db.exists::<_, bool>(marker).await.unwrap());
        }
        let output = tokio::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-rtsp_transport",
                "tcp",
                "-timeout",
                "5000000",
                "-show_entries",
                "stream=codec_name,width,height",
                "-of",
                "json",
                "-i",
                &format!("{target}_preview_annotated_{code}"),
            ])
            .output()
            .await
            .unwrap();
        assert!(output.status.success(), "Cannot decode {language} preview");
        let dims: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(dims["streams"][0]["codec_name"], "h264");
        assert_eq!(dims["streams"][0]["width"], 320);
        assert_eq!(dims["streams"][0]["height"], 176);
        let _: () = db.expire(&demand, 1).await.unwrap();
        tokio::time::timeout(Duration::from_secs(8), async {
            while ready(&requested).await || db.exists::<_, bool>(&marker).await.unwrap() {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("Expired {language} preview did not stop"));
    }
    stop.cancel();
    let report = tokio::time::timeout(Duration::from_secs(15), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(report.cameras[0].processed > 10);
    assert!(report.cameras[0].encoded > 0);
    assert!(report.cameras[0].rendered > 0);
    assert!(report.cameras[0].encoder_annotated_host_frames > 0);
    assert_eq!(report.cameras[0].host_materialized, 0);
    let keys: Vec<String> = db.keys(format!("{prefix}*")).await.unwrap();
    if !keys.is_empty() {
        let _: () = db.del(keys).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "Requires loopback CHD_RUST_TEST_RTSP_URL and a running RTSP server"]
async fn scaled_preview_keeps_publishing_offline_then_accepts_a_fresh_frame() {
    use construction_pipeline::{
        frame::{Frame, Pool},
        publish,
        render::Text,
    };
    use std::sync::{Arc, atomic::Ordering};
    use std::time::Instant;

    let root = tempfile::tempdir().unwrap();
    let mut cfg = fixture(root.path());
    cfg.lossless = false;
    let mut target =
        reqwest::Url::parse(&std::env::var("CHD_RUST_TEST_RTSP_URL").unwrap()).unwrap();
    assert_eq!(target.host_str(), Some("127.0.0.1"));
    target.set_path(&format!("/scaled-offline-{}", uuid::Uuid::new_v4()));

    let stop = CancellationToken::new();
    let publisher = publish::spawn_gated(
        Arc::new(cfg),
        target.into(),
        320,
        176,
        15,
        false,
        Text::new(&Default::default()).unwrap(),
        construction_pipeline::demand::Gate::new(true),
        stop.clone(),
    );
    let pool = Pool::new(1920 * 1080 * 3, 2);
    let frame = |sequence, captured| {
        Arc::new(Frame {
            pixels: pool.take(),
            device: None,
            host: None,
            annotation: None,
            width: 1920,
            height: 1080,
            sequence,
            generation: sequence,
            timestamp_ms: 0,
            captured,
            source_pts: None,
            source_lag_ms: 0.,
        })
    };
    assert!(
        publisher
            .sender
            .send(frame(1, Instant::now() - Duration::from_secs(6)), &stop)
            .await
    );
    tokio::time::timeout(Duration::from_secs(5), async {
        while publisher.stats.offline.load(Ordering::Relaxed) < 3 {
            assert!(
                !publisher.task.is_finished(),
                "Offline preview terminated its publisher"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    assert!(publisher.stats.ready.load(Ordering::Acquire));
    assert!(publisher.stats.conversion_reused.load(Ordering::Relaxed) >= 2);
    assert!(publisher.sender.send(frame(2, Instant::now()), &stop).await);
    tokio::time::timeout(Duration::from_secs(5), async {
        while publisher.stats.unique.load(Ordering::Relaxed) < 2 {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    stop.cancel();
    publisher.task.await.unwrap().unwrap();
}
