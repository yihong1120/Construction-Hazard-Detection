//! These tests require an isolated database, never the application's database.
use axum::{
    Json, Router, body::Bytes, extract::State, http::HeaderMap, response::IntoResponse,
    routing::post,
};
use construction_pipeline::{
    config::Delivery,
    delivery::dispatch,
    hazard::Warnings,
    outbox::{Event, Outbox},
};
use serde_json::json;
use sqlx::PgPool;
use std::{
    collections::HashSet,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering::Relaxed},
    },
    time::Duration,
};
use tokio_util::sync::CancellationToken;
fn event(id: &str) -> Event {
    Event {
        event_id: id.into(),
        site_id: 1,
        site: "fixture".into(),
        stream_name: "test".into(),
        detection_time: "2026-11-18T00:00:00Z".into(),
        model_id: "fixture".into(),
        model_version: "v1".into(),
        warnings: Warnings::new(),
        detections: vec![],
        cone_polygon: Default::default(),
        pole_polygon: Default::default(),
    }
}
fn url() -> String {
    let url = std::env::var("CHD_RUST_TEST_DATABASE_URL")
        .expect("Run an isolated PostgreSQL test service first");
    let parsed = reqwest::Url::parse(&url).unwrap();
    assert!(matches!(parsed.host_str(), Some("127.0.0.1" | "localhost")));
    assert_eq!(parsed.path(), "/chd_rust_pipeline_test");
    url
}
async fn fixture(limit: u64) -> (Outbox, String) {
    let scope = format!("outbox-test:{}", uuid::Uuid::new_v4());
    (
        Outbox::connect(&url(), &scope, limit, 3).await.unwrap(),
        scope,
    )
}
#[tokio::test]
async fn restart_preserves_receipts_and_rejects_stale_delivery_leases() {
    let (store, scope) = fixture(16384).await;
    assert!(store.put(&event("one"), b"jpeg").await.unwrap());
    assert!(!store.put(&event("one"), b"duplicate").await.unwrap());
    let first = store.claim(1).await.unwrap().unwrap();
    assert!(store.claim(2).await.unwrap().is_none());
    store.uploaded(&first, "42").await.unwrap();
    drop(store);
    let store = Outbox::connect(&url(), &scope, 16384, 3).await.unwrap();
    let recovered = store.claim(122).await.unwrap().unwrap();
    assert_eq!(recovered.violation_id.as_deref(), Some("42"));
    assert_eq!(recovered.image, b"jpeg");
    assert!(store.uploaded(&first, "99").await.is_err());
    assert!(store.complete(&first).await.is_err());
    assert!(store.retry(&first, 123, false, "stale").await.is_err());
    store.complete(&recovered).await.unwrap();
    assert_eq!(store.counts().await.unwrap()["done"], 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stalled_notifications_leave_evidence_capacity_and_honour_retry_after() {
    struct Endpoints {
        uploads: AtomicUsize,
        notifications: AtomicUsize,
        release: tokio::sync::Semaphore,
    }
    async fn upload(State(state): State<Arc<Endpoints>>, _image: Bytes) -> impl IntoResponse {
        let id = state.uploads.fetch_add(1, Relaxed) + 1;
        Json(json!({"violation_id": id}))
    }
    async fn notify(State(state): State<Arc<Endpoints>>) -> impl IntoResponse {
        let attempt = state.notifications.fetch_add(1, Relaxed);
        let mut headers = HeaderMap::new();
        let reply = match attempt {
            0 | 1 => {
                state.release.acquire().await.unwrap().forget();
                if attempt == 0 {
                    headers.insert("Retry-After", "120".parse().unwrap());
                    json!({"outcome": "retryable"})
                } else {
                    json!({"outcome": "retryable", "retry_after_seconds": 180})
                }
            }
            _ => json!({"outcome": "completed"}),
        };
        (headers, Json(reply))
    }

    let (store, scope) = fixture(1024 * 1024).await;
    for id in 0..8 {
        store
            .put(&event(&id.to_string()), &[23; 32768])
            .await
            .unwrap();
    }
    let state = Arc::new(Endpoints {
        uploads: AtomicUsize::new(0),
        notifications: AtomicUsize::new(0),
        release: tokio::sync::Semaphore::new(0),
    });
    let app = Router::new()
        .route("/upload", post(upload))
        .route("/notify", post(notify))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let stop = CancellationToken::new();
    let worker = tokio::spawn(dispatch(
        store.clone(),
        Delivery {
            upload_url: Some(format!("http://{addr}/upload")),
            notification_url: Some(format!("http://{addr}/notify")),
            ..Delivery::default()
        },
        stop.clone(),
    ));
    let pool = PgPool::connect(&url()).await.unwrap();

    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let receipts: i64 = sqlx::query_scalar(
                "SELECT count(*) FROM rust_pipeline.outbox_events
                WHERE namespace=$1 AND violation_id IS NOT NULL AND octet_length(image)=32768",
            )
            .bind(&scope)
            .fetch_one(&pool)
            .await
            .unwrap();
            if receipts == 8 && state.notifications.load(Relaxed) >= 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("Every JPEG should be uploaded while the two notification requests are stalled");
    assert_eq!(state.notifications.load(Relaxed), 2);
    let released_at = chrono::Utc::now().timestamp();
    state.release.add_permits(2);

    tokio::time::timeout(Duration::from_secs(10), async {
        while store.counts().await.unwrap()["done"] != 6 {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .unwrap();
    let retry_at: Vec<i64> = sqlx::query_scalar(
        "SELECT next_at FROM rust_pipeline.outbox_events
        WHERE namespace=$1 AND state='pending' ORDER BY next_at",
    )
    .bind(&scope)
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(retry_at.len(), 2);
    for (at, delay) in retry_at.into_iter().zip([120, 180]) {
        assert!(at >= released_at + delay);
        assert!(at <= chrono::Utc::now().timestamp() + delay);
    }
    // Make the two persisted retries due in the fixture without sleeping for minutes.
    sqlx::query(
        "UPDATE rust_pipeline.outbox_events SET next_at=0 WHERE namespace=$1 AND state='pending'",
    )
    .bind(&scope)
    .execute(&pool)
    .await
    .unwrap();
    tokio::time::timeout(Duration::from_secs(10), async {
        while store.counts().await.unwrap()["done"] != 8 {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .unwrap();
    stop.cancel();
    worker.await.unwrap().unwrap();
    assert_eq!(state.uploads.load(Relaxed), 8);
    assert_eq!(state.notifications.load(Relaxed), 10);
    server.abort();
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_workers_claim_each_event_once() {
    let (store, _) = fixture(1024 * 1024).await;
    for i in 0..40 {
        store
            .put(&event(&format!("event-{i}")), b"jpeg")
            .await
            .unwrap();
    }
    let seen = Arc::new(tokio::sync::Mutex::new(HashSet::new()));
    let barrier = Arc::new(tokio::sync::Barrier::new(12));
    let mut tasks = tokio::task::JoinSet::new();
    for _ in 0..12 {
        let (store, seen, barrier) = (store.clone(), seen.clone(), barrier.clone());
        tasks.spawn(async move {
            barrier.wait().await;
            while let Some(pending) = store.claim(10).await.unwrap() {
                assert!(seen.lock().await.insert(pending.metadata.event_id.clone()));
                store.complete(&pending).await.unwrap();
            }
        });
    }
    while let Some(result) = tasks.join_next().await {
        result.unwrap();
    }
    assert_eq!(seen.lock().await.len(), 40);
    assert_eq!(store.counts().await.unwrap()["done"], 40);
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_budget_checks_are_atomic_and_completion_releases_capacity() {
    let (store, scope) = fixture(4096).await;
    let mut tasks = tokio::task::JoinSet::new();
    for i in 0..20 {
        let store = store.clone();
        tasks.spawn(async move {
            store
                .put(&event(&format!("budget-{i}")), &[0; 512])
                .await
                .is_ok()
        });
    }
    let mut accepted = 0;
    while let Some(result) = tasks.join_next().await {
        accepted += usize::from(result.unwrap());
    }
    assert!(accepted > 0 && accepted < 20);
    let pool = PgPool::connect(&url()).await.unwrap();
    let (used,total):(i64,i64)=sqlx::query_as("SELECT n.payload_bytes,COALESCE(SUM(e.payload_bytes),0)::bigint
        FROM rust_pipeline.outbox_namespaces n LEFT JOIN rust_pipeline.outbox_events e USING(namespace)
        WHERE n.namespace=$1 GROUP BY n.namespace").bind(&scope).fetch_one(&pool).await.unwrap();
    assert_eq!(used, total);
    assert!(used <= 4096);
    while let Some(pending) = store.claim(10).await.unwrap() {
        store.complete(&pending).await.unwrap();
    }
    let used: i64 = sqlx::query_scalar(
        "SELECT payload_bytes FROM rust_pipeline.outbox_namespaces WHERE namespace=$1",
    )
    .bind(&scope)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(used, 0);
    assert!(store.put(&event("after-release"), &[0; 512]).await.unwrap());
}
#[tokio::test]
async fn namespaces_are_isolated_and_failed_events_can_be_requeued() {
    let (a, _) = fixture(16384).await;
    let (b, _) = fixture(16384).await;
    for store in [&a, &b] {
        store.put(&event("same-id"), b"jpeg").await.unwrap();
    }
    let pending = a.claim(1).await.unwrap().unwrap();
    a.retry(&pending, 1, true, "rejected").await.unwrap();
    assert!(a.claim(10000).await.unwrap().is_none());
    assert_eq!(a.counts().await.unwrap()["failed"], 1);
    assert_eq!(b.counts().await.unwrap()["pending"], 1);
    assert_eq!(a.requeue_failed().await.unwrap(), 1);
    assert_eq!(a.claim(10001).await.unwrap().unwrap().attempts, 0);
    assert!(b.claim(10001).await.unwrap().is_some());
}
fn legacy(path: &std::path::Path) -> rusqlite::Connection {
    let db = rusqlite::Connection::open(path).unwrap();
    db.execute_batch("CREATE TABLE events(id TEXT PRIMARY KEY,metadata TEXT,image BLOB,violation_id TEXT,
        state TEXT,attempts INTEGER,next_at INTEGER,lease_until INTEGER,last_error TEXT,created_at INTEGER)").unwrap();
    db
}
fn legacy_row(
    db: &rusqlite::Connection,
    id: &str,
    state: &str,
    receipt: Option<&str>,
    image: Option<&[u8]>,
    attempts: i64,
    created: i64,
) {
    db.execute(
        "INSERT INTO events VALUES(?1,?2,?3,?4,?5,?6,0,999999,'fixture error',?7)",
        rusqlite::params![
            id,
            serde_json::to_string(&event(id)).unwrap(),
            image,
            receipt,
            state,
            attempts,
            created
        ],
    )
    .unwrap();
}
#[tokio::test]
async fn sqlite_import_is_read_only_atomic_idempotent_and_preserves_receipts() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("outbox.sqlite3");
    let db = legacy(&path);
    legacy_row(
        &db,
        "a-inflight",
        "inflight",
        Some("42"),
        Some(b"first"),
        1,
        1,
    );
    legacy_row(&db, "b-pending", "pending", None, Some(b"second"), 0, 2);
    legacy_row(&db, "c-failed", "failed", None, Some(b"third"), 3, 3);
    legacy_row(&db, "d-done", "done", Some("44"), None, 0, 4);
    let mut metadata = serde_json::to_value(event("a-inflight")).unwrap();
    metadata["operator_note"] = json!({"preserve":"original extension"});
    db.execute(
        "UPDATE events SET metadata=?1 WHERE id='a-inflight'",
        [metadata.to_string()],
    )
    .unwrap();
    drop(db);
    let original = std::fs::read(&path).unwrap();
    let (store, scope) = fixture(16384).await;
    assert_eq!(store.migrate_sqlite(&path).await.unwrap(), 4);
    let pool = PgPool::connect(&url()).await.unwrap();
    let note:serde_json::Value=sqlx::query_scalar("SELECT metadata->'operator_note' FROM rust_pipeline.outbox_events WHERE namespace=$1 AND event_id='a-inflight'")
        .bind(&scope).fetch_one(&pool).await.unwrap();
    assert_eq!(note, json!({"preserve":"original extension"}));
    assert_eq!(
        store.counts().await.unwrap(),
        json!({"pending":2,"failed":1,"done":1})
    );
    let pending = store.claim(1).await.unwrap().unwrap();
    assert_eq!(pending.metadata.event_id, "a-inflight");
    assert_eq!(pending.violation_id.as_deref(), Some("42"));
    assert_eq!(pending.attempts, 1);
    store.complete(&pending).await.unwrap();
    assert_eq!(store.migrate_sqlite(&path).await.unwrap(), 0);
    assert_eq!(store.counts().await.unwrap()["done"], 2);
    assert_eq!(std::fs::read(&path).unwrap(), original);
    let db = rusqlite::Connection::open(&path).unwrap();
    legacy_row(&db, "z-new", "pending", None, Some(b"new"), 0, 5);
    drop(db);
    assert!(store.migrate_sqlite(&path).await.is_err());
    assert_eq!(store.counts().await.unwrap()["pending"], 1);
}
#[tokio::test]
async fn corrupt_import_rolls_back_prior_rows_and_quota() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("outbox.sqlite3");
    let db = legacy(&path);
    legacy_row(&db, "a-good", "pending", None, Some(b"jpeg"), 0, 1);
    db.execute(
        "INSERT INTO events VALUES('z-bad','invalid',X'00',NULL,'pending',0,0,0,NULL,2)",
        [],
    )
    .unwrap();
    drop(db);
    let (store, scope) = fixture(16384).await;
    assert!(store.migrate_sqlite(&path).await.is_err());
    assert_eq!(store.counts().await.unwrap(), json!({}));
    let pool = PgPool::connect(&url()).await.unwrap();
    let used: i64 = sqlx::query_scalar(
        "SELECT payload_bytes FROM rust_pipeline.outbox_namespaces WHERE namespace=$1",
    )
    .bind(scope)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(used, 0);
}
#[tokio::test]
async fn import_rejects_conflicting_evidence_and_never_discards_a_receipt() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("outbox.sqlite3");
    let db = legacy(&path);
    legacy_row(&db, "conflict", "pending", None, Some(b"different"), 0, 1);
    drop(db);
    let (store, _) = fixture(16384).await;
    store.put(&event("conflict"), b"jpeg").await.unwrap();
    assert!(store.migrate_sqlite(&path).await.is_err());
    let db = rusqlite::Connection::open(&path).unwrap();
    db.execute(
        "UPDATE events SET image=?1,violation_id='55'",
        [b"jpeg".as_slice()],
    )
    .unwrap();
    drop(db);
    assert!(store.migrate_sqlite(&path).await.is_err());
    let pending = store.claim(1).await.unwrap().unwrap();
    assert_eq!(pending.image, b"jpeg");
    assert!(pending.violation_id.is_none());
}
#[tokio::test]
async fn over_budget_import_is_all_or_nothing() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("outbox.sqlite3");
    let db = legacy(&path);
    legacy_row(&db, "a", "pending", None, Some(&[0; 400]), 0, 1);
    legacy_row(&db, "b", "pending", None, Some(&[0; 400]), 0, 2);
    drop(db);
    let (store, _) = fixture(1024).await;
    assert!(store.migrate_sqlite(&path).await.is_err());
    assert_eq!(store.counts().await.unwrap(), json!({}));
}
