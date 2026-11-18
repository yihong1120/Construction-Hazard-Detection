//! One-way, read-only import of the previous Rust SQLite queue.
use super::*;
use rusqlite::{Connection, OpenFlags};
use tokio::sync::mpsc;
struct Legacy {
    id: String,
    metadata: String,
    image: Option<Vec<u8>>,
    violation_id: Option<String>,
    state: String,
    attempts: i64,
    next_at: i64,
    last_error: Option<String>,
    created_at: i64,
}
fn read(path: std::path::PathBuf, tx: mpsc::Sender<Vec<Legacy>>) -> Result<String> {
    let db = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    db.busy_timeout(Duration::from_secs(5))?;
    db.execute_batch("PRAGMA query_only=ON; BEGIN")?;
    let mut statement=db.prepare("SELECT id,metadata,image,violation_id,state,attempts,next_at,last_error,created_at FROM events ORDER BY id")?;
    let mut rows = statement.query([])?;
    let mut batch = Vec::new();
    let mut hash = Sha256::new();
    while let Some(row) = rows.next()? {
        let value = Legacy {
            id: row.get(0)?,
            metadata: row.get(1)?,
            image: row.get(2)?,
            violation_id: row.get(3)?,
            state: row.get(4)?,
            attempts: row.get(5)?,
            next_at: row.get(6)?,
            last_error: row.get(7)?,
            created_at: row.get(8)?,
        };
        let fields = serde_json::to_vec(&(
            &value.id,
            &value.metadata,
            &value.violation_id,
            &value.state,
            value.attempts,
            value.next_at,
            &value.last_error,
            value.created_at,
        ))?;
        hash.update((fields.len() as u64).to_le_bytes());
        hash.update(fields);
        match &value.image {
            Some(image) => {
                hash.update((image.len() as u64).to_le_bytes());
                hash.update(image);
            }
            None => hash.update(u64::MAX.to_le_bytes()),
        }
        batch.push(value);
        if batch.len() == 16 {
            tx.blocking_send(std::mem::take(&mut batch))
                .context("Legacy import consumer closed")?;
        }
    }
    if !batch.is_empty() {
        tx.blocking_send(batch)
            .context("Legacy import consumer closed")?;
    }
    Ok(hex::encode(hash.finalize()))
}
pub(super) async fn migrate(store: &Outbox, path: &Path) -> Result<u64> {
    if !path.try_exists()? {
        return Ok(0);
    }
    ensure!(
        path.is_file(),
        "Legacy outbox must be a regular SQLite file"
    );
    let source = path.canonicalize()?;
    let source_key = hex::encode(Sha256::digest(source.to_string_lossy().as_bytes()));
    let mut tx = store.pool.begin().await?;
    let (mut used, limit) = store.quota(&mut tx).await?;
    let previous: Option<String> = sqlx::query_scalar(
        "SELECT fingerprint FROM rust_pipeline.outbox_imports WHERE namespace=$1 AND source_key=$2",
    )
    .bind(&store.namespace)
    .bind(&source_key)
    .fetch_optional(&mut *tx)
    .await?;
    let (sender, mut receiver) = mpsc::channel(2);
    let reader = tokio::task::spawn_blocking(move || read(source, sender));
    let result:Result<u64>=async {
        let mut inserted=0;
        while let Some(batch)=receiver.recv().await {
            for old in batch {
                let event:Event=serde_json::from_str(&old.metadata).context("Invalid legacy event metadata")?;
                ensure!(old.id==event.event_id,"Legacy event identifier conflicts with its metadata");
                ensure!(matches!(old.state.as_str(),"pending"|"inflight"|"failed"|"done")
                    && old.attempts>=0&&old.attempts<=i64::from(u32::MAX),"Invalid legacy event state");
                ensure!((old.state=="done")==old.image.is_none(),"Legacy evidence image is missing");
                if previous.is_some(){continue;}
                let existing=sqlx::query("SELECT metadata=$3::jsonb AS metadata_matches,image,violation_id,state,attempts,lease_until FROM rust_pipeline.outbox_events
                    WHERE namespace=$1 AND event_id=$2 FOR UPDATE")
                    .bind(&store.namespace).bind(&old.id).bind(&old.metadata).fetch_optional(&mut *tx).await?;
                if let Some(row)=existing {
                    let state:String=row.try_get("state")?;
                    let receipt:Option<String>=row.try_get("violation_id")?;
                    ensure!(row.try_get::<bool,_>("metadata_matches")?,"Legacy import conflicts with existing event metadata");
                    ensure!(old.violation_id.is_none()||old.violation_id==receipt,"Legacy import would lose an upload receipt");
                    if state!="done" {
                        ensure!(row.try_get::<Option<Vec<u8>>,_>("image")?==old.image,"Legacy import conflicts with existing evidence");
                        ensure!(old.state!="done"&&row.try_get::<i64,_>("attempts")?>=old.attempts,
                            "Existing PostgreSQL event is behind the legacy delivery state");
                        ensure!(state!="inflight"||row.try_get::<i64,_>("lease_until")?<=chrono::Utc::now().timestamp(),
                            "Stop active delivery workers before legacy import");
                    }
                    continue;
                }
                let bytes=if let Some(image)=&old.image {old.metadata.len().checked_add(image.len()).context("Legacy payload overflow")?} else {0};
                ensure!(bytes as u64<=(limit-used) as u64,"Legacy import exceeds outbox storage budget");
                sqlx::query("INSERT INTO rust_pipeline.outbox_events(namespace,event_id,metadata,image,payload_bytes,
                    violation_id,state,attempts,next_at,last_error,created_at) VALUES($1,$2,$3::jsonb,$4,$5,$6,$7,$8,$9,$10,$11)")
                    .bind(&store.namespace).bind(&old.id).bind(&old.metadata).bind(&old.image).bind(bytes as i64)
                    .bind(&old.violation_id).bind(if old.state=="inflight"{"pending"}else{&old.state})
                    .bind(old.attempts).bind(old.next_at).bind(&old.last_error).bind(old.created_at)
                    .execute(&mut *tx).await?;
                used+=bytes as i64;inserted+=1;
            }
        }
        Ok(inserted)
    }.await;
    drop(receiver);
    let fingerprint = reader.await.context("Legacy reader panicked")?;
    let inserted = result?;
    let fingerprint = fingerprint?;
    if let Some(previous) = previous {
        ensure!(
            previous == fingerprint,
            "SQLite changed after its PostgreSQL import; preserve both stores and reconcile before restarting"
        );
    } else {
        sqlx::query(
            "UPDATE rust_pipeline.outbox_namespaces SET payload_bytes=$2 WHERE namespace=$1",
        )
        .bind(&store.namespace)
        .bind(used)
        .execute(&mut *tx)
        .await?;
        sqlx::query("INSERT INTO rust_pipeline.outbox_imports(namespace,source_key,fingerprint,imported_rows) VALUES($1,$2,$3,$4)")
            .bind(&store.namespace).bind(source_key).bind(fingerprint).bind(inserted as i64).execute(&mut *tx).await?;
    }
    tx.commit().await?;
    Ok(inserted)
}
