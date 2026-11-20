//! PostgreSQL event/JPEG transactions and fenced delivery leases.
use crate::{
    config::Config,
    hazard::{Polygons, Track, Warnings},
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sqlx::{PgPool, Postgres, Row, Transaction, postgres::PgPoolOptions, types::Json};
use std::{path::Path, time::Duration};
#[path = "outbox_legacy.rs"]
mod legacy;

#[derive(Clone, Serialize, Deserialize)]
pub struct Event {
    pub event_id: String,
    pub site_id: i64,
    pub site: String,
    pub stream_name: String,
    pub detection_time: String,
    pub model_id: String,
    pub model_version: String,
    pub warnings: Warnings,
    pub detections: Vec<Track>,
    pub cone_polygon: Polygons,
    pub pole_polygon: Polygons,
}
pub struct Pending {
    pub metadata: Event,
    pub image: Vec<u8>,
    pub violation_id: Option<String>,
    pub attempts: u32,
    claim_token: String,
}
#[derive(Clone, Copy, Debug)]
pub(crate) enum ClaimStage {
    Any,
    Evidence,
    Notification,
}
#[derive(Clone)]
pub struct Outbox {
    pub(crate) pool: PgPool,
    pub(crate) namespace: String,
    max_attempts: u32,
}
impl Outbox {
    pub async fn open(config: &Config) -> Result<Self> {
        let env = config
            .outbox
            .url_env
            .as_deref()
            .or_else(|| config.database.as_ref().map(|d| d.url_env.as_str()))
            .unwrap_or("DATABASE_URL");
        let url = std::env::var(env)
            .with_context(|| format!("PostgreSQL outbox requires environment variable {env}"))?
            .replacen("postgresql+asyncpg://", "postgresql://", 1);
        std::fs::create_dir_all(&config.state_dir)?;
        let namespace = match &config.outbox.namespace {
            Some(value) => value.clone(),
            None => format!(
                "state:{}",
                hex::encode(Sha256::digest(
                    config
                        .state_dir
                        .canonicalize()?
                        .to_string_lossy()
                        .as_bytes()
                ))
            ),
        };
        Self::connect(
            &url,
            &namespace,
            config.delivery.outbox_limit_bytes,
            config.delivery.max_attempts,
        )
        .await
    }
    pub async fn connect(
        url: &str,
        namespace: &str,
        limit: u64,
        max_attempts: u32,
    ) -> Result<Self> {
        ensure!(
            !namespace.is_empty() && namespace.len() <= 256,
            "Invalid PostgreSQL outbox namespace"
        );
        ensure!(
            limit >= 1024 && limit <= i64::MAX as u64 && max_attempts > 0,
            "Invalid PostgreSQL outbox bounds"
        );
        let pool=PgPoolOptions::new().max_connections(4).acquire_timeout(Duration::from_secs(5))
            .after_connect(|c,_| Box::pin(async move {
                sqlx::query("SELECT set_config('TimeZone','UTC',false),set_config('synchronous_commit','on',false),set_config('statement_timeout','10s',false),set_config('lock_timeout','5s',false)").execute(c).await?;Ok(())
            })).connect(url).await.context("Cannot connect to PostgreSQL outbox")?;
        let mut tx = pool.begin().await?;
        // Serialize DDL across simultaneous pipeline/CLI starts.
        sqlx::query("SELECT pg_advisory_xact_lock(1128813135,1)")
            .execute(&mut *tx)
            .await?;
        sqlx::Executor::execute(&mut *tx, include_str!("../migrations/001_outbox.sql")).await?;
        sqlx::query(
            "INSERT INTO rust_pipeline.outbox_namespaces(namespace,byte_limit) VALUES($1,$2)
            ON CONFLICT(namespace) DO UPDATE SET byte_limit=EXCLUDED.byte_limit",
        )
        .bind(namespace)
        .bind(limit as i64)
        .execute(&mut *tx)
        .await?;
        let (used,budget):(i64,i64)=sqlx::query_as("SELECT payload_bytes,byte_limit FROM rust_pipeline.outbox_namespaces WHERE namespace=$1")
            .bind(namespace).fetch_one(&mut *tx).await?;
        ensure!(
            used <= budget,
            "Existing outbox exceeds the configured storage budget"
        );
        tx.commit().await?;
        Ok(Self {
            pool,
            namespace: namespace.into(),
            max_attempts,
        })
    }
    pub(crate) async fn quota(&self, tx: &mut Transaction<'_, Postgres>) -> Result<(i64, i64)> {
        Ok(sqlx::query_as("SELECT payload_bytes,byte_limit FROM rust_pipeline.outbox_namespaces WHERE namespace=$1 FOR UPDATE")
            .bind(&self.namespace).fetch_one(&mut **tx).await?)
    }
    pub async fn put(&self, event: &Event, image: &[u8]) -> Result<bool> {
        let bytes = serde_json::to_vec(event)?
            .len()
            .checked_add(image.len())
            .context("Outbox payload overflow")?;
        let mut tx = self.pool.begin().await?;
        let (used, limit) = self.quota(&mut tx).await?;
        let exists:bool=sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM rust_pipeline.outbox_events WHERE namespace=$1 AND event_id=$2)")
            .bind(&self.namespace).bind(&event.event_id).fetch_one(&mut *tx).await?;
        if exists {
            tx.commit().await?;
            return Ok(false);
        }
        ensure!(
            bytes as u64 <= (limit - used) as u64,
            "Evidence outbox storage budget exhausted"
        );
        sqlx::query("INSERT INTO rust_pipeline.outbox_events(namespace,event_id,metadata,image,payload_bytes,created_at)
            VALUES($1,$2,$3,$4,$5,$6)")
            .bind(&self.namespace).bind(&event.event_id).bind(Json(event)).bind(image).bind(bytes as i64)
            .bind(chrono::Utc::now().timestamp()).execute(&mut *tx).await?;
        sqlx::query("UPDATE rust_pipeline.outbox_namespaces SET payload_bytes=payload_bytes+$2 WHERE namespace=$1")
            .bind(&self.namespace).bind(bytes as i64).execute(&mut *tx).await?;
        tx.commit().await?;
        Ok(true)
    }
    pub async fn claim(&self, now: i64) -> Result<Option<Pending>> {
        self.claim_stage(now, ClaimStage::Any).await
    }
    pub(crate) async fn claim_stage(&self, now: i64, stage: ClaimStage) -> Result<Option<Pending>> {
        let token = uuid::Uuid::new_v4().to_string();
        let has_receipt = match stage {
            ClaimStage::Any => None,
            ClaimStage::Evidence => Some(false),
            ClaimStage::Notification => Some(true),
        };
        let row = sqlx::query(
            "WITH due AS (
            SELECT event_id FROM rust_pipeline.outbox_events WHERE namespace=$1
            AND ((state='pending' AND next_at<=$2) OR (state='inflight' AND lease_until<=$2))
            AND ($5::bool IS NULL OR (violation_id IS NOT NULL)=$5)
            ORDER BY CASE WHEN $5::bool IS NULL THEN created_at ELSE
                GREATEST(created_at, CASE WHEN state='inflight' THEN lease_until ELSE next_at END)
                END,created_at,event_id LIMIT 1 FOR UPDATE SKIP LOCKED)
            UPDATE rust_pipeline.outbox_events e SET state='inflight',lease_until=$3,claim_token=$4
            FROM due WHERE e.namespace=$1 AND e.event_id=due.event_id
            RETURNING e.metadata,CASE WHEN e.violation_id IS NOT NULL AND $5::bool IS TRUE
                THEN ''::bytea ELSE e.image END AS image,e.violation_id,e.attempts",
        )
        .bind(&self.namespace)
        .bind(now)
        .bind(now.checked_add(120).context("Lease timestamp overflow")?)
        .bind(&token)
        .bind(has_receipt)
        .fetch_optional(&self.pool)
        .await?;
        row.map(|row| -> Result<Pending> {
            Ok(Pending {
                metadata: row.try_get::<Json<Event>, _>("metadata")?.0,
                image: row.try_get("image")?,
                violation_id: row.try_get("violation_id")?,
                attempts: u32::try_from(row.try_get::<i64, _>("attempts")?)?,
                claim_token: token,
            })
        })
        .transpose()
    }
    fn owned(rows: u64) -> Result<()> {
        ensure!(rows == 1, "Outbox delivery lease is no longer owned");
        Ok(())
    }
    pub async fn uploaded(&self, pending: &Pending, violation_id: &str) -> Result<()> {
        Self::owned(
            sqlx::query(
                "UPDATE rust_pipeline.outbox_events SET violation_id=$4
            WHERE namespace=$1 AND event_id=$2 AND claim_token=$3 AND state='inflight'",
            )
            .bind(&self.namespace)
            .bind(&pending.metadata.event_id)
            .bind(&pending.claim_token)
            .bind(violation_id)
            .execute(&self.pool)
            .await?
            .rows_affected(),
        )
    }
    pub(crate) async fn schedule_notification(
        &self,
        pending: &Pending,
        violation_id: &str,
        now: i64,
    ) -> Result<()> {
        Self::owned(
            sqlx::query(
                "UPDATE rust_pipeline.outbox_events SET violation_id=$4,state='pending',
                 next_at=$5,attempts=0,claim_token=NULL,lease_until=0,last_error=NULL
                 WHERE namespace=$1 AND event_id=$2 AND claim_token=$3 AND state='inflight'",
            )
            .bind(&self.namespace)
            .bind(&pending.metadata.event_id)
            .bind(&pending.claim_token)
            .bind(violation_id)
            .bind(now)
            .execute(&self.pool)
            .await?
            .rows_affected(),
        )
    }
    pub async fn complete(&self, pending: &Pending) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        self.quota(&mut tx).await?;
        let bytes: Option<i64> = sqlx::query_scalar(
            "SELECT payload_bytes FROM rust_pipeline.outbox_events
            WHERE namespace=$1 AND event_id=$2 AND claim_token=$3 AND state='inflight' FOR UPDATE",
        )
        .bind(&self.namespace)
        .bind(&pending.metadata.event_id)
        .bind(&pending.claim_token)
        .fetch_optional(&mut *tx)
        .await?;
        let bytes = bytes.context("Outbox delivery lease is no longer owned")?;
        sqlx::query(
            "UPDATE rust_pipeline.outbox_events SET state='done',image=NULL,payload_bytes=0,
            last_error=NULL,lease_until=0,claim_token=NULL WHERE namespace=$1 AND event_id=$2",
        )
        .bind(&self.namespace)
        .bind(&pending.metadata.event_id)
        .execute(&mut *tx)
        .await?;
        sqlx::query("UPDATE rust_pipeline.outbox_namespaces SET payload_bytes=payload_bytes-$2 WHERE namespace=$1")
            .bind(&self.namespace).bind(bytes).execute(&mut *tx).await?;
        sqlx::query("DELETE FROM rust_pipeline.outbox_events WHERE namespace=$1 AND state='done' AND event_id IN
            (SELECT event_id FROM rust_pipeline.outbox_events WHERE namespace=$1 AND state='done'
             ORDER BY created_at DESC,event_id DESC OFFSET 10000)")
            .bind(&self.namespace).execute(&mut *tx).await?;
        tx.commit().await?;
        Ok(())
    }
    pub async fn retry(
        &self,
        pending: &Pending,
        now: i64,
        permanent: bool,
        reason: &str,
    ) -> Result<()> {
        self.retry_with_delay(pending, now, permanent, reason, 0)
            .await
    }
    pub(crate) async fn retry_with_delay(
        &self,
        pending: &Pending,
        now: i64,
        permanent: bool,
        reason: &str,
        retry_after: i64,
    ) -> Result<()> {
        let attempts = pending.attempts.saturating_add(1);
        let failed = permanent || attempts >= self.max_attempts;
        let next = now
            .checked_add((1i64 << attempts.min(10)).max(retry_after))
            .context("Retry timestamp overflow")?;
        Self::owned(sqlx::query("UPDATE rust_pipeline.outbox_events SET state=$4,attempts=$5,next_at=$6,
            lease_until=0,claim_token=NULL,last_error=$7 WHERE namespace=$1 AND event_id=$2 AND claim_token=$3 AND state='inflight'")
            .bind(&self.namespace).bind(&pending.metadata.event_id).bind(&pending.claim_token)
            .bind(if failed{"failed"}else{"pending"}).bind(i64::from(attempts)).bind(next).bind(reason)
            .execute(&self.pool).await?.rows_affected())
    }
    pub async fn recover_error(&self, pending: &Pending) -> Result<()> {
        self.retry(
            pending,
            chrono::Utc::now().timestamp(),
            false,
            "Invalid transport response",
        )
        .await
    }
    pub async fn counts(&self) -> Result<serde_json::Value> {
        let rows=sqlx::query("SELECT state,count(*) AS n FROM rust_pipeline.outbox_events WHERE namespace=$1 GROUP BY state")
            .bind(&self.namespace).fetch_all(&self.pool).await?;
        let mut values = serde_json::Map::new();
        for row in rows {
            values.insert(
                row.try_get("state")?,
                serde_json::json!(row.try_get::<i64, _>("n")?),
            );
        }
        Ok(values.into())
    }
    pub async fn requeue_failed(&self) -> Result<u64> {
        Ok(sqlx::query("UPDATE rust_pipeline.outbox_events SET state='pending',attempts=0,next_at=0,lease_until=0,
            claim_token=NULL WHERE namespace=$1 AND state='failed'")
            .bind(&self.namespace).execute(&self.pool).await?.rows_affected())
    }
    /// Read-only legacy import. The caller must stop the previous state-dir
    /// owner before migration; runtime and the CLI hold that existing lease.
    pub async fn migrate_sqlite(&self, path: &Path) -> Result<u64> {
        legacy::migrate(self, path).await
    }
}
