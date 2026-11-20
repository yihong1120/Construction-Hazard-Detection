use std::{env, str::FromStr, time::Duration};

use anyhow::{Result, anyhow};
use sqlx::PgPool;
use tokio::time::{sleep, timeout};
use tokio_util::sync::CancellationToken;

const PRUNE: &str = "
    WITH expired AS (
        SELECT event_digest, user_id, token_hash, registration_at
        FROM public.notification_delivery_receipts
        WHERE expires_at <= now()
        ORDER BY expires_at
        LIMIT $1
        FOR UPDATE SKIP LOCKED
    )
    DELETE FROM public.notification_delivery_receipts AS receipt
    USING expired
    WHERE receipt.event_digest = expired.event_digest
      AND receipt.user_id = expired.user_id
      AND receipt.token_hash = expired.token_hash
      AND receipt.registration_at = expired.registration_at";

struct Settings {
    interval: Duration,
    batch_size: i64,
    max_batches: usize,
    budget: Duration,
}

impl Settings {
    fn load() -> Result<Self> {
        let interval = value("NOTIFICATION_RECEIPTS_INTERVAL_SECONDS", 60.0, |value| {
            *value > 0.0
        })?;
        let budget = value("NOTIFICATION_RECEIPTS_BUDGET_SECONDS", 2.0, |value| {
            *value > 0.0 && *value <= 30.0
        })?;
        Ok(Self {
            interval: duration(interval, "NOTIFICATION_RECEIPTS_INTERVAL_SECONDS")?,
            batch_size: value("NOTIFICATION_RECEIPTS_BATCH_SIZE", 500, |value| {
                (1..=5000).contains(value)
            })?,
            max_batches: value("NOTIFICATION_RECEIPTS_MAX_BATCHES", 4, |value| {
                (1..=32).contains(value)
            })?,
            budget: duration(budget, "NOTIFICATION_RECEIPTS_BUDGET_SECONDS")?,
        })
    }

    async fn sweep(&self, db: &PgPool) -> Result<(), sqlx::Error> {
        for _ in 0..self.max_batches {
            let mut tx = db.begin().await?;
            sqlx::query("SET LOCAL statement_timeout = '1000ms'")
                .execute(&mut *tx)
                .await?;
            let count = sqlx::query(PRUNE)
                .bind(self.batch_size)
                .execute(&mut *tx)
                .await?
                .rows_affected();
            tx.commit().await?;
            if count < self.batch_size as u64 {
                break;
            }
        }
        Ok(())
    }
}

fn value<T: FromStr>(key: &str, default: T, check: impl FnOnce(&T) -> bool) -> Result<T> {
    match env::var(key) {
        Ok(raw) => raw
            .trim()
            .parse()
            .ok()
            .filter(check)
            .ok_or_else(|| anyhow!("Invalid {key}")),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(env::VarError::NotUnicode(_)) => Err(anyhow!("Invalid {key}")),
    }
}

fn duration(seconds: f64, key: &str) -> Result<Duration> {
    Duration::try_from_secs_f64(seconds)
        .ok()
        .filter(|duration| !duration.is_zero())
        .ok_or_else(|| anyhow!("Invalid {key}"))
}

pub(super) fn start(db: PgPool, shutdown: CancellationToken) -> Result<()> {
    let settings = Settings::load()?;
    tokio::spawn(async move {
        loop {
            tokio::select! {
                biased;
                () = shutdown.cancelled() => break,
                () = sleep(settings.interval) => (),
            }
            tokio::select! {
                biased;
                () = shutdown.cancelled() => break,
                result = timeout(settings.budget, settings.sweep(&db)) => {
                    if !matches!(result, Ok(Ok(()))) {
                        tracing::warn!("Notification receipt maintenance deferred");
                    }
                }
            }
        }
    });
    Ok(())
}
