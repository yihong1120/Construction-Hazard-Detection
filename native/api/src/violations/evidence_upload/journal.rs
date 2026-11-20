//! Write-ahead media journal shared with the application database schema.

use std::{
    fs::{self, File, OpenOptions},
    io::{self, Write as _},
    os::unix::fs::{DirBuilderExt, OpenOptionsExt},
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use axum::body::Bytes;
use chrono::{DateTime, NaiveDate, Utc};
use image::ImageFormat;
use sqlx::{PgConnection, PgPool};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use super::{Store, database_error, storage_error};
use crate::ApiError;

const DIRECTORY: &str = "rust-evidence";
const STAGING: &str = ".staging";

pub(super) struct Write {
    id: String,
    pub(super) path: String,
}

impl Write {
    pub(super) fn new(detection_time: DateTime<Utc>, format: ImageFormat) -> Self {
        let id = Uuid::new_v4().simple().to_string();
        let date = detection_time.date_naive();
        let extension = if format == ImageFormat::Png {
            "png"
        } else {
            "jpg"
        };
        Self {
            path: format!("{DIRECTORY}/{date}/{id}.{extension}"),
            id,
        }
    }

    pub(super) async fn begin(&self, tx: &mut PgConnection, db: &PgPool) -> Result<(), ApiError> {
        // Share the legacy maintenance lock. A separate committed journal row
        // survives crashes that abort the violation transaction.
        sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended($1, 4103))")
            .bind(&self.id)
            .execute(tx)
            .await
            .map_err(database_error)?;
        sqlx::query("INSERT INTO public.evidence_write_operations (id, path, created_at) VALUES ($1, $2, now())")
            .bind(&self.id).bind(&self.path).execute(db).await.map_err(database_error)?;
        Ok(())
    }

    pub(super) async fn persist(&self, media: Arc<Store>, bytes: Bytes) -> Result<(), ApiError> {
        let id = self.id.clone();
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || write_to_disk(media.root(), &id, &path, &bytes))
            .await
            .map_err(|_| storage_error())?
            .map_err(|error| {
                tracing::warn!(kind = ?error.kind(), "evidence storage write failed");
                storage_error()
            })
    }

    pub(super) async fn complete(&self, tx: &mut PgConnection) -> Result<(), ApiError> {
        let result = sqlx::query("DELETE FROM public.evidence_write_operations WHERE id = $1")
            .bind(&self.id)
            .execute(tx)
            .await
            .map_err(database_error)?;
        if result.rows_affected() != 1 {
            return Err(storage_error());
        }
        Ok(())
    }
}

fn checked_parent(root: &Path, path: &Path) -> io::Result<PathBuf> {
    let parent = path
        .parent()
        .ok_or(io::ErrorKind::InvalidInput)?
        .canonicalize()?;
    if !parent.starts_with(root) {
        return Err(io::ErrorKind::PermissionDenied.into());
    }
    Ok(parent)
}

fn sync(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
}

fn sync_parent(path: &Path) -> io::Result<()> {
    sync(path.parent().ok_or(io::ErrorKind::InvalidInput)?)
}

fn names(root: &Path, id: &str, path: &str) -> (PathBuf, PathBuf) {
    (
        root.join(STAGING).join(format!("{id}.part")),
        root.join(path),
    )
}

fn ensure_directory(root: &Path, child: &Path) -> io::Result<PathBuf> {
    match fs::DirBuilder::new().mode(0o750).create(child) {
        Ok(()) => (),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => (),
        Err(error) => return Err(error),
    }
    let path = child.canonicalize()?;
    if !path.starts_with(root) || !path.is_dir() {
        return Err(io::ErrorKind::PermissionDenied.into());
    }
    Ok(path)
}

fn write_to_disk(root: &Path, id: &str, path: &str, bytes: &[u8]) -> io::Result<()> {
    let parts: Vec<_> = path.split('/').collect();
    let [DIRECTORY, day, name] = parts.as_slice() else {
        return Err(io::ErrorKind::InvalidInput.into());
    };
    let staging_dir = ensure_directory(root, &root.join(STAGING))?;
    let evidence_dir = ensure_directory(root, &root.join(DIRECTORY))?;
    // Validate the container before creating its dated child, even if a
    // legacy deployment has left an unexpected symlink in the media root.
    let final_dir = ensure_directory(root, &evidence_dir.join(day))?;
    let staged = staging_dir.join(format!("{id}.part"));
    let final_file = final_dir.join(name);

    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o640)
        .open(&staged)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    // An atomic same-volume link refuses to replace any existing destination.
    // Recovery can discard the staged name if the machine stops after linking.
    fs::hard_link(&staged, &final_file)?;
    fs::remove_file(&staged)?;
    sync_parent(&staged)?;
    sync_parent(&final_file)?;
    sync_parent(&final_dir)?;
    sync(root)
}

pub(super) fn start(db: PgPool, media: Arc<Store>, stop: CancellationToken) {
    tokio::spawn(async move {
        let mut clock = tokio::time::interval(Duration::from_secs(30));
        clock.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                () = stop.cancelled() => return,
                _ = clock.tick() => {
                    if let Err(error) = recover_batch(&db, &media).await {
                        tracing::warn!(code = ?error.as_database_error().and_then(|error| error.code()),
                            "violation evidence recovery scan failed");
                    }
                }
            }
        }
    });
}

async fn recover_batch(db: &PgPool, media: &Arc<Store>) -> Result<(), sqlx::Error> {
    let ids: Vec<String> = sqlx::query_scalar(
        "SELECT id FROM public.evidence_write_operations
         WHERE path LIKE 'rust-evidence/%' AND created_at < now() - INTERVAL '60 seconds'
           AND next_attempt <= now()
         ORDER BY created_at LIMIT 32",
    )
    .fetch_all(db)
    .await?;
    for id in ids {
        if let Err(error) = recover_one(db, media, &id).await {
            tracing::warn!(kind = ?error.kind(), "violation evidence recovery failed");
            sqlx::query(
                "UPDATE public.evidence_write_operations SET attempts = attempts + 1,
                   last_error = 'rust-evidence-recovery', next_attempt = now() + INTERVAL '300 seconds'
                 WHERE id = $1",
            ).bind(&id).execute(db).await?;
        }
    }
    Ok(())
}

async fn recover_one(db: &PgPool, media: &Arc<Store>, id: &str) -> io::Result<()> {
    let other = |error: sqlx::Error| io::Error::other(error);
    let mut tx = db.begin().await.map_err(other)?;
    sqlx::query("SET LOCAL statement_timeout = '3s'")
        .execute(&mut *tx)
        .await
        .map_err(other)?;
    let locked: bool =
        sqlx::query_scalar("SELECT pg_try_advisory_xact_lock(hashtextextended($1, 4103))")
            .bind(id)
            .fetch_one(&mut *tx)
            .await
            .map_err(other)?;
    if !locked {
        return Ok(());
    }
    let Some(path) = sqlx::query_scalar::<_, String>(
        "SELECT path FROM public.evidence_write_operations WHERE id = $1 AND path LIKE 'rust-evidence/%'",
    ).bind(id).fetch_optional(&mut *tx).await.map_err(other)? else {
        return Ok(());
    };
    let referenced: bool =
        sqlx::query_scalar("SELECT EXISTS (SELECT 1 FROM violations WHERE image_path = $1)")
            .bind(&path)
            .fetch_one(&mut *tx)
            .await
            .map_err(other)?;
    let operation = id.to_owned();
    let media = media.clone();
    tokio::task::spawn_blocking(move || {
        remove_abandoned(media.root(), &operation, &path, referenced)
    })
    .await
    .map_err(io::Error::other)??;
    sqlx::query("DELETE FROM public.evidence_write_operations WHERE id = $1")
        .bind(id)
        .execute(&mut *tx)
        .await
        .map_err(other)?;
    tx.commit().await.map_err(other)
}

fn remove_abandoned(root: &Path, id: &str, path: &str, referenced: bool) -> io::Result<()> {
    let segments: Vec<_> = path.split('/').collect();
    let [DIRECTORY, date, file] = segments.as_slice() else {
        return Err(io::ErrorKind::InvalidInput.into());
    };
    if id.len() != 32
        || Uuid::parse_str(id).is_err()
        || date.parse::<NaiveDate>().is_err()
        || !file
            .strip_prefix(id)
            .is_some_and(|suffix| matches!(suffix, ".jpg" | ".png"))
    {
        return Err(io::ErrorKind::InvalidInput.into());
    }
    let (staged, final_file) = names(root, id, path);
    remove(root, &staged)?;
    if !referenced {
        remove(root, &final_file)?;
    }
    Ok(())
}

fn remove(root: &Path, path: &Path) -> io::Result<()> {
    let checked = match checked_parent(root, path) {
        Ok(parent) => parent.join(path.file_name().ok_or(io::ErrorKind::InvalidInput)?),
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    match fs::remove_file(&checked) {
        Ok(()) => sync_parent(&checked),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    #[test]
    fn staging_rejects_an_escaped_destination_before_creating_directories() {
        let media = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        symlink(outside.path(), media.path().join(DIRECTORY)).unwrap();
        let operation = Write::new(Utc::now(), ImageFormat::Jpeg);
        let error =
            write_to_disk(media.path(), &operation.id, &operation.path, b"jpeg").unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
        assert!(outside.path().read_dir().unwrap().next().is_none());
    }

    #[test]
    fn promotion_does_not_replace_an_existing_file() {
        let media = tempfile::tempdir().unwrap();
        let operation = Write::new(Utc::now(), ImageFormat::Png);
        write_to_disk(media.path(), &operation.id, &operation.path, b"first").unwrap();
        let error =
            write_to_disk(media.path(), &operation.id, &operation.path, b"duplicate").unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        let (part, file) = names(media.path(), &operation.id, &operation.path);
        assert_eq!(fs::read(&file).unwrap(), b"first");
        remove_abandoned(media.path(), &operation.id, &operation.path, true).unwrap();
        assert!(!part.exists());
        assert_eq!(fs::read(&file).unwrap(), b"first");
    }
}
