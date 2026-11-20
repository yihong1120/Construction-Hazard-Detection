use std::{
    io,
    path::{Component, Path, PathBuf},
    time::Duration,
};

use sqlx::PgPool;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

type Job = (i64, String, i32);

pub(crate) async fn start(
    db: &PgPool,
    root: &Path,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    let available: bool =
        sqlx::query_scalar("SELECT to_regclass('site_media_cleanup_jobs') IS NOT NULL")
            .fetch_one(db)
            .await?;
    if available {
        tokio::spawn(run(db.clone(), root.to_owned(), shutdown));
    }
    Ok(())
}

async fn run(db: PgPool, root: PathBuf, shutdown: CancellationToken) {
    let mut interval = tokio::time::interval(Duration::from_secs(15));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            _ = shutdown.cancelled() => return,
            _ = interval.tick() => {}
        }
        if let Err(error) = drain(&db, &root, &shutdown).await {
            tracing::warn!(code = ?error.as_database_error().and_then(|error| error.code()), "site media cleanup unavailable");
        }
    }
}

async fn drain(db: &PgPool, root: &Path, shutdown: &CancellationToken) -> Result<(), sqlx::Error> {
    let lease = Uuid::new_v4().hyphenated().to_string();
    let jobs: Vec<Job> = sqlx::query_as(
        "WITH pending AS (
            SELECT id FROM site_media_cleanup_jobs WHERE completed_at IS NULL
              AND (lease_expires_at IS NULL OR lease_expires_at<=now())
            ORDER BY id LIMIT 32 FOR UPDATE SKIP LOCKED
         ) UPDATE site_media_cleanup_jobs jobs
            SET attempt_count=jobs.attempt_count+1,lease_token=$1,lease_expires_at=now()+interval '2 minutes'
         FROM pending WHERE jobs.id=pending.id RETURNING jobs.id,jobs.path,jobs.attempt_count",
    ).bind(&lease).fetch_all(db).await?;
    for (id, path, attempt) in jobs {
        if shutdown.is_cancelled() {
            break;
        }
        let referenced: bool = sqlx::query_scalar(
            "SELECT EXISTS (SELECT 1 FROM violations v WHERE v.image_path=$1
                AND (v.site_id IS NOT NULL OR EXISTS (SELECT 1 FROM sites WHERE name=v.site)))",
        )
        .bind(&path)
        .fetch_one(db)
        .await?;
        let deleted = if referenced {
            true
        } else {
            let media_root = root.to_owned();
            tokio::task::spawn_blocking(move || remove_media(&media_root, &path))
                .await
                .is_ok_and(|outcome| outcome.is_ok())
        };
        if deleted {
            sqlx::query(
                "UPDATE site_media_cleanup_jobs SET completed_at=now(),last_error=NULL,
                lease_token=NULL,lease_expires_at=NULL WHERE id=$1 AND lease_token=$2",
            )
            .bind(id)
            .bind(&lease)
            .execute(db)
            .await?;
        } else {
            let delay = (1_i32 << attempt.clamp(1, 12)).min(3600);
            sqlx::query(
                "UPDATE site_media_cleanup_jobs SET last_error='Could not remove evidence media',
                lease_token=NULL,lease_expires_at=now()+make_interval(secs => $3::float8)
                WHERE id=$1 AND lease_token=$2",
            )
            .bind(id)
            .bind(&lease)
            .bind(f64::from(delay))
            .execute(db)
            .await?;
            tracing::warn!(job_id = id, "could not remove site evidence media");
        }
    }
    Ok(())
}

fn remove_media(root: &Path, name: &str) -> io::Result<()> {
    let path = Path::new(name);
    let relative = if path.is_absolute() {
        path.strip_prefix(root)
            .map_err(|_| io::ErrorKind::PermissionDenied)?
    } else {
        path
    };
    if relative.as_os_str().is_empty()
        || relative
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(io::ErrorKind::PermissionDenied.into());
    }
    let candidate = root.join(relative);
    let canonical = match candidate.canonicalize() {
        Ok(path) => path,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    // The queued path must resolve within the evidence directory. Remove the
    // queued entry, so a file symlink cannot remove its target.
    if !canonical.starts_with(root) || !canonical.metadata()?.is_file() {
        return Err(io::ErrorKind::PermissionDenied.into());
    }
    match std::fs::remove_file(candidate) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        result => result,
    }
}
