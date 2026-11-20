use std::{net::SocketAddr, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use construction_api::{Config, app, connect_database, init_database};
use tokio::net::TcpListener;
use tokio::signal::unix::{SignalKind, signal};
use tokio_util::sync::CancellationToken;

#[derive(Parser)]
#[command(about = "Rust HTTP API for the isolated Construction Hazard Detection release")]
struct Args {
    #[arg(long)]
    config: PathBuf,
    /// Initialise the isolated evidence schema and exit.
    #[arg(long)]
    init_db: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let args = Args::parse();
    let config: Config =
        serde_json::from_slice(&std::fs::read(&args.config).context("Read API configuration")?)
            .context("Parse API configuration")?;
    config.validate()?;
    let db = connect_database().await?;
    if args.init_db {
        return init_database(&db).await;
    }
    let listen = config.listen;
    let shutdown = CancellationToken::new();
    let app = app(config, db, shutdown.clone()).await?;
    let listener = TcpListener::bind(listen).await?;
    let mut sigterm = signal(SignalKind::terminate())?;
    tracing::info!(address = %listener.local_addr()?, "Rust API listening");
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(async move {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {},
            _ = sigterm.recv() => {},
        }
        shutdown.cancel();
    })
    .await?;
    Ok(())
}
