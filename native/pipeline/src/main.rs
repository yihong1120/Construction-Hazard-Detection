use anyhow::Result;
use clap::{Parser, Subcommand};
use construction_pipeline::{config::Config, runtime};
use std::path::PathBuf;
use tokio_util::sync::CancellationToken;

#[derive(Parser)]
#[command(about = "Standalone Rust construction-camera pipeline (experimental)")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}
#[derive(Subcommand)]
enum Commands {
    /// Create a finite, local-output experiment without a Python setup script.
    Init {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        engine: PathBuf,
        #[arg(long)]
        source: String,
        #[arg(long)]
        state_dir: PathBuf,
        #[arg(long)]
        version: String,
        #[arg(long, default_value_t = 120)]
        max_frames: u64,
    },
    Analyze {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Validate {
        #[arg(long)]
        config: PathBuf,
    },
    Run {
        #[arg(long)]
        config: PathBuf,
    },
    Dispatch {
        #[arg(long)]
        config: PathBuf,
    },
    Outbox {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        requeue_failed: bool,
        /// Import state_dir/outbox.sqlite3 without modifying the legacy file.
        #[arg(long)]
        migrate_sqlite: bool,
    },
}
#[tokio::main]
async fn main() -> Result<()> {
    unsafe {
        libc::umask(0o077);
    }
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .json()
        .with_writer(std::io::stderr)
        .init();
    match Cli::parse().command {
        Commands::Init {
            config,
            engine,
            source,
            state_dir,
            version,
            max_frames,
        } => {
            use sha2::{Digest, Sha256};
            use std::io::Write;
            let engine = engine.canonicalize()?;
            let state_dir = std::path::absolute(state_dir)?;
            let source = if ["rtsp://", "rtsps://", "http://", "https://"]
                .iter()
                .any(|prefix| source.starts_with(prefix))
            {
                source
            } else {
                PathBuf::from(source)
                    .canonicalize()?
                    .to_str()
                    .ok_or_else(|| anyhow::anyhow!("Non-UTF8 source path"))?
                    .to_owned()
            };
            let (width, height) = construction_pipeline::catalog::probe(&source).await?;
            let settings: Config = serde_json::from_value(serde_json::json!({
                "state_dir":state_dir,"lossless":true,"max_frames":max_frames,
                "models":[{"key":"default","artifact":engine,"sha256":hex::encode(Sha256::digest(std::fs::read(&engine)?)),"version":version}],
                "streams":[{"id":"camera-1","site_id":1,"site":"experiment","name":"camera-1","source":source,"model":"default",
                    "width":width,"height":height,"languages":["en","zh-TW"],
                    "output":{"clean":state_dir.join("clean.mp4"),"annotated":state_dir.join("annotated_{language}.mp4")}}]
            }))?;
            settings.validate()?;
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(config)?;
            file.write_all(&serde_json::to_vec_pretty(&settings)?)?;
            file.sync_all()?;
            println!(
                "{}",
                serde_json::json!({"created":true,"width":width,"height":height,"frames":max_frames})
            );
        }
        Commands::Analyze { input, output } => {
            construction_pipeline::analysis::run(&input, &output)?
        }
        Commands::Validate { config } => {
            let config = Config::load(&config)?;
            println!(
                "{}",
                serde_json::json!({"valid":true,"cameras":config.streams.len(),"models":config.models.len(),"replay":config.experimental_replay})
            );
        }
        Commands::Outbox {
            config,
            requeue_failed,
            migrate_sqlite,
        } => {
            let config = Config::load(&config)?;
            let store = construction_pipeline::outbox::Outbox::open(&config).await?;
            if migrate_sqlite {
                let _owner =
                    construction_pipeline::lease::acquire(&config.state_dir, "pipeline", 1, 1)?
                        .ok_or_else(|| {
                            anyhow::anyhow!("Stop the previous pipeline before SQLite import")
                        })?;
                println!(
                    "imported {}",
                    store
                        .migrate_sqlite(&config.state_dir.join("outbox.sqlite3"))
                        .await?
                );
            }
            if requeue_failed {
                println!("requeued {}", store.requeue_failed().await?);
            }
            println!("{}", store.counts().await?);
        }
        Commands::Dispatch { config } => {
            let config = Config::load(&config)?;
            anyhow::ensure!(config.delivery.enabled, "Delivery is disabled");
            let store = construction_pipeline::outbox::Outbox::open(&config).await?;
            let stop = CancellationToken::new();
            let token = stop.clone();
            tokio::spawn(async move {
                let mut term =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                        .expect("signal handler");
                tokio::select! {_=tokio::signal::ctrl_c()=>(),_=term.recv()=>()};
                token.cancel();
            });
            construction_pipeline::delivery::dispatch(store, config.delivery, stop).await?;
        }
        Commands::Run { config } => {
            let config = Config::load(&config)?;
            let stop = CancellationToken::new();
            let token = stop.clone();
            tokio::spawn(async move {
                let mut term =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                        .expect("signal handler");
                tokio::select! {_ = tokio::signal::ctrl_c()=>(),_ = term.recv()=>()};
                token.cancel();
            });
            let report = runtime::run(config, stop).await?;
            println!("{}", serde_json::to_string(&report)?);
        }
    }
    Ok(())
}
