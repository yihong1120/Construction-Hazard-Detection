use std::{path::PathBuf, time::Duration};

use anyhow::Result;
use clap::{Parser, Subcommand};
use construction_api::{
    connect_database,
    website_access_gateway::{AgentConfig, snapshot},
};

#[derive(Parser)]
#[command(about = "Export and apply the construction website access policies")]
struct Args {
    #[command(subcommand)]
    mode: Mode,
}

#[derive(Subcommand)]
enum Mode {
    /// Export the three policies using the configured PostgreSQL role.
    Snapshot,
    /// Validate a policy export and print the proposed Nginx geo rules.
    Render {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        snapshot: PathBuf,
    },
    /// Apply valid policy changes through the last applied configuration.
    Sync {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        once: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    match Args::parse().mode {
        Mode::Snapshot => {
            let db = connect_database().await?;
            let policy = snapshot(&db).await?;
            println!("{policy}");
        }
        Mode::Render { config, snapshot } => {
            let content = AgentConfig::load(&config)?.render_file(&snapshot)?;
            print!("{content}");
        }
        Mode::Sync { config, once } => {
            let agent = AgentConfig::load(&config)?;
            if once {
                agent.sync_once().await?;
            } else {
                loop {
                    if let Err(error) = agent.sync_once().await {
                        eprintln!("Website access sync failed: {error}");
                    }
                    tokio::time::sleep(Duration::from_secs(5)).await;
                }
            }
        }
    }
    Ok(())
}
