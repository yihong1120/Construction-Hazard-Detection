use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, OpenOptions},
    io::Write,
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};

use anyhow::{Context, Result, ensure};
use chrono::{SecondsFormat, Utc};
use serde::Deserialize;
use serde_json::{Value, json};
use sqlx::PgPool;
use tokio::{io::AsyncReadExt, process::Command, time::timeout};
use uuid::Uuid;

use super::{
    Hosts, MAX_RULES, Policy, Receipt, Settings, SiteKey, canonical_network, load_catalog,
    validate_hosts,
};

const MAX_SNAPSHOT_BYTES: u64 = 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentConfig {
    hosts: Hosts,
    reader: Vec<String>,
    output: PathBuf,
    last_good: PathBuf,
    status: PathBuf,
}

impl AgentConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let config: Self = serde_json::from_reader(
            fs::File::open(path).context("Open website agent configuration")?,
        )
        .context("Read website agent configuration")?;
        validate_hosts(&config.hosts)?;
        ensure!(
            config
                .reader
                .first()
                .is_some_and(|item| Path::new(item).is_absolute()),
            "Website policy reader needs an absolute command"
        );
        let paths = [&config.output, &config.last_good, &config.status];
        ensure!(
            paths
                .iter()
                .all(|path| path.is_absolute() && path.file_name().is_some())
                && paths.into_iter().collect::<BTreeSet<_>>().len() == paths.len(),
            "Website agent paths must be distinct absolute files"
        );
        Ok(config)
    }

    pub fn render_file(&self, path: &Path) -> Result<String> {
        let file = fs::File::open(path).context("Open website policy snapshot")?;
        ensure!(
            file.metadata()?.len() <= MAX_SNAPSHOT_BYTES,
            "Website policy snapshot too large"
        );
        let snapshot = serde_json::from_reader(file).context("Read website policy snapshot")?;
        let (nginx, _) = render(snapshot, &self.hosts)?;
        Ok(nginx)
    }

    pub async fn sync_once(&self) -> Result<()> {
        self.sync_once_with(apply_nginx(), reload_nginx()).await
    }

    async fn sync_once_with(
        &self,
        activate: impl Future<Output = Result<()>>,
        restore: impl Future<Output = Result<()>>,
    ) -> Result<()> {
        let data = read_snapshot(&self.reader).await?;
        let snapshot = serde_json::from_slice(&data).context("Parse website policy snapshot")?;
        let (nginx, versions) = render(snapshot, &self.hosts)?;
        let previous =
            fs::read_to_string(&self.last_good).context("Read last applied website policy")?;
        if nginx != previous
            || fs::read_to_string(&self.output).ok().as_deref() != Some(nginx.as_str())
        {
            atomic_write(&self.output, nginx.as_bytes()).context("Stage website policy")?;
            if let Err(error) = activate.await {
                atomic_write(&self.output, previous.as_bytes())
                    .context("Restore previous website policy")?;
                restore
                    .await
                    .context("Restore previous Nginx configuration")?;
                return Err(error);
            }
            atomic_write(&self.last_good, nginx.as_bytes())
                .context("Record applied website policy")?;
        }
        let receipt = Receipt {
            checked_at: Utc::now().to_rfc3339_opts(SecondsFormat::Micros, false),
            versions,
            hosts: self.hosts.clone(),
        };
        let mut receipt = serde_json::to_vec(&receipt)?;
        receipt.push(b'\n');
        atomic_write(&self.status, &receipt).context("Record website gateway status")
    }
}

pub async fn snapshot(db: &PgPool) -> Result<Value> {
    let config = Settings::from_env()?;
    // One PostgreSQL statement observes a consistent catalog and rule snapshot.
    let policies = load_catalog(db)
        .await
        .map_err(|_| anyhow::anyhow!("Website policy catalog unavailable"))?;
    Ok(
        json!({ "hosts": config.hosts, "policies": policies.iter().map(Policy::snapshot).collect::<Vec<_>>() }),
    )
}

#[derive(Deserialize)]
struct Snapshot {
    hosts: Hosts,
    policies: Vec<NetworkPolicy>,
}

#[derive(Deserialize)]
struct NetworkPolicy {
    site_key: SiteKey,
    restricted: bool,
    version: i32,
    rules: Vec<NetworkRule>,
}

#[derive(Deserialize)]
struct NetworkRule {
    network: String,
    enabled: bool,
}

fn render(data: Snapshot, expected_hosts: &Hosts) -> Result<(String, BTreeMap<SiteKey, i32>)> {
    ensure!(
        &data.hosts == expected_hosts,
        "Website policy belongs to another deployment"
    );
    let mut policies = data.policies;
    policies.sort_unstable_by_key(|policy| policy.site_key);
    ensure!(
        policies
            .iter()
            .map(|policy| policy.site_key)
            .eq(SiteKey::ALL),
        "Incomplete website catalog"
    );

    let mut content =
        String::from("# Managed by website-access-agent; numeric IP networks only.\n");
    let mut versions = BTreeMap::new();
    use std::fmt::Write as _;
    for policy in policies {
        ensure!(
            policy.version >= 0 && policy.rules.len() <= MAX_RULES,
            "Invalid website policy state"
        );
        versions.insert(policy.site_key, policy.version);
        let mut active = BTreeSet::new();
        for rule in policy.rules {
            let network =
                canonical_network(&rule.network, true).context("Invalid website network")?;
            if rule.enabled {
                active.insert(network.to_string());
            }
        }
        ensure!(
            !policy.restricted || !active.is_empty(),
            "Empty active website allowlist"
        );
        writeln!(
            content,
            "geo $vn_website_{}_allowed {{",
            policy.site_key.as_str()
        )?;
        writeln!(content, "    default {};", u8::from(!policy.restricted))?;
        if policy.restricted {
            for network in active {
                writeln!(content, "    {network} 1;")?;
            }
        }
        content.push_str("}\n");
    }
    Ok((content, versions))
}

async fn read_snapshot(reader: &[String]) -> Result<Vec<u8>> {
    let mut child = Command::new(&reader[0])
        .args(&reader[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .context("Start website policy reader")?;
    let mut pipe = child
        .stdout
        .take()
        .context("Open website policy reader output")?
        .take(MAX_SNAPSHOT_BYTES + 1);
    timeout(Duration::from_secs(20), async move {
        let mut data = Vec::new();
        pipe.read_to_end(&mut data)
            .await
            .context("Read website policy snapshot")?;
        ensure!(
            data.len() <= MAX_SNAPSHOT_BYTES as usize,
            "Website policy snapshot too large"
        );
        ensure!(
            child.wait().await?.success(),
            "Website policy reader failed"
        );
        Ok(data)
    })
    .await
    .context("Website policy reader timed out")?
}

fn atomic_write(path: &Path, content: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .context("Website policy path needs a parent directory")?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(".website-access-{}", Uuid::new_v4()));
    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o644)
            .open(&temporary)?;
        file.write_all(content)?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        fs::File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temporary);
    }
    result
}

async fn apply_nginx() -> Result<()> {
    check_command("/usr/sbin/nginx", &["-t"], Duration::from_secs(10))
        .await
        .context("Check Nginx policy")?;
    reload_nginx().await
}

async fn reload_nginx() -> Result<()> {
    check_command(
        "/usr/bin/systemctl",
        &["reload", "nginx"],
        Duration::from_secs(15),
    )
    .await
    .context("Reload Nginx policy")
}

async fn check_command(binary: &str, args: &[&str], deadline: Duration) -> Result<()> {
    let status = timeout(
        deadline,
        Command::new(binary)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .status(),
    )
    .await
    .context("Website gateway command timed out")??;
    ensure!(status.success(), "Website gateway command failed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    #[tokio::test]
    async fn failed_activation_restores_previous_config_and_keeps_receipt() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path();
        let hosts = BTreeMap::from([
            (SiteKey::Ai, "ai.example.test".to_owned()),
            (SiteKey::Main, "main.example.test".to_owned()),
            (SiteKey::Wiki, "wiki.example.test".to_owned()),
        ]);
        let snapshot = path.join("snapshot.json");
        fs::write(&snapshot, json!({"hosts": hosts, "policies": [
            {"site_key":"ai","version":0,"restricted":false,"rules":[]},
            {"site_key":"main","version":2,"restricted":true,"rules":[{"network":"192.0.2.0/24","enabled":true}]},
            {"site_key":"wiki","version":0,"restricted":false,"rules":[]},
        ]}).to_string()).unwrap();
        let config = AgentConfig {
            hosts,
            reader: vec!["/usr/bin/cat".to_owned(), snapshot.display().to_string()],
            output: path.join("nginx.conf"),
            last_good: path.join("previous.conf"),
            status: path.join("status.json"),
        };
        fs::write(&config.last_good, "last applied").unwrap();
        fs::write(&config.output, "last applied").unwrap();
        fs::write(&config.status, "prior receipt").unwrap();
        let restored = Cell::new(false);
        let activation = async {
            assert!(
                fs::read_to_string(&config.output)
                    .unwrap()
                    .contains("192.0.2.0/24 1;")
            );
            anyhow::bail!("reload failed")
        };
        let restore = async {
            assert_eq!(fs::read_to_string(&config.output).unwrap(), "last applied");
            restored.set(true);
            Ok(())
        };
        assert!(config.sync_once_with(activation, restore).await.is_err());
        assert!(restored.get());
        assert_eq!(
            fs::read_to_string(&config.last_good).unwrap(),
            "last applied"
        );
        assert_eq!(fs::read_to_string(&config.status).unwrap(), "prior receipt");
    }
}
