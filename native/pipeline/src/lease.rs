//! Compatible with src/gpu_resources.py and the postprocess Rust lease protocol.
use anyhow::{Context, Result, ensure};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    os::{fd::AsRawFd, unix::fs::OpenOptionsExt},
    path::Path,
};

pub struct Lease {
    files: Vec<File>,
}
impl Lease {
    pub fn fds(&self) -> Vec<i32> {
        self.files.iter().map(AsRawFd::as_raw_fd).collect()
    }
}
fn open(path: &Path) -> Result<File> {
    Ok(OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .mode(0o600)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)?)
}
fn lock(file: &File) -> Result<bool> {
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
        return Ok(true);
    }
    let e = std::io::Error::last_os_error();
    if e.kind() == std::io::ErrorKind::WouldBlock {
        Ok(false)
    } else {
        Err(e.into())
    }
}
pub fn acquire(root: &Path, key: &str, capacity: usize, units: usize) -> Result<Option<Lease>> {
    ensure!(
        !key.is_empty()
            && key
                .bytes()
                .all(|v| v.is_ascii_alphanumeric() || b"._-".contains(&v)),
        "Invalid lease key"
    );
    ensure!(
        units > 0 && units <= capacity && capacity <= 16384,
        "Invalid resource capacity"
    );
    std::fs::create_dir_all(root)?;
    let mut control = open(&root.join(format!("{key}.control")))?;
    ensure!(
        unsafe { libc::flock(control.as_raw_fd(), libc::LOCK_EX) } == 0,
        "Resource allocator lock failed"
    );
    let mut text = String::new();
    Read::by_ref(&mut control)
        .take(32)
        .read_to_string(&mut text)?;
    let old = if text.trim().is_empty() {
        capacity
    } else {
        text.trim()
            .parse::<usize>()
            .context("Corrupt resource capacity")?
    };
    ensure!((1..=16384).contains(&old), "Corrupt resource capacity");
    let mut files = Vec::new();
    if old != capacity {
        for index in 0..old {
            let file = open(&root.join(format!("{key}.{index}")))?;
            ensure!(lock(&file)?, "GPU budget differs from active processes");
            files.push(file);
        }
        files.clear();
    }
    control.seek(SeekFrom::Start(0))?;
    control.set_len(0)?;
    write!(control, "{capacity}")?;
    for index in 0..capacity {
        let file = open(&root.join(format!("{key}.{index}")))?;
        if lock(&file)? {
            files.push(file);
            if files.len() == units {
                return Ok(Some(Lease { files }));
            }
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partial_reservation_rolls_back_and_drop_recovers() {
        let dir = tempfile::tempdir().unwrap();
        let a = acquire(dir.path(), "gpu0.memory", 3, 2).unwrap().unwrap();
        assert!(acquire(dir.path(), "gpu0.memory", 3, 2).unwrap().is_none());
        let b = acquire(dir.path(), "gpu0.memory", 3, 1).unwrap().unwrap();
        assert!(acquire(dir.path(), "gpu0.memory", 4, 1).is_err());
        drop(a);
        drop(b);
        assert!(acquire(dir.path(), "gpu0.memory", 4, 4).unwrap().is_some());
    }
}
