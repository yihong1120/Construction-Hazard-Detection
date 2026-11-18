//! Finite receiver/decoder parity probe. It intentionally hashes full BGR
//! frames and is not used to claim throughput of the production pipeline.
use anyhow::{Result, ensure};
use construction_pipeline::rtsp::{Source, Stats};
use sha2::{Digest, Sha256};
use std::{
    sync::{Arc, atomic::Ordering},
    time::Duration,
};
use tokio_util::sync::CancellationToken;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 6,
        "Usage: rtsp_probe URL FRAMES WIDTH HEIGHT libav|rust"
    );
    let handle = tokio::runtime::Handle::current();
    let count: usize = args[2].parse()?;
    let width: u32 = args[3].parse()?;
    let height: u32 = args[4].parse()?;
    ensure!(
        matches!(args[5].as_str(), "rust" | "libav"),
        "Invalid receiver mode"
    );
    let rust = args[5] == "rust";
    let result=tokio::task::spawn_blocking(move|| -> Result<_> {
        let stats=Arc::new(Stats::default());
        let mut decoder=Source::open(&handle,rust,&args[1],-1,CancellationToken::new(),Duration::from_secs(10),Arc::default(),stats.clone())?;
        let mut pixels=vec![0u8;width as usize*height as usize*3+64];let mut frames=vec![];
        for _ in 0..count {
            let decoded=decoder.next_frame()?.ok_or_else(||anyhow::anyhow!("Short source"))?;
            decoder.bgr(width,height,&mut pixels)?;
            frames.push(serde_json::json!({"pts":decoded.pts,"fps":decoded.fps,
                "sha256":hex::encode(Sha256::digest(&pixels[..pixels.len()-64]))}));
        }
        drop(decoder);
        Ok(serde_json::json!({"mode":if rust{"rust"}else{"libav"},"frames":frames,
            "ingress_frames":stats.frames.load(Ordering::Relaxed),"bytes":stats.bytes.load(Ordering::Relaxed),
            "lost_packets":stats.lost_packets.load(Ordering::Relaxed),
            "keyframe_dropped":stats.keyframe_dropped.load(Ordering::Relaxed)}))
    }).await??;
    println!("{}", result);
    Ok(())
}
