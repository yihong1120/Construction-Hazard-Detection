//! CPU YUV ownership and fused input parity, not a throughput benchmark.
use anyhow::{Result, ensure};
use construction_inference::{
    core::{Image, layout},
    resize,
};
use construction_media::{Decoder, GpuMetrics};
use sha2::{Digest, Sha256};
use std::{
    sync::{Arc, atomic::Ordering},
    time::Duration,
};
use tokio_util::sync::CancellationToken;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 5, "host_probe SOURCE FRAMES WIDTH HEIGHT");
    let (count, w, h): (usize, u32, u32) = (a[2].parse()?, a[3].parse()?, a[4].parse()?);
    let stats = Arc::new(GpuMetrics::default());
    let mut decoder = Decoder::open_device(
        &a[1],
        -1,
        CancellationToken::new(),
        Duration::from_secs(10),
        stats.clone(),
    )?;
    let l = layout(h as usize, w as usize, 640, 32, false);
    let mut reference = vec![0; w as usize * h as usize * 3 + 64];
    let mut expected = vec![0; l.h * l.w * 3];
    let mut actual = expected.clone();
    let mut rows = vec![];
    let mut retained = None;
    for i in 0..count {
        ensure!(decoder.next_frame()?.is_some(), "Short source");
        let host = decoder.retain_host(w, h)?;
        decoder.bgr(w, h, &mut reference)?;
        let supported = resize::letterbox(
            &Image {
                h: h as usize,
                w: w as usize,
                data: &reference[..reference.len() - 64],
            },
            &l,
            &mut expected,
        );
        let fused = host.letterbox(&mut actual, l.h, l.w, l.rh, l.rw, l.top, l.left);
        ensure!(fused == supported, "Geometry eligibility differs");
        if fused {
            ensure!(expected == actual, "Model input differs on frame {i}");
        }
        ensure!(
            host.bgr_padded()?[..reference.len() - 64] == reference[..reference.len() - 64],
            "Full BGR differs"
        );
        rows.push(serde_json::json!({"frame":i,"fused":fused,"sha256":hex::encode(Sha256::digest(&actual))}));
        if retained.is_none() {
            retained = Some((
                decoder.retain_host(w, h)?,
                reference.clone(),
                if supported {
                    Some(expected.clone())
                } else {
                    None
                },
            ));
        }
    }
    drop(decoder);
    let (owner, first, model) = retained.unwrap();
    ensure!(
        owner.bgr_padded()? == first,
        "Decoder Drop invalidated retained frame"
    );
    if let Some(model) = model {
        let mut fresh = vec![0; model.len()];
        ensure!(owner.letterbox(&mut fresh, l.h, l.w, l.rh, l.rw, l.top, l.left));
        ensure!(
            fresh == model,
            "Decoder Drop invalidated retained YUV planes"
        );
    }
    println!(
        "{}",
        serde_json::json!({"frames":rows,"host_retained":stats.host_retained.load(Ordering::Relaxed),"host_sampled":stats.host_sampled.load(Ordering::Relaxed)})
    );
    Ok(())
}
