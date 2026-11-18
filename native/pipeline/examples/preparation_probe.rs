//! Mixed-shape, pinned-slot and tensor parity gate. Not a timing benchmark.
use anyhow::{Result, ensure};
use construction_inference::{
    core::{Error, Image},
    engine::NativeEngine,
};
use construction_media::Decoder;
use std::{
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};
use tokio_util::sync::CancellationToken;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 4, "preparation_probe ENGINE 1080P 720P");
    let mut decoders = vec![
        Decoder::open(&a[2], CancellationToken::new(), Duration::from_secs(10))?,
        Decoder::open(&a[3], CancellationToken::new(), Duration::from_secs(10))?,
    ];
    let shapes = vec![(1080, 1920), (720, 1280)];
    let mut hosts = vec![];
    for (d, &(h, w)) in decoders.iter_mut().zip(&shapes) {
        ensure!(d.next_frame()?.is_some(), "Short fixture");
        hosts.push(d.retain_host(w as u32, h as u32)?);
    }
    drop(decoders);
    let mut engine = NativeEngine::new(&a[1], 8, 640, 0, false, 0.25, true)?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build()?;
    let fused = AtomicUsize::new(0);
    let fallback = AtomicUsize::new(0);
    let fill = |i: usize, l: &construction_inference::core::Layout, target: &mut [u8]| {
        if hosts[i].letterbox(target, l.h, l.w, l.rh, l.rw, l.top, l.left) {
            fused.fetch_add(1, Ordering::Relaxed);
            Ok(1)
        } else {
            fallback.fetch_add(1, Ordering::Relaxed);
            let data = hosts[i]
                .bgr_padded()
                .map_err(|e| Error::Runtime(e.to_string()))?;
            NativeEngine::fill_letterbox(
                &Image {
                    h: shapes[i].0,
                    w: shapes[i].1,
                    data: &data[..shapes[i].0 * shapes[i].1 * 3],
                },
                l,
                target,
            )?;
            Ok(0)
        }
    };
    let (batch, count) = engine.prepare_surfaces_each(&shapes, Some(&pool), fill)?;
    ensure!(
        count == 1 && fused.load(Ordering::Relaxed) == 1 && fallback.load(Ordering::Relaxed) == 1,
        "Per-frame fallback repeated work"
    );
    let reference = engine.submit_prepared(batch)?;
    let expected = engine.inspect_input(reference)?;
    let expected_rows = engine.collect(reference)?;
    // Errors and panic must both release the reservation before another fill.
    ensure!(
        engine
            .submit_surfaces_pinned(&shapes, None, |_, _, _| Err(Error::Runtime(
                "injected".into()
            )))
            .is_err()
    );
    ensure!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = engine
                .submit_surfaces_pinned(&shapes, None, |_, _, _| panic!("injected fill panic"));
        }))
        .is_err()
    );
    let constant = |_: usize, _: &construction_inference::core::Layout, target: &mut [u8]| {
        target.fill(227);
        Ok(0)
    };
    let (constant_batch, _) = engine.prepare_surfaces_each(&shapes, None, constant)?;
    let constant_ticket = engine.submit_prepared(constant_batch)?;
    let constant_tensor = engine.inspect_input(constant_ticket)?;
    let constant_rows = engine.collect(constant_ticket)?;
    ensure!(
        constant_tensor != expected,
        "Distinct-slot fixture is not distinct"
    );
    for _ in 0..4 {
        let (a, count) = engine.submit_surfaces_pinned(&shapes, Some(&pool), fill)?;
        ensure!(count == 1, "Mixed batch lost fused input");
        let (b, _) = engine.submit_surfaces_pinned(&shapes, Some(&pool), constant)?;
        ensure!(
            engine.submit_surfaces_pinned(&shapes, None, fill).is_err(),
            "Third occupied slot reused"
        );
        ensure!(
            engine
                .submit_surfaces_pinned(&shapes[..1], None, fill)
                .is_err(),
            "Pending storage reshaped"
        );
        for (ticket, reference, rows) in [
            (a, &expected, &expected_rows),
            (b, &constant_tensor, &constant_rows),
        ] {
            let actual = engine.inspect_input(ticket)?;
            ensure!(
                actual
                    .iter()
                    .zip(reference)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "Pinned tensor differs"
            );
            ensure!(
                &engine.collect(ticket)? == rows,
                "Pinned predictions differ"
            );
        }
    }
    let images: Vec<_> = hosts
        .iter()
        .zip(&shapes)
        .map(|(f, &(h, w))| {
            Ok(Image {
                h,
                w,
                data: &f.bgr_padded()?[..h * w * 3],
            })
        })
        .collect::<Result<_>>()?;
    let ordinary = engine.submit_images(&images)?;
    ensure!(
        engine
            .inspect_input(ordinary)?
            .iter()
            .zip(&expected)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "Mixed fallback differs from ordinary BGR reference"
    );
    ensure!(
        engine.collect(ordinary)? == expected_rows,
        "Mixed predictions differ"
    );
    println!(
        "{}",
        serde_json::json!({"mixed_fused":1,"mixed_fallback":1,"tensor_elements":expected.len(),"pending_cycles":4,"distinct_slot_inputs":true,
        "error_and_panic_recovery":true,"lookup_bytes":hosts[0].lookup_bytes(),"parity":true})
    );
    Ok(())
}
