//! Pixel compatibility and native-handle lifetime gates for in-process media.
use construction_media::{Decoder, Encoder, GpuMetrics, live_handles};
use std::{
    path::Path,
    sync::{Arc, Mutex, atomic::Ordering},
    time::{Duration, Instant},
};
use tokio_util::sync::CancellationToken;
static SERIAL: Mutex<()> = Mutex::new(());
fn source() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/videos/test.mp4")
        .canonicalize()
        .unwrap()
}

#[test]
fn retained_yuv_half_scale_matches_bgr_average_without_materialising_a_frame() {
    let _guard = SERIAL.lock().unwrap();
    let metrics = Arc::new(GpuMetrics::default());
    let mut decoder = Decoder::open_device(
        source().to_str().unwrap(),
        -1,
        CancellationToken::new(),
        Duration::from_secs(5),
        metrics.clone(),
    )
    .unwrap();
    const WIDTH: usize = 640;
    const HEIGHT: usize = 352;
    const OUT_WIDTH: usize = WIDTH / 2 + 16;
    const OUT_HEIGHT: usize = HEIGHT / 2 + 16;
    let mut reference = vec![0; WIDTH * HEIGHT * 3 + 64];
    let mut target = vec![0; OUT_WIDTH * OUT_HEIGHT * 3];
    decoder.next_frame().unwrap().unwrap();
    let host = decoder.retain_host(WIDTH as u32, HEIGHT as u32).unwrap();
    decoder
        .bgr(WIDTH as u32, HEIGHT as u32, &mut reference)
        .unwrap();
    drop(decoder);

    assert!(host.letterbox(
        &mut target,
        OUT_HEIGHT,
        OUT_WIDTH,
        HEIGHT / 2,
        WIDTH / 2,
        8,
        8,
    ));
    assert_eq!(metrics.host_materialized.load(Ordering::Relaxed), 0);
    let mut expected = vec![114; target.len()];
    for row in 0..HEIGHT / 2 {
        for column in 0..WIDTH / 2 {
            let source = (2 * row * WIDTH + 2 * column) * 3;
            let output = ((row + 8) * OUT_WIDTH + column + 8) * 3;
            for channel in 0..3 {
                let position = source + channel;
                let sum: u16 = [
                    position,
                    position + 3,
                    position + WIDTH * 3,
                    position + WIDTH * 3 + 3,
                ]
                .iter()
                .map(|&offset| u16::from(reference[offset]))
                .sum();
                expected[output + channel] = ((sum + 2) / 4) as u8;
            }
        }
    }
    assert_eq!(target, expected);
    assert_eq!(live_handles(), (0, 0));
}

#[test]
fn native_decode_matches_legacy_ffmpeg_bgr_and_encoder_finishes_every_frame() {
    let _guard = SERIAL.lock().unwrap();
    for (width, height) in [(640, 352), (1920, 1080)] {
        let reference = std::process::Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-threads",
                "1",
                "-filter_threads",
                "1",
                "-i",
            ])
            .arg(source())
            .args([
                "-vf",
                &format!("scale={width}:{height}:flags=bilinear"),
                "-frames:v",
                "8",
                "-pix_fmt",
                "bgr24",
                "-f",
                "rawvideo",
                "pipe:1",
            ])
            .output()
            .unwrap();
        assert!(reference.status.success());
        let mut decoder = Decoder::open(
            source().to_str().unwrap(),
            CancellationToken::new(),
            Duration::from_secs(5),
        )
        .unwrap();
        let bytes = width as usize * height as usize * 3;
        let mut buffer = vec![0u8; bytes + 64];
        let mut actual = vec![];
        let root = tempfile::tempdir().unwrap();
        let output = root.path().join("recording.mp4");
        let mut encoder = Encoder::open(output.to_str().unwrap(), width, height, 15, None).unwrap();
        for _ in 0..8 {
            assert!(decoder.next_frame().unwrap().is_some());
            decoder.bgr(width, height, &mut buffer).unwrap();
            actual.extend_from_slice(&buffer[..bytes]);
            encoder.write(&buffer).unwrap();
        }
        encoder.finish().unwrap();
        drop(encoder);
        drop(decoder);
        let max = reference
            .stdout
            .iter()
            .zip(&actual)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        assert_eq!(reference.stdout.len(), actual.len());
        assert_eq!(max, 0, "Native BGR colour/resize parity changed");
        let probe = std::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "csv=p=0",
            ])
            .arg(output)
            .output()
            .unwrap();
        assert!(probe.status.success());
        assert_eq!(String::from_utf8(probe.stdout).unwrap().trim(), "8");
    }
    assert_eq!(live_handles(), (0, 0));
}
#[test]
fn stalled_rtsp_open_is_interruptible_without_leaking_native_handles() {
    let _guard = SERIAL.lock().unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let (stop_tx, stop_rx) = std::sync::mpsc::channel();
    let server = std::thread::spawn(move || {
        let (socket, _) = listener.accept().unwrap();
        let _ = stop_rx.recv_timeout(Duration::from_secs(5));
        drop(socket);
    });
    let stop = CancellationToken::new();
    let token = stop.clone();
    let start = Instant::now();
    let reader = std::thread::spawn(move || {
        Decoder::open(
            &format!("rtsp://127.0.0.1:{port}/blackhole"),
            token,
            Duration::from_secs(4),
        )
        .is_err()
    });
    std::thread::sleep(Duration::from_millis(200));
    stop.cancel();
    assert!(reader.join().unwrap());
    assert!(start.elapsed() < Duration::from_secs(2));
    let _ = stop_tx.send(());
    server.join().unwrap();
    assert_eq!(live_handles(), (0, 0));
}

#[test]
fn b_frames_bt709_full_range_and_eof_preserve_every_decoded_pixel() {
    let _guard = SERIAL.lock().unwrap();
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("bt709.mp4");
    let status = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-threads", "1", "-filter_threads", "1", "-i"])
        .arg(source())
        .args([
            "-frames:v",
            "12",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-bf",
            "2",
            "-g",
            "12",
            "-colorspace",
            "bt709",
            "-color_range",
            "pc",
        ])
        .arg(&path)
        .status()
        .unwrap();
    assert!(status.success());
    let reference = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-threads", "1", "-filter_threads", "1", "-i"])
        .arg(&path)
        .args(["-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"])
        .output()
        .unwrap();
    assert!(reference.status.success());
    let mut decoder = Decoder::open(
        path.to_str().unwrap(),
        CancellationToken::new(),
        Duration::from_secs(5),
    )
    .unwrap();
    let mut actual = vec![];
    let mut pixels = vec![0; 640 * 352 * 3 + 64];
    while decoder.next_frame().unwrap().is_some() {
        decoder.bgr(640, 352, &mut pixels).unwrap();
        actual.extend_from_slice(&pixels[..640 * 352 * 3]);
    }
    assert_eq!(actual.len(), 12 * 640 * 352 * 3);
    assert_eq!(reference.stdout.len(), actual.len());
    assert_eq!(
        reference
            .stdout
            .iter()
            .zip(&actual)
            .map(|(a, b)| a.abs_diff(*b))
            .max(),
        Some(0)
    );
    drop(decoder);
    assert_eq!(live_handles(), (0, 0));
}

#[test]
fn early_encoder_drop_finalizes_accepted_frames_without_an_explicit_finish() {
    let _guard = SERIAL.lock().unwrap();
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("early-drop.mp4");
    let mut encoder = Encoder::open(path.to_str().unwrap(), 640, 352, 15, None).unwrap();
    let pixels = vec![20; 640 * 352 * 3 + 64];
    for _ in 0..4 {
        encoder.write(&pixels).unwrap();
    }
    drop(encoder);
    let probe = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
        ])
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        probe.status.success(),
        "{}",
        String::from_utf8_lossy(&probe.stderr)
    );
    assert_eq!(String::from_utf8(probe.stdout).unwrap().trim(), "4");
    assert_eq!(live_handles(), (0, 0));
}

#[test]
fn retained_full_range_yuv_encodes_limited_range_preview_without_bgr() {
    let _guard = SERIAL.lock().unwrap();
    let root = tempfile::tempdir().unwrap();
    let raw = root.path().join("colour-bars.yuv");
    let input = root.path().join("full-range.mp4");
    let output = root.path().join("preview.mp4");
    let mut bars = Vec::<u8>::new();
    for (width, height, levels) in [
        (640, 352, [0, 255, 128]),
        (320, 176, [128, 128, 0]),
        (320, 176, [128, 128, 255]),
    ] {
        for _ in 0..height {
            bars.extend((0..width).map(|x| levels[3 * x / width]));
        }
    }
    std::fs::write(&raw, bars).unwrap();
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "yuvj420p",
            "-video_size",
            "640x352",
            "-framerate",
            "15",
            "-i",
        ])
        .arg(&raw)
        .args([
            "-frames:v",
            "1",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-qp",
            "0",
            "-colorspace",
            "bt709",
            "-color_range",
            "pc",
        ])
        .arg(&input)
        .status()
        .unwrap();
    assert!(status.success());

    let metrics = Arc::new(GpuMetrics::default());
    let mut decoder = Decoder::open_device(
        input.to_str().unwrap(),
        -1,
        CancellationToken::new(),
        Duration::from_secs(5),
        metrics.clone(),
    )
    .unwrap();
    decoder.next_frame().unwrap().unwrap();
    let frame = decoder.retain_host(640, 352).unwrap();
    drop(decoder);
    let mut encoder =
        Encoder::open_with_host(output.to_str().unwrap(), 320, 176, 15, None, Some(&frame))
            .unwrap();
    encoder.write_host(&frame).unwrap();
    encoder.repeat_frame().unwrap();
    // The same encoder can write a synthetic offline slate before YUV resumes.
    encoder.write(&vec![0; 320 * 176 * 3 + 64]).unwrap();
    encoder.write_host(&frame).unwrap();
    encoder.finish().unwrap();
    drop(encoder);
    assert_eq!(metrics.host_materialized.load(Ordering::Relaxed), 0);
    assert_eq!(live_handles(), (0, 0));

    let probe = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-of",
            "json",
            "-show_streams",
        ])
        .arg(&output)
        .output()
        .unwrap();
    assert!(probe.status.success());
    let spec: serde_json::Value = serde_json::from_slice(&probe.stdout).unwrap();
    let stream = &spec["streams"][0];
    assert_eq!(stream["color_space"], "bt709");
    assert_eq!(stream["color_range"], "tv");
    assert_eq!(stream["width"], 320);
    assert_eq!(stream["height"], 176);
    assert_eq!(stream["nb_read_frames"], "4");
    assert_eq!(stream["avg_frame_rate"], "15/1");

    let reference = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-threads", "1", "-filter_threads", "1", "-i"])
        .arg(input)
        .args([
            "-vf",
            "scale=320:176:flags=bilinear:in_range=pc:out_range=tv:in_color_matrix=bt709:out_color_matrix=bt709",
            "-pix_fmt", "yuv420p", "-threads", "1", "-f", "rawvideo", "pipe:1",
        ])
        .output()
        .unwrap();
    assert!(reference.status.success());

    let decoded = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-threads", "1", "-i"])
        .arg(output)
        .args([
            "-pix_fmt", "yuv420p", "-threads", "1", "-f", "rawvideo", "pipe:1",
        ])
        .output()
        .unwrap();
    assert!(decoded.status.success());
    const LUMA: usize = 320 * 176;
    let frames = decoded
        .stdout
        .chunks_exact(LUMA * 3 / 2)
        .collect::<Vec<_>>();
    assert_eq!(frames.len(), 4);
    assert_eq!(reference.stdout.len(), LUMA * 3 / 2);
    let row = 88 * 320;
    let chroma = 44 * 160 + 140;
    assert_eq!(reference.stdout[row + 30], 16);
    assert_eq!(reference.stdout[row + 160], 235);
    for index in [0, 1, 3] {
        for offset in [row + 30, row + 160, LUMA + chroma, LUMA + LUMA / 4 + chroma] {
            let value = frames[index][offset];
            let expected = reference.stdout[offset];
            assert!(
                value.abs_diff(expected) <= 5,
                "frame {index} offset {offset}: {value} != {expected}"
            );
        }
    }
    assert!(frames[2][88 * 320 + 160].abs_diff(16) <= 3);
}

#[test]
fn retained_yuv_evidence_and_chroma_sampling_skip_full_bgr() {
    let _guard = SERIAL.lock().unwrap();
    let root = tempfile::tempdir().unwrap();
    let raw = root.path().join("chroma-motion.yuv");
    let mut source_pixels = Vec::<u8>::new();
    const WIDTH: usize = 96;
    const HEIGHT: usize = 66;
    for (u, v) in [(97, 159), (166, 108)] {
        source_pixels.extend(std::iter::repeat_n(126, WIDTH * HEIGHT));
        source_pixels.extend(std::iter::repeat_n(u, WIDTH * HEIGHT / 4));
        source_pixels.extend(std::iter::repeat_n(v, WIDTH * HEIGHT / 4));
    }
    std::fs::write(&raw, source_pixels).unwrap();

    for (matrix, range) in [("bt709", "tv"), ("bt709", "pc"), ("smpte170m", "tv")] {
        let input = root.path().join(format!("{matrix}-{range}.mp4"));
        let status = std::process::Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "yuv420p",
                "-video_size",
                "96x66",
                "-i",
            ])
            .arg(&raw)
            .args([
                "-vf",
                &format!("setparams=colorspace={matrix}:range={range}"),
                "-frames:v",
                "2",
                "-c:v",
                "libx264",
                "-threads",
                "1",
                "-qp",
                "0",
                "-colorspace",
                matrix,
                "-color_range",
                range,
            ])
            .arg(&input)
            .status()
            .unwrap();
        assert!(status.success());

        let metrics = Arc::new(GpuMetrics::default());
        let mut decoder = Decoder::open_device(
            input.to_str().unwrap(),
            -1,
            CancellationToken::new(),
            Duration::from_secs(5),
            metrics.clone(),
        )
        .unwrap();
        decoder.next_frame().unwrap().unwrap();
        let first = decoder.retain_host(WIDTH as u32, HEIGHT as u32).unwrap();
        decoder.next_frame().unwrap().unwrap();
        let second = decoder.retain_host(WIDTH as u32, HEIGHT as u32).unwrap();
        drop(decoder);

        let mut samples = [[0; 4096]; 2];
        for (frame, sample) in [&first, &second].into_iter().zip(&mut samples) {
            assert_eq!(frame.freeze_sample(sample).unwrap().len(), 4096);
        }
        // A visual change can be present even when the Y plane is identical.
        assert!(samples[0][..3072] == samples[1][..3072], "{matrix} {range}");
        assert!(samples[0][3072..] != samples[1][3072..], "{matrix} {range}");

        let mut evidence = Vec::new();
        for frame in [&first, &second] {
            for (width, height) in [(WIDTH as u32, HEIGHT as u32), (93, 65)] {
                let jpeg = frame.jpeg(width, height).unwrap();
                let decoded = image::load_from_memory(&jpeg).unwrap().to_rgb8();
                assert_eq!(decoded.dimensions(), (width, height));
                evidence.push((frame, decoded.get_pixel(10, 33).0, matrix, range));
            }
        }
        assert_eq!(metrics.host_materialized.load(Ordering::Relaxed), 0);
        for (frame, actual, matrix, range) in evidence {
            let bgr = frame.bgr_padded().unwrap();
            let expected = [bgr[2], bgr[1], bgr[0]];
            for (encoded, original) in actual.into_iter().zip(expected) {
                assert!(
                    encoded.abs_diff(original) <= 6,
                    "{matrix} {range}: JPEG {actual:?}, BGR reference {expected:?}"
                );
            }
        }
    }
    assert_eq!(live_handles(), (0, 0));
}
