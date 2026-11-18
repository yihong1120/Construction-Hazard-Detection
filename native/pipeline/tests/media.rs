//! Pixel compatibility and native-handle lifetime gates for in-process media.
use construction_media::{Decoder, Encoder, live_handles};
use std::{
    path::Path,
    sync::Mutex,
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
