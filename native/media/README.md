🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# construction-media

This crate connects the standalone pipeline to FFmpeg in the same process. Rust owns the decoders, encoders, frame buffers, cancellation deadlines and scheduling. A C bridge handles the opaque state of libavformat, libavcodec, libavutil and libswscale. The codecs run in FFmpeg libraries, without an FFmpeg subprocess or a Python/PyO3 dependency. Frames can still require copies.

- Each handle requires exclusive `&mut` access. Ownership can move between threads.
- The interrupt state passed to C lives in a `Box` at a stable address from open through I/O, finish and free. Its callback checks only cancellation and the monotonic deadline.
- The decoder keeps its AVFrame inside the ABI boundary. It writes BGR directly into a caller-owned buffer with 64 trailing bytes of SIMD padding.
- The encoder accepts a retained `HostFrame` or BGR with 64 bytes of padding. It scales a retained frame directly into the encoder's YUV planes, borrowing the source for one call. The handle reuses YUV and packet buffers.
- `Encoder::Drop` tears down the codec and muxer before freeing the handle. It closes the muxer even if draining packets fails. RTSP sockets close when the trailer is written, so freeing the context alone is insufficient. Hold the GPU session lease until the native handle is released. The pipeline joins the owning thread on shutdown.
- The libav interrupt callback interrupts decoder I/O on timeout or cancellation. Each encoder I/O or trailer operation has a five-second deadline, allowing a local MP4 to finish after capture stops.
- The decoder rejects non-identity display matrices (rotation or reflection) and limits input to 64 Mi pixels.
- All four FFmpeg libraries must have the same major versions at build time and run time respectively. Raw libav logs are disabled. Errors contain the stage and error-code text to keep source URL credentials out of logs.

`PacketDecoder` opens the codec without creating an AVFormatContext or network socket. The Rust RTSP receiver supplies complete H.264/H.265 access units, AVCC/HVCC parameters and RTP timestamps. The bridge copies each unit into a padded, reference-counted AVPacket for libavcodec. On EAGAIN, `send` retains the packet, drains output and retries. At EOF it flushes the codec and drains the remaining B-frames. The same codec owner can use CPU decoding or NVDEC independently of the transport. [RTSP connection, access-unit assembly and reconnection](../pipeline/src/rtsp.rs)

`Decoder::retain_host` and `HostFrame` retain an immutable reference to a CPU AVFrame. Rust can read it after the decoder advances or drops. Fused letterbox sampling reads the required YUV pixels. A thread-safe OnceLock builds the full BGR image on first use. Unsupported scales, colours and formats use the existing conversion. The GPU LUT uses the same calibrator. Each path has its own regression checks.

`Encoder::open_with_host` uses the first frame's YUV metadata to select the output colour matrix. `write_host` produces limited-range YUV420P in a single scaling pass. Later frames and BGR offline notices use the same output colour settings.

`Encoder::write_host_with` briefly lends the scaled YUV420 planes to the backend for annotation before encoding. Alert snapshots use `HostFrame::jpeg` and the freeze detector uses `HostFrame::freeze_sample`. Both read the retained decoded frame.

`tap_packets` takes reference-counted packet references as the input is read. `Remuxer` muxes them for RTSP. A bounded queue, a separate owner, output epochs and keyframe recovery prevent output I/O from blocking the decoder. [On-demand output and compressed stream forwarding](../pipeline/README.md)

Build and verify with `native/pipeline/Cargo.lock`. Install the following development packages on Ubuntu.

```bash
sudo apt-get install ffmpeg libavformat-dev libavcodec-dev libavutil-dev libswscale-dev pkg-config
cargo test --locked --manifest-path native/pipeline/Cargo.toml --test media
```

For an SDK outside the system paths, set `FFMPEG_ROOT` to the prefix containing `usr/include/x86_64-linux-gnu`, `usr/include` and `usr/lib/x86_64-linux-gnu`. The tested versions were libavformat 62, libavcodec 62, libavutil 60 and libswscale 9 (FFmpeg 8). The adapter also has a version-specific branch for FFmpeg 6's display-matrix API. That version has yet to be verified locally.

`native/pipeline/tests/media.rs` checks pixel-exact BGR output, Full HD resizing, full-range BT.709, B-frame draining at EOF, complete MP4 frame counts, cancellation and early drop. The [service integration tests](../pipeline/tests/services.rs) use an isolated loopback MediaMTX instance to verify that successive drops close the RTSP sockets.

See the native interfaces for [decoder send/receive](https://ffmpeg.org/doxygen/8.0/demux_decode_8c-example.html), [encoding and muxing](https://ffmpeg.org/doxygen/8.0/mux_8c-example.html) and [RTSP teardown](https://github.com/FFmpeg/FFmpeg/blob/n8.0/libavformat/rtspenc.c).
