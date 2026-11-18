# construction-media

Standalone pipeline 的程序內 FFmpeg adapter。Rust 擁有解碼器／編碼器、影格 buffer、取消期限與排程；C 只橋接 libavformat、libavcodec、libavutil 及 libswscale 的 opaque 狀態。
不啟動 FFmpeg 子程序，不依賴 PyO3／Python。這不是把 codec 實作重新寫成 Rust，也不是 GPU 零拷貝。

- 同一 handle 的所有呼叫都要求獨占 `&mut`；可以移交 owner 執行緒，不能同時共享使用。
- 傳入 C 的 interrupt state 放在穩定位址的 Box，生命週期涵蓋 open、I/O、finish 與 free；回呼只查詢取消與單調時鐘期限。
- Decoder 內部 AVFrame 不跨 ABI 借用。BGR 直接寫入呼叫端擁有的 buffer，包含最後 64 bytes SIMD padding。
- Encoder 的輸入也必須包含 64 bytes padding；C 不持有輸入影格指標。YUV／packet buffer 在 handle 內重用。
- `Encoder::Drop` 先完成 codec／muxer teardown，再釋放 handle；即使 packet drain 失敗仍關閉 muxer。RTSP socket 的關閉在 trailer，不可只呼叫 free_context。GPU session lease 必須持有至原生 handle 釋放；pipeline 關閉會 join owner 執行緒。
- Decoder 逾時／取消由 libav interrupt callback 中斷；encoder 每次 I/O／trailer 各有 5 秒期限，讓停止收格後仍能完成本機 MP4。
- 不含旋轉／鏡射轉換；帶有非 identity display matrix 的來源會明確拒絕。decoder 最大輸入影像為 64 Mi pixels。
- 編譯／執行的四個 FFmpeg library major 必須一致。libav 原始日誌關閉，錯誤只回傳操作階段與錯誤碼文字，避免來源 URL 的憑證外洩。

`PacketDecoder` 為 codec-only 入口，不建立 AVFormatContext 或網路 socket。
Rust RTSP 接收器交入完整 H.264／H.265 access unit、AVCC／HVCC 參數與 RTP 時戳；bridge 複製至帶 padding 的 reference-counted AVPacket 後才交給 libavcodec。
`send` 遇 EAGAIN 時保留原 packet、排空輸出再重試，EOF 則 flush 後排空 B-frame。
同一 codec owner 可選 CPU 或既有 NVDEC backend；接收協定與硬體解碼彼此獨立。
RTSP 連線、access unit 組裝和重連邏輯見 [Rust RTSP 接收器](../pipeline/src/rtsp.rs)。

`Decoder::retain_host`／`HostFrame` 保留 CPU AVFrame 的不可變引用，Rust 可在解碼器前進或 Drop 後繼續使用。
融合 letterbox 取樣只讀需要的 YUV 像素；完整 BGR 由 thread-safe OnceLock 在第一次消費時產生。
不支援的比例／色彩／格式保留原轉換路徑。共享校準器也用於既有 GPU LUT，兩條路徑均另有回歸驗證。

`tap_packets` 在既有輸入讀取處複製 reference-counted packet 引用；`Remuxer` 只做 RTSP mux。
Rust 使用有界 queue、獨立 owner、輸出 epoch 與 keyframe 恢復策略，不讓輸出 I/O 堵住解碼。
按需輸出與原始壓縮串流轉送的設定見 [攝影機流程說明](../pipeline/README.md)。

以 `native/pipeline/Cargo.lock` 建置與驗證，Ubuntu 需安裝：

```bash
sudo apt-get install ffmpeg libavformat-dev libavcodec-dev libavutil-dev libswscale-dev pkg-config
cargo test --locked --manifest-path native/pipeline/Cargo.toml --test media
```

非系統 SDK 可設定 `FFMPEG_ROOT`，包含 `usr/include/x86_64-linux-gnu`、`usr/include` 與 `usr/lib/x86_64-linux-gnu`。
本次實測 libavformat 62／libavcodec 62／libavutil 60／libswscale 9（FFmpeg 8）；adapter 為 FFmpeg 6 的 display matrix API 保留版本分支，尚未在本機驗證該版本。

`native/pipeline/tests/media.rs` 驗證逐像素 BGR、Full HD resize、BT.709 full-range、B-frame EOF drain、MP4 完整格數、取消與提前 Drop。
[服務整合測試](../pipeline/tests/services.rs) 使用隔離的 loopback MediaMTX，驗證 RTSP socket 在反覆 Drop 後正確關閉。

參考原生協定：[解碼 send／receive](https://ffmpeg.org/doxygen/8.0/demux_decode_8c-example.html)、[編碼與 mux](https://ffmpeg.org/doxygen/8.0/mux_8c-example.html)、[RTSP teardown](https://github.com/FFmpeg/FFmpeg/blob/n8.0/libavformat/rtspenc.c)。
