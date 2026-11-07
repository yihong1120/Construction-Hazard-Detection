from __future__ import annotations

from src.settings.media import EncoderSettings


class EncoderOptions:
    """One validated encoding configuration shared by a camera's outputs."""

    def __init__(self, settings: EncoderSettings | None = None) -> None:
        self.settings = settings if settings is not None else EncoderSettings()

    def x264_options(
        self,
        *,
        bitrate: str | None = None,
        maxrate: str | None = None,
        bufsize: str | None = None,
    ) -> list[str]:
        """Return ffmpeg options for CPU H.264 encoding."""
        config = self.settings
        options = [
            '-c:v',
            'libx264',
            '-preset',
            config.preset,
            '-tune',
            'zerolatency',
            '-pix_fmt',
            'yuv420p',
        ]
        if bitrate:
            options.extend(
                [
                    '-b:v',
                    bitrate,
                    '-maxrate',
                    maxrate or bitrate,
                    '-bufsize',
                    bufsize or maxrate or bitrate,
                ],
            )
        else:
            options.extend(
                [
                    '-crf',
                    str(config.crf),
                ],
            )
        return options

    def nvenc_options(
        self,
        *,
        bitrate: str | None = None,
        maxrate: str | None = None,
        bufsize: str | None = None,
    ) -> list[str]:
        """Return ffmpeg options for NVIDIA H.264 encoding."""
        config = self.settings
        return [
            '-c:v',
            'h264_nvenc',
            '-preset',
            config.nvenc_preset,
            '-tune',
            'ull',
            '-rc',
            config.nvenc_rc,
            '-cq',
            str(config.nvenc_cq),
            '-b:v',
            bitrate or config.nvenc_bitrate or '0',
            '-maxrate',
            maxrate or config.nvenc_maxrate or '8M',
            '-bufsize',
            bufsize or config.nvenc_bufsize or '16M',
            '-pix_fmt',
            'yuv420p',
        ]

    def vaapi_options(
        self,
        *,
        bitrate: str | None = None,
        maxrate: str | None = None,
        bufsize: str | None = None,
    ) -> list[str]:
        """Return Intel VAAPI H.264 options for low-latency live streams."""
        config = self.settings
        target_bitrate = bitrate or config.vaapi_bitrate or '4M'
        target_maxrate = maxrate or config.vaapi_maxrate or '8M'
        target_bufsize = bufsize or config.vaapi_bufsize or '16M'
        return [
            '-vf',
            'format=nv12,hwupload',
            '-c:v',
            'h264_vaapi',
            '-rc_mode',
            config.vaapi_rc_mode,
            '-b:v',
            target_bitrate,
            '-maxrate',
            target_maxrate,
            '-bufsize',
            target_bufsize,
            '-bf',
            '0',
            '-async_depth',
            str(config.vaapi_async_depth),
            '-profile:v',
            'high',
        ]
