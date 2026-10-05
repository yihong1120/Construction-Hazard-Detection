#!/usr/bin/env bash
# Verify real Intel VAAPI encoding without changing live stream settings.
set -euo pipefail
vaapi_node=$(/usr/bin/python3 - <<'PY'
from pathlib import Path
for node in sorted(Path('/sys/class/drm').glob('renderD*')):
    try:
        if (node / 'device/vendor').read_text().strip().lower() == '0x8086':
            print('/dev/dri/' + node.name)
            break
    except OSError:
        pass
else:
    raise SystemExit('Intel render device is absent. Enable integrated graphics / iGPU Multi-Monitor in BIOS first.')
PY
)
if ! command -v vainfo >/dev/null; then
    echo 'Install prerequisites: sudo apt install intel-media-va-driver-non-free vainfo' >&2
    exit 2
fi
printf 'Testing Intel device: %s\n' "$vaapi_node"
LIBVA_DRIVER_NAME=iHD vainfo --display drm --device "$vaapi_node"
LIBVA_DRIVER_NAME=iHD ffmpeg -hide_banner -loglevel error \
    -vaapi_device "$vaapi_node" -f lavfi -i color=size=640x360:rate=15 \
    -vf 'format=nv12,hwupload' -c:v h264_vaapi -frames:v 3 -f null -
echo 'Intel VAAPI H.264 encoding passed.'
