#!/usr/bin/env bash
# Keep the build supervisor alive when the invoking SSH session disconnects.
set -euo pipefail

if [[ $(id -u) -ne 0 ]]; then
    echo 'Run this script with sudo on B.' >&2
    exit 1
fi

bundle=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
unit=yolo26-mixed-int8-build

systemd-run --unit="$unit" --collect --service-type=exec \
    --property="WorkingDirectory=$bundle" \
    /usr/bin/python3 "$bundle/manage.py" build

echo "Build started. Follow progress: sudo journalctl -fu $unit.service"
echo 'Closing the log viewer does not stop the build. Production selection changes only with manage.py apply.'
