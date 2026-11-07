# Device-specific YOLO26 stream engines

The published model catalog continues to identify and verify the original
checkpoint. `YOLO_WORKER_ARTIFACT_MANIFEST` optionally selects a separately
hashed TensorRT engine for streaming when `YOLO_WORKER_PRECISION=int8`.
Each variant is bound to the catalog's version and checkpoint checksum.
Image inference and catalog authorization are unchanged.

The B deployment bundle lives in
`~/service-migration-phase7/runtime-fixes/yolo26-mixed-int8`. It contains pinned
ONNX/checkpoint checksums and candidate runtime modules in addition to these
scripts. The existing A-built engines require compute capability 8.9; B has
8.6. `build.py` preserves the calibrated Q/DQ and FP16 tensors in the ONNX
and builds new batch-1, square-640 profiles directly on B with its TensorRT.
Original engines are retained. No calibration package or dataset is needed.

On B:

```bash
python3 "$HOME/service-migration-phase7/runtime-fixes/yolo26-mixed-int8/manage.py" check
sudo bash "$HOME/service-migration-phase7/runtime-fixes/yolo26-mixed-int8/resume-build.sh"
sudo journalctl -fu yolo26-mixed-int8-build.service
sudo python3 "$HOME/service-migration-phase7/runtime-fixes/yolo26-mixed-int8/manage.py" apply
```

The wrapper starts a transient systemd service so an SSH disconnect or closing
the log viewer does not interrupt the supervisor. Wait for the journal message
`All five B engines are ready` and the camera-service resume message before
applying. Press Ctrl+C to leave the log viewer; this does not cancel the build.
An existing build unit prevents a second wrapper launch.

Build pauses camera workers to free VRAM and resumes the original service
even if a build fails. This interrupts live camera publication and event
collection for the build duration. Websites and the image API are retained.
Completed engines are reused only when their source and output hashes match.
Build success does not switch production; `apply` is the separate switch.
Engine filenames now include the model revision and source digest. Rebuilding
preserves previous engine files and retains two previous entries in the manifest,
so a catalog publication can verify and restore its prior engine. Do not remove
those artifacts before a publication has completed its restoration window.
Apply backs up the two runtime modules and private service configuration,
tests every engine through the installed worker with real sample images,
then starts camera workers and checks the three websites. Failure restores
the original code and configuration.

```bash
sudo python3 "$HOME/service-migration-phase7/runtime-fixes/yolo26-mixed-int8/manage.py" rollback
```

Smoke tests verify executable compatibility, output validity and class IDs;
they do not establish detection accuracy or an mAP regression bound. Review
representative camera detections after switching and measure throughput
under the real camera topology. Annotated frame/box synchronization remains
handled by the independent overlay fix.
