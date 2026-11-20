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

## Optional profiles for the Rust pipeline

The default build keeps batch 1, the established artifact paths and the
`engines.json` consumed by `manage.py`. For an isolated RTX 3050 candidate,
pass `--max-batch` between 1 and 4 to `build.py`. The input dimensions stay at
640 × 640. The profile accepts all batch sizes from 1 through the maximum. The
optimisation point defaults to 2 for larger profiles and can be selected with
`--opt-batch`.

For example, in a staged bundle with pinned inputs and the GPU runtime
environment, validate the proposed profile before building it.

```bash
python build.py --check --max-batch 4 --opt-batch 2
python build.py --max-batch 4 --opt-batch 2
```

This produces the engines, profile-specific receipts and manifest in
`candidates/batch-1-2-4/`. The build checks the serialized TensorRT profile and
runs representative inputs at sizes 1, 2, 3 and 4 before publishing the
candidate manifest. The legacy manager continues to use the batch-1 manifest.
An isolated Rust pipeline can select the candidate manifest and test it with
`batch_size` set to 4 before choosing a deployment configuration.

Use `--max-batch 2 --model yolo26m` to build only that pinned model. Overlay
the resulting entry on a copy of the active manifest to test a mixed profile.
Keep the active entries for the other models and verify the artifact
checksums in the mixed manifest before selecting it.

## Batch-1 deployment

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
