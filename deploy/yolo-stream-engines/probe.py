from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = Path('/home/changdar02/service-migration-phase7')


def main():
    config = json.loads((BASE / 'gpu-production.json').read_text())
    os.environ.update(config['environment'])
    os.chdir(config['cwd'])
    sys.path.insert(0, config['cwd'])
    import cv2
    import numpy as np
    import torch
    from ultralytics.nn.backends.tensorrt import TensorRTBackend
    from examples.YOLO_server_api.model_registry import load_registry
    from src.yolo_worker import YoloWorker

    if config['environment'].get('YOLO_WORKER_ARTIFACT_MANIFEST') != str(
        ROOT / 'engines.json',
    ):
        raise RuntimeError(
            'Production configuration does not select this manifest',
        )
    manifest = json.loads((ROOT / 'engines.json').read_text())
    registry = {e.id: e for e in load_registry().models}
    image = next(
        (
            im
            for p in sorted((BASE / 'samples').glob('*'))
            if (im := cv2.imread(str(p))) is not None
        ),
        None,
    )
    if image is None:
        raise RuntimeError('No real test image')
    for key, variant in manifest['models'].items():
        worker = YoloWorker(None)
        if worker.precision_mode != 'int8':
            raise RuntimeError('INT8 selection is not enabled')
        model = worker._get_model(key)
        result = model.predict(
            source=[image],
            device=worker.device,
            imgsz=worker.imgsz,
            batch=1,
            verbose=False,
            **worker.precision_args,
        )[0]
        backend = model.predictor.model
        if (
            not isinstance(backend.backend, TensorRTBackend)
            or Path(str(model.model)).resolve()
            != Path(variant['artifact']).resolve()
        ):
            raise RuntimeError(
                'Worker did not select the verified TensorRT engine: ' + key,
            )
        data = result.boxes.data.cpu().numpy()
        if not np.isfinite(data).all():
            raise RuntimeError('Non-finite detections: ' + key)
        if model._catalog_metadata['model_version'] != registry[key].version:
            raise RuntimeError('Published model version was lost')
        print(
            (
                'Installed worker uses TensorRT engine and preserves catalog '
                'metadata: '
            )
            + key,
            flush=True,
        )
        del result, model, worker, backend
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
