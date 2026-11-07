from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = Path('/home/changdar02/service-migration-phase7')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def context():
    config = json.loads((BASE / 'gpu-production.json').read_text())
    os.environ.update(config['environment'])
    os.chdir(config['cwd'])
    sys.path.insert(0, config['cwd'])
    return config


def add_batch_one_profile(builder, network, config):
    """Validate the stored profile; set_shape returns None on TensorRT 11."""
    if network.num_inputs != 1:
        raise RuntimeError('Expected one image input')
    inp = network.get_input(0)
    shape = (1, 3, 640, 640)
    profile = builder.create_optimization_profile()
    profile.set_shape(inp.name, shape, shape, shape)
    stored = tuple(
        tuple(dimensions)
        for dimensions in profile.get_shape(inp.name)
    )
    if stored != (shape, shape, shape) or not bool(profile):
        raise RuntimeError('TensorRT rejected the batch-1 profile')
    index = config.add_optimization_profile(profile)
    if index < 0:
        raise RuntimeError('TensorRT could not add the batch-1 profile')
    return index


def batch_one_metadata(original):
    """Keep export metadata from overwriting the backend's CUDA device."""
    metadata = dict(original)
    metadata.pop('device', None)
    metadata.update(batch=1, imgsz=[640, 640])
    metadata['args'] = {
        **metadata.get('args', {}),
        'batch': 1, 'dynamic': True, 'workspace': 0.5,
    }
    return metadata


def repair_device_metadata(engine_path, receipt_path, receipt):
    """Repair an unpublished candidate's header without rebuilding its plan."""
    if digest(engine_path) != receipt['sha256']:
        raise RuntimeError('Cannot repair an engine whose checksum changed')
    with engine_path.open('rb') as stream:
        size = int.from_bytes(stream.read(4), 'little')
        if not 0 < size < 1000000:
            raise RuntimeError('Missing Ultralytics engine metadata')
        metadata = json.loads(stream.read(size))
        plan = stream.read()
    if 'device' not in metadata:
        return receipt
    if metadata['device'] != 'RTX 3050':
        raise RuntimeError('Unexpected candidate device metadata')
    metadata = batch_one_metadata(metadata)
    encoded = json.dumps(metadata).encode()
    backup = engine_path.with_name(
        engine_path.name + '.before-device-fix-' + str(time.time_ns()),
    )
    shutil.copy2(engine_path, backup)
    temporary = engine_path.with_suffix('.engine.tmp')
    with temporary.open('wb') as stream:
        stream.write(len(encoded).to_bytes(4, 'little'))
        stream.write(encoded)
        stream.write(plan)
    os.replace(temporary, engine_path)
    receipt = {
        **receipt, 'sha256': digest(engine_path),
        'plan_sha256': hashlib.sha256(plan).hexdigest(),
        'metadata_backup': str(backup),
    }
    receipt.pop('smoke_passed', None)
    save(receipt_path, receipt)
    print(
        'Corrected candidate metadata; TensorRT plan retained: ' +
        engine_path.name, flush=True,
    )
    return receipt


def main():
    context()
    check_only = '--check' in sys.argv
    import cv2
    import numpy as np
    import onnx
    import tensorrt as trt
    import torch
    from ultralytics import YOLO
    from examples.YOLO_server_api.model_registry import load_registry
    from examples.YOLO_server_api.model_registry import verify_artifact
    from src.yolo_runtime_artifacts import retain_engine_history

    pins = json.loads((ROOT / 'inputs.json').read_text())
    if torch.cuda.get_device_capability(
        0,
    ) != tuple(pins['compute_capability']):
        raise RuntimeError('This bundle is pinned to the B GPU architecture')
    entries = {entry.id: entry for entry in load_registry().models}
    samples = [
        image for path in sorted((BASE / 'samples').glob('*'))
        if (image := cv2.imread(str(path))) is not None
    ][:3]
    if not samples:
        raise RuntimeError('No representative images are available')
    destination = BASE / 'source-models/int8_engine/rtx3050'
    if not check_only:
        destination.mkdir(exist_ok=True)
    logger = trt.Logger(trt.Logger.WARNING)
    manifest = {'format': 1, 'models': {}}
    for key, pin in pins['models'].items():
        entry = entries[key]
        if (
            entry.version != pin['version']
            or entry.sha256 != pin['source_sha256']
        ):
            raise RuntimeError('Published checkpoint changed: ' + key)
        source_path = verify_artifact(entry)
        onnx_path = BASE / 'source-models/pt' / ('best_' + key + '.int8.onnx')
        original = BASE / 'source-models/int8_engine' / \
            ('best_' + key + '.engine')
        if digest(onnx_path) != pin['onnx_sha256'] or digest(
            original,
        ) != pin['original_engine_sha256']:
            raise RuntimeError('Quantized source changed: ' + key)
        with original.open('rb') as stream:
            size = int.from_bytes(stream.read(4), 'little')
            if not 0 < size < 1000000:
                raise RuntimeError('Missing Ultralytics engine metadata')
            metadata = json.loads(stream.read(size))
        published = YOLO(str(source_path), task='detect')
        if metadata['names'] != {
            str(k): v for k,
            v in published.names.items()
        }:
            raise RuntimeError(
                'Engine class names differ from published checkpoint',
            )
        if set(map(int, metadata['names'])) != {
            cls.id for cls in entry.classes
        }:
            raise RuntimeError(
                'Engine class IDs differ from published catalog',
            )
        del published
        gc.collect()
        if check_only:
            graph = onnx.load(str(onnx_path), load_external_data=False)
            if (
                not any(
                    n.op_type == 'QuantizeLinear' for n in graph.graph.node
                ) or not any(
                    t.data_type == onnx.TensorProto.FLOAT16
                    for t in graph.graph.initializer
                )
            ):
                raise RuntimeError('Source is not a mixed INT8/FP16 ONNX')
            del graph
            builder = trt.Builder(logger)
            network = builder.create_network(0)
            parser = trt.OnnxParser(network, logger)
            if not parser.parse_from_file(str(onnx_path)):
                raise RuntimeError(
                    'B TensorRT cannot parse quantized ONNX: ' + key,
                )
            config = builder.create_builder_config()
            add_batch_one_profile(builder, network, config)
            del parser, config, network, builder
            gc.collect()
            print(
                'Verified source, classes, ONNX parser and batch-1 profile: ' +
                key,
                flush=True,
            )
            continue
        revision = hashlib.sha256(entry.version.encode()).hexdigest()[:12]
        engine_path = destination / (
            f'best_{key}-{revision}-{entry.sha256[:12]}.engine'
        )
        receipt_path = ROOT / (key + '-build.json')
        reusable = None
        if receipt_path.exists() and engine_path.exists():
            old = json.loads(receipt_path.read_text())
            if (
                old['inputs'] == pin and old['tensorrt'] == trt.__version__
                and old['compute_capability'] == pins['compute_capability']
                and old['sha256'] == digest(engine_path)
            ):
                reusable = old
        if reusable is None:
            print(
                'Building mixed INT8/FP16 engine on RTX 3050: ' +
                key,
                flush=True,
            )
            graph = onnx.load(str(onnx_path), load_external_data=False)
            counts = {
                op: sum(node.op_type == op for node in graph.graph.node)
                for op in ('QuantizeLinear', 'DequantizeLinear')
            }
            half_weights = sum(
                t.data_type == onnx.TensorProto.FLOAT16
                for t in graph.graph.initializer
            )
            if not all(counts.values()) or not half_weights:
                raise RuntimeError(
                    'Expected an explicitly quantized mixed-precision ONNX',
                )
            if any(t.external_data for t in graph.graph.initializer):
                raise RuntimeError(
                    'External ONNX data is not pinned in this bundle',
                )
            del graph
            free, _ = torch.cuda.mem_get_info()
            if free < 2 * 1024**3:
                raise RuntimeError(
                    'Less than 2 GiB GPU headroom; '
                    'original cameras will be resumed',
                )
            builder = trt.Builder(logger)
            network = builder.create_network(0)
            config = builder.create_builder_config()
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, 512 * 1024**2,
            )
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
            # TensorRT 11 reads the Q/DQ and FP16 types directly. On older
            # versions allow the same unquantized layers to use FP16.
            if int(trt.__version__.split('.')[0]) < 11:
                config.set_flag(trt.BuilderFlag.FP16)
            parser = trt.OnnxParser(network, logger)
            if not parser.parse_from_file(str(onnx_path)):
                errors = [
                    str(parser.get_error(i))
                    for i in range(parser.num_errors)
                ]
                raise RuntimeError('ONNX parse failed: ' + '; '.join(errors))
            add_batch_one_profile(builder, network, config)
            started = time.monotonic()
            plan = builder.build_serialized_network(network, config)
            if plan is None:
                raise RuntimeError('TensorRT engine build failed: ' + key)
            metadata = batch_one_metadata(metadata)
            encoded = json.dumps(metadata).encode()
            temporary = engine_path.with_suffix('.engine.tmp')
            with temporary.open('wb') as stream:
                stream.write(len(encoded).to_bytes(4, 'little'))
                stream.write(encoded)
                stream.write(plan)
            os.replace(temporary, engine_path)
            reusable = {
                'inputs': pin,
                'tensorrt': trt.__version__,
                'compute_capability': pins['compute_capability'],
                'artifact': str(engine_path),
                'sha256': digest(engine_path),
                'build_seconds': time.monotonic() - started,
                'qdq_nodes': counts,
                'half_weight_tensors': half_weights,
            }
            del parser, config, network, builder, plan
            gc.collect()
            save(receipt_path, reusable)
        else:
            print('Reusing verified completed engine: ' + key, flush=True)
            reusable = repair_device_metadata(
                engine_path, receipt_path, reusable,
            )
        # Real batch-1 inference exercises deserialization, allocations,
        # preprocessing, output decoding and the published class mapping.
        model = YOLO(str(engine_path), task='detect')
        timings = []
        detections = []
        for image in samples:
            started = time.monotonic()
            result = model.predict(
                image, device='cuda:0', imgsz=640, batch=1,
                rect=False, verbose=False,
            )[0]
            torch.cuda.synchronize()
            data = result.boxes.data.cpu().numpy()
            if data.ndim != 2 or data.shape[1] != 6 or not np.isfinite(
                data,
            ).all():
                raise RuntimeError('Invalid detection output: ' + key)
            if any(
                int(value) not in {c.id for c in entry.classes}
                for value in data[:, -1]
            ):
                raise RuntimeError('Unknown class in detection output: ' + key)
            timings.append(time.monotonic() - started)
            detections.append(len(data))
        reusable.update(
            smoke_inference_seconds=timings, sample_detections=detections,
            smoke_passed=True,
        )
        save(receipt_path, reusable)
        manifest['models'][key] = {
            'artifact': str(engine_path), 'sha256': reusable['sha256'],
            'source_sha256': entry.sha256, 'version': entry.version,
        }
        print('Verified engine and real image inference: ' + key, flush=True)
        del model, result, data
        gc.collect()
        torch.cuda.empty_cache()
    if check_only:
        print(
            'Preflight passed. No engine was built; production is unchanged.',
            flush=True,
        )
        return
    previous_path = ROOT / 'engines.json'
    previous = json.loads(
        previous_path.read_text(),
    ) if previous_path.exists() else {}
    save(previous_path, retain_engine_history(manifest, previous))
    print(
        'All five B engines are ready. Production selection has not changed.',
        flush=True,
    )


if __name__ == '__main__':
    os.umask(0o077)
    main()
