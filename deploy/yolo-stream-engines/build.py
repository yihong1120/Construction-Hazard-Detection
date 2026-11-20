from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent
BASE = Path('/home/changdar02/service-migration-phase7')


class BatchProfile(NamedTuple):
    optimum: int = 1
    maximum: int = 1

    @property
    def shapes(self):
        return tuple(
            (batch, 3, 640, 640) for batch in (1, self.optimum, self.maximum)
        )

    @property
    def identity(self):
        return dict(
            zip(('min', 'opt', 'max'), map(list, self.shapes), strict=True),
        )

    @property
    def label(self):
        if self.maximum == 1:
            return 'batch-1'
        return f'batch-1-{self.optimum}-{self.maximum}'


def build_options(argv=None):
    parser = argparse.ArgumentParser(
        description='Build pinned RTX 3050 stream engines',
    )
    parser.add_argument(
        '--check', action='store_true',
        help='Validate the inputs and requested profile',
    )
    parser.add_argument(
        '--max-batch', type=int,
        choices=range(1, 5), default=1,
    )
    parser.add_argument('--opt-batch', type=int, choices=range(1, 5))
    parser.add_argument(
        '--model',
        help='Build a single pinned model in a separate batch profile',
    )
    args = parser.parse_args(argv)
    optimum = args.opt_batch if args.opt_batch is not None else min(
        2, args.max_batch,
    )
    if optimum > args.max_batch:
        parser.error('--opt-batch must be at most --max-batch')
    if args.model is not None and args.max_batch == 1:
        parser.error('--model requires --max-batch greater than 1')
    return args.check, BatchProfile(optimum, args.max_batch), args.model


def output_locations(profile):
    if profile.maximum == 1:
        return BASE / 'source-models/int8_engine/rtx3050', ROOT
    candidate = ROOT / 'candidates' / profile.label
    if candidate.resolve() != candidate:
        raise RuntimeError(
            'Batched engines require a separate profile directory',
        )
    return candidate, candidate


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


def add_batch_profile(builder, network, config, requested=BatchProfile()):
    """Validate the stored profile; set_shape returns None on TensorRT 11."""
    if network.num_inputs != 1:
        raise RuntimeError('Expected one image input')
    inp = network.get_input(0)
    profile = builder.create_optimization_profile()
    profile.set_shape(inp.name, *requested.shapes)
    stored = tuple(
        tuple(dimensions)
        for dimensions in profile.get_shape(inp.name)
    )
    if stored != requested.shapes or not bool(profile):
        raise RuntimeError(f'TensorRT rejected the {requested.label} profile')
    index = config.add_optimization_profile(profile)
    if index < 0:
        raise RuntimeError(
            f'TensorRT could not add the {requested.label} profile',
        )
    return index


def engine_metadata(original, profile=BatchProfile()):
    """Keep export metadata from overwriting the backend's CUDA device."""
    metadata = dict(original)
    metadata.pop('device', None)
    metadata.update(batch=profile.maximum, imgsz=[640, 640])
    metadata['args'] = {
        **metadata.get('args', {}),
        'batch': profile.maximum, 'dynamic': True, 'workspace': 0.5,
    }
    return metadata


def repair_device_metadata(
    engine_path, receipt_path, receipt, profile=BatchProfile(),
):
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
    metadata = engine_metadata(metadata, profile)
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


def cached_receipt(
    receipt_path, engine_path, pin, tensorrt_version, compute_capability,
    profile,
):
    if not receipt_path.is_file() or not engine_path.is_file():
        return None
    old = json.loads(receipt_path.read_text())
    previous_profile = old.get('optimization_profile')
    if previous_profile is None and profile.maximum == 1:
        # Batch 1 is the sole profile created by the previous builder.
        previous_profile = BatchProfile().identity
    if (
        old.get('inputs') == pin
        and old.get('tensorrt') == tensorrt_version
        and old.get('compute_capability') == compute_capability
        and previous_profile == profile.identity
        and old.get('sha256') == digest(engine_path)
    ):
        return old
    return None


def validate_engine_profile(engine_path, trt, logger, profile):
    with engine_path.open('rb') as stream:
        size = int.from_bytes(stream.read(4), 'little')
        if not 0 < size < 1000000:
            raise RuntimeError('Missing Ultralytics engine metadata')
        metadata = json.loads(stream.read(size))
        plan = stream.read()
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None or engine.num_optimization_profiles != 1:
        raise RuntimeError('Cannot inspect the TensorRT optimization profile')
    names = map(engine.get_tensor_name, range(engine.num_io_tensors))
    inputs = [
        name for name in names
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
    ]
    if len(inputs) != 1:
        raise RuntimeError('Expected one engine image input')
    actual = tuple(
        tuple(dims)
        for dims in engine.get_tensor_profile_shape(inputs[0], 0)
    )
    if actual != profile.shapes or metadata.get('batch') != profile.maximum:
        raise RuntimeError(
            f'Engine does not match the requested {profile.label} profile',
        )


def verify_inference_batch(model, images, batch, allowed_classes, torch, np):
    started = time.monotonic()
    results = model.predict(
        images, device='cuda:0', imgsz=640, batch=batch,
        rect=False, verbose=False,
    )
    torch.cuda.synchronize()
    if len(results) != batch:
        raise RuntimeError(
            'TensorRT did not return one response per image',
        )
    detection_counts = []
    for result in results:
        data = result.boxes.data.cpu().numpy()
        if data.ndim != 2 or data.shape[1] != 6 or not np.isfinite(data).all():
            raise RuntimeError('Invalid detection output')
        if any(int(value) not in allowed_classes for value in data[:, -1]):
            raise RuntimeError('Unknown class in detection output')
        detection_counts.append(len(data))
    return time.monotonic() - started, detection_counts


def verify_inference(model, samples, allowed_classes, profile, torch, np):
    singles = [
        verify_inference_batch(model, image, 1, allowed_classes, torch, np)
        for image in samples
    ]
    smoke_batches = {}
    for batch in range(2, profile.maximum + 1):
        images = [samples[index % len(samples)] for index in range(batch)]
        duration, counts = verify_inference_batch(
            model, images, batch, allowed_classes, torch, np,
        )
        smoke_batches[str(batch)] = {'seconds': duration, 'detections': counts}
    return {
        'smoke_inference_seconds': [duration for duration, _ in singles],
        'sample_detections': [counts[0] for _, counts in singles],
        'smoke_batches': smoke_batches,
        'smoke_passed': True,
    }


def build(check_only, profile, selected_model=None):
    context()
    import cv2
    import numpy as np
    import onnx
    import tensorrt as trt
    import torch
    from ultralytics import YOLO
    from examples.YOLO_server_api.model_registry import load_registry
    from examples.YOLO_server_api.model_registry import verify_artifact

    pins = json.loads((ROOT / 'inputs.json').read_text())
    pinned_models = pins['models']
    if selected_model is not None:
        if selected_model not in pinned_models:
            raise RuntimeError('Model is absent from the pinned inputs')
        pinned_models = {selected_model: pinned_models[selected_model]}
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
    destination, receipts = output_locations(profile)
    if not check_only:
        destination.mkdir(parents=True, exist_ok=True)
    logger = trt.Logger(trt.Logger.WARNING)
    manifest = {'format': 1, 'models': {}}
    for key, pin in pinned_models.items():
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
            add_batch_profile(builder, network, config, profile)
            del parser, config, network, builder
            gc.collect()
            print(
                'Verified source, classes and ONNX parser for '
                f'{profile.label} profile: {key}',
                flush=True,
            )
            continue
        revision = hashlib.sha256(entry.version.encode()).hexdigest()[:12]
        engine_path = destination / (
            f'best_{key}-{revision}-{entry.sha256[:12]}.engine'
        )
        receipt_path = receipts / (key + '-build.json')
        reusable = cached_receipt(
            receipt_path, engine_path, pin, trt.__version__,
            pins['compute_capability'], profile,
        )
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
            add_batch_profile(builder, network, config, profile)
            started = time.monotonic()
            plan = builder.build_serialized_network(network, config)
            if plan is None:
                raise RuntimeError('TensorRT engine build failed: ' + key)
            metadata = engine_metadata(metadata, profile)
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
                'optimization_profile': profile.identity,
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
                engine_path, receipt_path, reusable, profile,
            )
        # Inspect the actual plan and exercise every supported partial batch.
        validate_engine_profile(engine_path, trt, logger, profile)
        model = YOLO(str(engine_path), task='detect')
        smoke = verify_inference(
            model, samples, {c.id for c in entry.classes}, profile, torch, np,
        )
        reusable.update(
            smoke, artifact=str(engine_path),
            optimization_profile=profile.identity,
        )
        save(receipt_path, reusable)
        manifest['models'][key] = {
            'artifact': str(engine_path), 'sha256': reusable['sha256'],
            'source_sha256': entry.sha256, 'version': entry.version,
        }
        print('Verified engine and real image inference: ' + key, flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if check_only:
        print(
            'Preflight passed. No engine was built; production is unchanged.',
            flush=True,
        )
        return
    previous_path = receipts / 'engines.json'
    if previous_path.exists():
        from src.yolo_runtime_artifacts import retain_engine_history

        previous = json.loads(previous_path.read_text())
        manifest = retain_engine_history(manifest, previous)
    save(previous_path, manifest)
    print(
        f'Engine candidates verified: {", ".join(pinned_models)}',
        flush=True,
    )


def main():
    check_only, profile, model = build_options()
    if check_only:
        return build(check_only, profile, model)
    # Serialise direct invocations and legacy manager invocations on this GPU.
    with (ROOT / 'builder.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return build(check_only, profile, model)


if __name__ == '__main__':
    os.umask(0o077)
    main()
