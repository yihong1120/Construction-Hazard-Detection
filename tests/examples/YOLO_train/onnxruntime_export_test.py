from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import helper
from onnx import numpy_helper
from onnx import TensorProto

from examples.YOLO_train import export_int8_trt as subject


class CalibrationBatches(list):
    """Expose the training-loader fields consumed by the calibration reader."""

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(0)
        super().__init__(
            [
                {
                    'img': torch.randint(
                        0,
                        256,
                        (1, 3, 8, 8),
                        generator=generator,
                    ),
                }
                for _ in range(3)
            ],
        )
        self.dataset = list(range(3))
        self.batch_size = 1


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    """Create two convolutions so node exclusion has observable effects."""
    rng = np.random.default_rng(0)
    initializers = [
        numpy_helper.from_array(
            rng.normal(size=shape).astype(np.float32),
            name,
        )
        for name, shape in [
            ('body.weight', (4, 3, 3, 3)),
            ('body.bias', (4,)),
            ('head.weight', (2, 4, 1, 1)),
        ]
    ]
    graph = helper.make_graph(
        [
            helper.make_node(
                'Conv',
                ['images', 'body.weight', 'body.bias'],
                ['features'],
                name='/body/Conv',
                pads=[1, 1, 1, 1],
            ),
            helper.make_node(
                'Conv',
                ['features', 'head.weight'],
                ['output'],
                name='/head/Conv',
            ),
        ],
        'quantization-test',
        [
            helper.make_tensor_value_info(
                'images',
                TensorProto.FLOAT,
                ['batch', 3, 8, 8],
            ),
        ],
        [
            helper.make_tensor_value_info(
                'output',
                TensorProto.FLOAT,
                ['batch', 2, 8, 8],
            ),
        ],
        initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid('', 13),
        ],
        ir_version=9,
    )
    path = tmp_path / 'model.onnx'
    onnx.save(model, path)
    return path


def test_int8_calibration_preserves_excluded_weights(
    model_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The calibrated graph executes and keeps the excluded head in FP32."""
    monkeypatch.setattr(subject, 'node_exclusion_patterns', (r'/head/.*',))
    monkeypatch.setattr(subject, 'calibration_method', 'entropy')
    output = subject.onnxruntime_quantize_onnx_all_images(
        str(model_path),
        quantize=8,
        dataset=CalibrationBatches(),
        dynamic=True,
    )
    model = onnx.load(output)
    onnx.checker.check_model(model)
    assert any(node.op_type == 'QuantizeLinear' for node in model.graph.node)
    assert (
        next(
            w for w in model.graph.initializer if w.name == 'body.bias'
        ).data_type
        == TensorProto.FLOAT
    )
    assert not any(
        node.name == 'body.bias_DequantizeLinear' for node in model.graph.node
    )
    head = next(node for node in model.graph.node if node.name == '/head/Conv')
    assert head.input[1] == 'head.weight'
    assert (
        next(
            w for w in model.graph.initializer if w.name == 'head.weight'
        ).data_type
        == TensorProto.FLOAT
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        output,
        sess_options=options,
        providers=[
            'CPUExecutionProvider',
        ],
    )
    result = session.run(
        None,
        {'images': np.ones((1, 3, 8, 8), dtype=np.float32)},
    )[0]
    assert result.shape == (1, 2, 8, 8)
    assert np.isfinite(result).all()


def test_fp16_conversion_keeps_float_io(model_path: Path) -> None:
    """FP16 conversion retains the runtime's float input/output contract."""
    output = subject.onnxruntime_quantize_onnx_all_images(
        str(model_path),
        quantize=16,
    )
    model = onnx.load(output)
    onnx.checker.check_model(model)
    assert model.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert (
        model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
    )
    assert any(
        w.data_type == TensorProto.FLOAT16 for w in model.graph.initializer
    )


def test_invalid_quantization_inputs(
    model_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported precision and missing calibration fail before export."""
    with pytest.raises(ValueError, match='INT8 or FP16'):
        subject.onnxruntime_quantize_onnx_all_images(
            str(model_path),
            quantize=4,
        )
    with pytest.raises(ValueError, match='calibration dataset'):
        subject.onnxruntime_quantize_onnx_all_images(
            str(model_path),
            quantize=8,
        )
    monkeypatch.setattr(subject, 'calibration_method', 'unknown')
    with pytest.raises(ValueError, match='Unsupported calibration method'):
        subject.onnxruntime_quantize_onnx_all_images(
            str(model_path),
            quantize=8,
            dataset=CalibrationBatches(),
        )
