from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import onnx
import torch
from onnxruntime.quantization import CalibrationDataReader
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.exporter import Exporter
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.export import engine as engine_export


base_dir = Path(__file__).resolve().parent
default_pt_file = (base_dir / '../../models/pt/best_yolo26x.pt').resolve()
default_trt_file = (
    base_dir / '../../models/int8_engine/best_yolo26x_int8.engine'
).resolve()
default_calibration_data_path = (base_dir / 'cv_dataset/data.yaml').resolve()


class AllImagesCalibrationReader(CalibrationDataReader):
    def __init__(self, input_name: str, dataloader) -> None:
        self.input_name = input_name
        self.dataloader = dataloader
        self.max_batches = len(dataloader)
        self.total_images = len(getattr(dataloader, 'dataset', []))
        self.batch_size = int(getattr(dataloader, 'batch_size', 1) or 1)
        self.progress = None
        self.rewind()

    def __len__(self) -> int:
        return self.max_batches

    def rewind(self) -> None:
        self._close_progress()
        if hasattr(self.dataloader, 'reset'):
            self.dataloader.reset()
        self.iterator = iter(self.dataloader)
        self.seen_batches = 0
        self.seen_images = 0

    def _close_progress(self) -> None:
        if self.progress is not None:
            self.progress.close()
            self.progress = None

    def set_range(self, start_index: int, end_index: int) -> None:
        raise NotImplementedError(
            'Range slicing is not needed for this calibration reader.',
        )

    def _to_input(self, batch) -> dict[str, object]:
        images = batch['img'].to(torch.float32) / 255.0
        return {self.input_name: images.cpu().numpy()}

    def get_first(self) -> dict[str, object]:
        self.rewind()
        first = self._to_input(next(self.iterator))
        self.rewind()
        return first

    def get_next(self) -> dict[str, object] | None:
        if self.seen_batches >= self.max_batches:
            self._close_progress()
            return None

        if self.progress is None:
            self.progress = tqdm(
                total=self.total_images,
                desc='ModelOpt calibration',
                unit='img',
                dynamic_ncols=True,
            )

        batch = next(self.iterator)
        self.seen_batches += 1
        batch_size = int(batch['img'].shape[0])
        self.seen_images += batch_size
        self.progress.update(batch_size)

        if self.seen_batches >= self.max_batches:
            self._close_progress()

        return self._to_input(batch)


original_modelopt_quantize_onnx = getattr(
    engine_export, 'modelopt_quantize_onnx', None,
)
original_get_int8_calibration_dataloader = Exporter.get_int8_calibration_dataloader


def modelopt_quantize_onnx_all_images(
    onnx_file: str,
    quantize: int | str | None = None,
    dataset=None,
    shape: tuple[int, int, int, int] = (1, 3, 640, 640),
    dynamic: bool = False,
    prefix: str = '',
) -> str:
    if quantize != 8:
        if original_modelopt_quantize_onnx is None:
            raise RuntimeError(
                'This Ultralytics version does not expose modelopt_quantize_onnx.',
            )
        return original_modelopt_quantize_onnx(
            onnx_file,
            quantize=quantize,
            dataset=dataset,
            shape=shape,
            dynamic=dynamic,
            prefix=prefix,
        )

    if dataset is None:
        raise ValueError(
            'INT8 ModelOpt quantization requires a calibration dataset.',
        )

    check_requirements('nvidia-modelopt[onnx]>=0.44')
    from modelopt.onnx.quantization import quantize as modelopt_quantize

    input_name = onnx.load(
        onnx_file, load_external_data=False,
    ).graph.input[0].name
    out_file = str(Path(onnx_file).with_suffix('.int8.onnx'))
    reader = AllImagesCalibrationReader(input_name, dataset)
    calibration_shape = (reader.batch_size, *shape[1:])
    kwargs = (
        {
            'calibration_shapes':
            f'{input_name}:{"x".join(str(d) for d in calibration_shape)}',
        }
        if dynamic
        else {}
    )

    LOGGER.info(
        f'{prefix} quantizing ONNX to INT8 with ModelOpt using all '
        f'{reader.total_images} calibration images...',
    )
    modelopt_quantize(
        onnx_file,
        quantize_mode='int8',
        calibration_data_reader=reader,
        calibration_method='max',
        calibration_eps=['cpu'],
        output_path=out_file,
        **kwargs,
    )
    return out_file


if original_modelopt_quantize_onnx is not None:
    engine_export.modelopt_quantize_onnx = modelopt_quantize_onnx_all_images


def set_calibration_batch_size(calib_batch: int) -> None:
    def get_int8_calibration_dataloader(self, prefix: str = ''):
        export_batch = int(self.args.batch)
        self.args.batch = calib_batch
        try:
            LOGGER.info(
                f'{prefix} using calibration batch={calib_batch} '
                f'for engine batch profile={export_batch}',
            )
            return original_get_int8_calibration_dataloader(self, prefix)
        finally:
            self.args.batch = export_batch

    Exporter.get_int8_calibration_dataloader = get_int8_calibration_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Export an INT8 TensorRT engine with all calibration images.',
    )
    parser.add_argument('--model', type=Path, default=default_pt_file)
    parser.add_argument('--output', '-o', type=Path, default=default_trt_file)
    parser.add_argument(
        '--data', type=Path,
        default=default_calibration_data_path,
    )
    parser.add_argument('--device', default=0)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Max/optimization batch for dynamic TensorRT export. Default: 16.',
    )
    parser.add_argument(
        '--calib-batch',
        type=int,
        default=1,
        help='Batch size used only while reading calibration images. Default: 1.',
    )
    parser.add_argument(
        '--workspace',
        type=int,
        default=None,
        help='TensorRT workspace in GB. Default: 2 for dynamic, 16 for static.',
    )
    parser.add_argument(
        '--static',
        action='store_true',
        help='Export a fixed-shape engine instead of dynamic batch/shape.',
    )
    parser.add_argument('--fraction', type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pt_file = args.model.resolve()
    trt_file = args.output.resolve()
    calibration_data_path = args.data.resolve()
    dynamic = not args.static
    workspace = args.workspace if args.workspace is not None else (
        2 if dynamic else 16
    )

    if dynamic and args.batch <= 1:
        raise SystemExit(
            'dynamic TensorRT export needs --batch > 1, for example --batch 16. '
            'Use --static if you really want a fixed batch=1 engine.',
        )
    if args.calib_batch < 1:
        raise SystemExit('--calib-batch must be >= 1.')
    if args.calib_batch > args.batch:
        raise SystemExit('--calib-batch should be <= --batch.')

    print(f'Model: {pt_file}')
    print(f'Calibration data: {calibration_data_path}')
    print(f'Output: {trt_file}')
    print(f'Dynamic batch: {dynamic}, batch profile: {args.batch}')
    print(f'Calibration batch: {args.calib_batch}')
    print(f'TensorRT workspace: {workspace} GB')

    set_calibration_batch_size(args.calib_batch)
    model = YOLO(str(pt_file))
    exported = Path(
        model.export(
            format='engine',
            device=args.device,
            dynamic=dynamic,
            batch=args.batch,
            imgsz=args.imgsz,
            workspace=workspace,
            quantize=8,
            data=str(calibration_data_path),
            fraction=args.fraction,
        ),
    ).resolve()

    trt_file.parent.mkdir(parents=True, exist_ok=True)
    if exported != trt_file:
        shutil.move(str(exported), trt_file)

    print(f'Exported: {trt_file}')


if __name__ == '__main__':
    main()
