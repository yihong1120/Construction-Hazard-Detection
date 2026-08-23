# TensorRT Engine 重建指南

## 问题说明

当升级 TensorRT 版本后，旧的 `.engine` 文件将不兼容，需要重新构建。

错误信息：
```
The engine plan file is not compatible with this version of TensorRT,
expecting library version 10.14.1.48 got ...
```

## 解决方案

批量重建與受管 deployment 的工具屬於私有 operator tooling，不隨 application
source 發布。需要在本機重建單一 engine 時，可直接使用 Ultralytics CLI。

### 手動使用 Ultralytics CLI

```bash
# 从 PT 文件导出
uv run yolo export model=models/pt/best_yolo11m.pt format=engine device=0 quantize=8 data=coco128.yaml

# 然后移动生成的 .engine 文件到正确位置
mv models/pt/best_yolo11m.engine models/int8_engine/
```

## 导出选项说明

- `format=engine`: 导出为 TensorRT engine 格式
- `device=0`: 使用 GPU 0
- `quantize=8`: 启用 INT8 量化（需要校准数据）
- `half=True`: 使用 FP16 精度（更快但可能略微降低精度）
- `data=...yaml`: INT8 校准数据集（建议使用 repo 内的 `examples/YOLO_train/cv_dataset/data.yaml` 或 `data_fold*.yaml`）

## 检查进度

```bash
# 查看新生成的 engine 文件
ls -lh models/int8_engine/*.engine

# 查看文件修改时间
ls -lt models/int8_engine/*.engine
```

## 常见问题

### Q: INT8 校准需要多长时间？
A: 通常 1-3 分钟，取决于模型大小和GPU性能。

### Q: 可以不使用 INT8 吗？
A: 可以，设置 `half=True` 使用 FP16，或两者都不设置使用 FP32。但 INT8 通常提供最佳的速度/精度平衡。

### Q: 校准数据集从哪里来？
A: 在 `data=` 指定 repo 内的 `examples/YOLO_train/cv_dataset/` 数据集 YAML；若使用
`coco128.yaml`，Ultralytics 可能会下载该数据集。

### Q: 导出失败怎么办？
A:
1. 检查 TensorRT 是否正确安装: `uv pip show tensorrt`
2. 确保有足够的 GPU 内存
3. 尝试不使用 INT8: 删除 `quantize=8` 和 `data=` 参数

## 验证导出

导出完成后，启动 YOLO Server API 进行测试：

```bash
uv run uvicorn examples.YOLO_server_api.app:app --host 127.0.0.1 --port 8000
```

如果看到类似以下输出，说明导出成功：
```
Loading models/int8_engine/best_yolo11m.engine for TensorRT inference...
[TRT] [I] Loaded engine size: 24 MiB
✅ TensorRT engine loaded successfully
```
