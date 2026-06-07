from __future__ import annotations

from typing import Any
from typing import cast

import torch
from ultralytics import YOLO

model = YOLO('../../models/pt/best_yolo11m.pt')
yolo_model = cast(Any, model.model)

# PyTorch QAT setup
yolo_model.fuse()
yolo_model.train()

yolo_model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
torch.ao.quantization.prepare_qat(yolo_model, inplace=True)

model.train(data='cv_dataset/data_fold2.yaml', epochs=30)  # , lr0=1e-4)

model_int8 = torch.ao.quantization.convert(yolo_model.eval(), inplace=False)
torch.save(model_int8.state_dict(), 'best_qat.pt')

# yolo export model=../../models/pt/best_yolo11s.pt format=engine
# batch=8 int8=True data=cv_dataset/data_fold2.yaml

# yolo detect train data=cv_dataset/data_fold2.yaml
# model=../../models/pt/best_yolo11s.pt epochs=100
