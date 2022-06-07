import os
import onnx
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.onnx
import segmentation_models_pytorch as smp
import config

from my_utils import get_model_and_preprocessing

weights_path = "logs/resnet18_Unet/exp_1/e4_loss_0.1352_iou_score_0.8894.pth"
model, _ = get_model_and_preprocessing(arch=config.ARCH, mode="eval", weights_path=weights_path)
# print(model)

model.cpu().eval()

# width = config.IMAGE_WIDTH_TRAIN_PADDED
# height = config.IMAGE_HEIGHT_TRAIN_PADDED

width = 256
height = 256
# batch_size = 4
batch_size = 1

x = torch.randn(1, 3, width, height)

# dynamic_axes = {'input': [0]}

torch.onnx.export(model,
                  x,
                  # "segm_model.onnx",
                  f"data/onnx_convert/smoke_{config.ENCODER}_{config.ARCH}_{width}_{height}_batch_{batch_size}_iou_0.8894.onnx",
                  opset_version=11,
                  do_constant_folding=True,
                  export_params=True,
                  keep_initializers_as_inputs=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}}
                  )

# onnx.checker.check_model("123.onnx")
#
# model = onnx.load('123.onnx')
# print()
# model = onnx.load("123.onnx")
# model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?'
# onnx.save(model, "123.onnx")
