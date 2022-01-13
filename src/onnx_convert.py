import os
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.onnx
import segmentation_models_pytorch as smp
import config

from my_utils import get_model_and_preprocessing

weights_path = "logs/FPN_inceptionv4/exp_64/e7_loss_0.0952_iou_score_0.9094.pth"
model, _ = get_model_and_preprocessing(mode="eval", weights_path=weights_path)
# print(model)

model.to("cpu")

# width = config.IMAGE_WIDTH_TRAIN_PADDED
# height = config.IMAGE_HEIGHT_TRAIN_PADDED

width = 256
height = 256

x = torch.randn(1, 3, width, height, requires_grad=True)

torch.onnx.export(model,
                  x,
                  "data/onnx_convert/smoke_" + str(width) + "_" + str(height) + "_" + config.ENCODER + "_new.onnx",
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  )

# net = cv2.dnn.readNetFromONNX('keras.onnx')
