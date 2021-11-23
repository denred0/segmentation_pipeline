import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.onnx
import segmentation_models_pytorch as smp
import config

from my_utils import get_model_and_preprocessing

# ENCODER = config.ENCODER
# CLASSES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
# CLASSES = ["1"]
# ACTIVATION = config.ACTIVATION  # could be None for logits or 'softmax2d' for multicalss segmentation


weights_path = "logs/FPN_inceptionv4/exp_10/e2_loss_0.0186_iou_score_0.9878.pth"
model, _ = get_model_and_preprocessing(mode="eval", weights_path=weights_path)
model.to("cpu")

# width = config.IMAGE_WIDTH_TRAIN_PADDED
# height = config.IMAGE_HEIGHT_TRAIN_PADDED

width = 864
height = 864

x = torch.randn(1, 3, width, height, requires_grad=True)

torch.onnx.export(model,
                  x,
                  "data/onnx_convert/smoke_" + str(width) + "_" + str(height) + "_" + config.ENCODER + ".onnx",
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  )
