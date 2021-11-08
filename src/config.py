import torch
import seaborn as sns
import numpy as np

CLASSES = ['unlabelled', 'smoke']
# CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 'tree', 'signsymbol', 'fence', 'car', 'pedestrian',
#            'bicyclist', 'unlabelled', ]


CLASS_INDEXES = [index for index, value in enumerate(CLASSES)]

palette_sns = sns.color_palette(palette='bright', n_colors=len(CLASSES))
PALETTE = []
for p in palette_sns:
    PALETTE.append([int(np.clip(x * 255, 0, 255)) for x in p])

# PALETTE = reversed(PALETTE)

IMAGE_HEIGHT = 1080
IMAGE_WIDTH = 1920

IMAGE_HEIGHT_RESIZE = 864
IMAGE_WIDTH_RESIZE = 864

IMAGE_HEIGHT_PADDED = IMAGE_HEIGHT_RESIZE if IMAGE_HEIGHT_RESIZE % 32 == 0 else (IMAGE_HEIGHT_RESIZE // 32 + 1) * 32
IMAGE_WIDTH_PADDED = IMAGE_WIDTH_RESIZE if IMAGE_WIDTH_RESIZE % 32 == 0 else (IMAGE_WIDTH_RESIZE // 32 + 1) * 32

IMAGE_EXTENSION_INPUT = ".jpg"
IMAGE_EXTENSION_OUTPUT = ".png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENCODER = "se_resnext101_32x4d"  # "se_resnext101_32x4d"
ARCH = "FPN"
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = "sigmoid"

LEARNING_RATE = 0.001
BATCH_SIZE = 2
EPOCH = 10
SCHEDULER = "ExponentialLR"
EARLY_STOP_PATIENCE = 3

MIN_LR = 1e-6,
T_MAX = 100,
T_0 = 25,
WARMUP_EPOCHS = 0,
WEIGHT_DECAY = 1e-4
GAMMA = 0.95

X_TRAIN_DIR = "data/train/train"
Y_TRAIN_DIR = "data/train/trainannot"
X_VALID_DIR = "data/train/val"
Y_VALID_DIR = "data/train/valannot"

WEIGHTS_SAVE_DIR = "logs"

SEED = 42
