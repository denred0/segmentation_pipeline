import torch
import seaborn as sns
import numpy as np

ALL_CLASSES = ['unlabelled', 'smoke']
CLASSES = ['unlabelled', 'smoke']

CLASS_INDEXES = [index for index, value in enumerate(ALL_CLASSES)]

palette_sns = sns.color_palette(palette='bright', n_colors=len(ALL_CLASSES))
PALETTE = []
for p in palette_sns:
    PALETTE.append([int(np.clip(x * 255, 0, 255)) for x in p])

IMAGE_HEIGHT_ORIGINAL = 1080
IMAGE_WIDTH_ORIGINAL = 1920

# IMAGE_HEIGHT_ORIGINAL = 520
# IMAGE_WIDTH_ORIGINAL = 704

# IMAGE_WIDTH_TRAIN = 1216
# IMAGE_HEIGHT_TRAIN = 684

IMAGE_WIDTH_TRAIN = 256
IMAGE_HEIGHT_TRAIN = 256

# IMAGE_HEIGHT_TRAIN = 520
# IMAGE_WIDTH_TRAIN = 704

IMAGE_HEIGHT_TRAIN_PADDED = IMAGE_WIDTH_TRAIN if IMAGE_WIDTH_TRAIN % 32 == 0 else (IMAGE_WIDTH_TRAIN // 32 + 1) * 32
IMAGE_WIDTH_TRAIN_PADDED = IMAGE_WIDTH_TRAIN if IMAGE_WIDTH_TRAIN % 32 == 0 else (IMAGE_WIDTH_TRAIN // 32 + 1) * 32

IMAGE_EXTENSION_INPUT = ".png"
IMAGE_EXTENSION_OUTPUT = ".png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ENCODER_LIST = [""]
ENCODER = "inceptionv4"  # "se_resnext101_32x4d"  # "se_resnext101_32x4d"
ARCH_LIST = ["Unet", "UnetPlusPlus", "MAnet", "Linknet", "FPN", "PSPNet", "DeepLabV3", "DeepLabV3Plus", "PAN"]
ARCH = "Unet"
ENCODER_WEIGHTS = 'imagenet+background'
ACTIVATION = "sigmoid"  # "softmax2d"  # "softmax2d"#"sigmoid"

# LEARNING_RATE = 0.001
LEARNING_RATE = 0.0001
BATCH_SIZE = 2
EPOCH = 30
SCHEDULER = "ExponentialLR"
EARLY_STOP_PATIENCE = 5

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

NOT_CONFIDENT_COLOR = [0, 255, 255]
THRESHOLD_INFERENCE = 0.5
