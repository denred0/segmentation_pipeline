import torch

CLASSES = ['smoke', 'unlabelled']

ENCODER = "se_resnext101_32x4d"
ARCH = "FPN"
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = "sigmoid"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.0001
BATCH_SIZE = 8
EPOCH = 50
SCHEDULER = "ExponentialLR"
MIN_LR = 1e-6,
T_MAX = 100,
T_0 = 25,
WARMUP_EPOCHS = 0,
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_DECAY = 1e-4
GAMMA = 0.99

X_TRAIN_DIR = "data/train/train"
Y_TRAIN_DIR = "data/train/trainannot"
X_VALID_DIR = "data/train/val"
Y_VALID_DIR = "data/train/valannot"

WEIGHTS_SAVE_DIR = "logs"
