import random
import os
import shutil
import numpy as np
import torch
import segmentation_models_pytorch as smp

from typing import List
from pathlib import Path

import config


def get_all_files_in_folder(folder, types):
    files_grabbed = []
    for t in types:
        files_grabbed.extend(folder.rglob(t))
    files_grabbed = sorted(files_grabbed, key=lambda x: x)
    return files_grabbed


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def save_checkpoint(model, experiment_path, epoch, loss, iou_score):
    path = Path(experiment_path).joinpath("e{:.0f}_loss_{:.4f}_iou_score_{:.4f}.pth".format(epoch, loss, iou_score))
    torch.save({'model_state_dict': model.state_dict(), }, path)


def recreate_folders(folders_list: List) -> None:
    for directory in folders_list:
        output_dir = Path("data").joinpath("inference").joinpath("output").joinpath(directory)
        if output_dir.exists() and output_dir.is_dir():
            shutil.rmtree(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)


def get_model_and_preprocessing(mode: str, weights_path=""):
    model = smp.FPN(
        encoder_name=config.ENCODER,
        encoder_weights=config.ENCODER_WEIGHTS,
        classes=len(config.CLASSES),
        activation=config.ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(config.ENCODER, config.ENCODER_WEIGHTS)

    if mode == "train":
        model.train()
    else:
        model = load_checkpoint(model, weights_path)
        model.eval()

    return model, preprocessing_fn


def load_checkpoint(model, weights_path: str):
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)

    return model
