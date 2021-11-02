# https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb

import os
import shutil
import torch
import segmentation_models_pytorch as smp
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from dataset_wrapper import DatasetWrapper
from augmentation import get_training_augmentation, get_validation_augmentation, get_preprocessing

import config

from my_utils import seed_everything


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def fetch_scheduler(optimizer):
    if config.SCHEDULER == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_MAX[0], eta_min=config.MIN_LR[0])
    elif config.SCHEDULER == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.T_0[0], eta_min=config.MIN_LR[0])
    elif config.SCHEDULER == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.GAMMA, verbose=False)
    elif config.SCHEDULER == None:
        return None

    return scheduler


def train():
    seed_everything(config.SEED)

    experiment = config.ARCH + "_" + config.ENCODER
    exp_number = get_last_exp_number(experiment)

    x_train_dir = config.X_TRAIN_DIR
    y_train_dir = config.Y_TRAIN_DIR
    x_valid_dir = config.X_VALID_DIR
    y_valid_dir = config.Y_VALID_DIR

    # create segmentation model with pretrained encoder
    model = smp.FPN(
        encoder_name=config.ENCODER,
        encoder_weights=config.ENCODER_WEIGHTS,
        classes=len(config.CLASSES),
        activation=config.ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(config.ENCODER, config.ENCODER_WEIGHTS)

    train_dataset = DatasetWrapper(
        x_train_dir,
        y_train_dir,
        all_classes=config.CLASSES,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=config.CLASSES,
    )

    valid_dataset = DatasetWrapper(
        x_valid_dir,
        y_valid_dir,
        all_classes=config.CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=config.CLASSES,
    )

    train_loader = DataLoader(train_dataset, batch_size=int(config.BATCH_SIZE), shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

    loss = smp.utils.losses.JaccardLoss()
    metrics = [smp.utils.metrics.IoU(threshold=0.4)]
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=float(config.LEARNING_RATE))])
    scheduler = fetch_scheduler(optimizer)

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=config.DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=config.DEVICE,
        verbose=True,
    )

    max_score = 0
    Path("logs").joinpath(experiment).joinpath("exp_" + str(exp_number)).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(config.EPOCH) + 1):

        print('\nEpoch: {}'.format(epoch))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            loss = valid_logs['jaccard_loss']
            weights_path = Path("logs").joinpath(experiment).joinpath("exp_" + str(exp_number)).joinpath(
                "e{:.0f}_jaccard_loss_{:.4f}_iou_score_{:.4f}.pth".format(epoch, loss, max_score))

            torch.save(model, weights_path)

        if epoch == 1:
            if scheduler is not None:
                scheduler.step()


def get_last_exp_number(model_name):
    folders = [x[0] for x in os.walk(os.path.join("logs", model_name))][1:]
    folders = [x.split("/")[-1] for x in folders]
    folders_exp = []
    for f in folders:
        if "exp" in f:
            folders_exp.append(f)

    if not folders_exp:
        return 0
    else:
        return max([int(x.split("_")[1]) for x in folders_exp]) + 1


if __name__ == '__main__':
    train()
