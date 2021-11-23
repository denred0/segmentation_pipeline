# https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb

import os
import shutil

import numpy as np
import torch
import segmentation_models_pytorch as smp
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from collections import defaultdict
import matplotlib.pyplot as plt

from dataset_wrapper import DatasetWrapper
from augmentation import get_training_augmentation, get_validation_transformation, get_preprocessing

import config
import losses2
import dice_loss

from my_utils import seed_everything, save_checkpoint, get_model_and_preprocessing


def fetch_scheduler(optimizer: torch.optim) -> torch.optim.lr_scheduler:
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_MAX[0], eta_min=config.MIN_LR[0])

    if config.SCHEDULER == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_MAX[0], eta_min=config.MIN_LR[0])
    elif config.SCHEDULER == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.T_0[0], eta_min=config.MIN_LR[0])
    elif config.SCHEDULER == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.GAMMA)
    elif config.SCHEDULER is None:
        return None

    return scheduler


def create_and_get_experiment_path() -> str:
    experiment_name = config.ARCH + "_" + config.ENCODER
    experiment_number = get_last_exp_number(experiment_name)

    experiment_path = "logs" + os.sep + experiment_name + os.sep + "exp_" + str(experiment_number)

    Path(experiment_path).mkdir(parents=True, exist_ok=True)

    return experiment_path


def train():
    seed_everything(config.SEED)

    experiment_path = create_and_get_experiment_path()

    model, pretrain_prepocessing = get_model_and_preprocessing(mode="train")

    train_loader, valid_loader = get_train_valid_loaders(pretrain_prepocessing)

    train_epoch, valid_epoch, scheduler, loss_name = get_train_valid_epoch(model)

    min_loss = np.Inf
    current_early_stop_patience = 0
    history = defaultdict(list)
    epochs_before_early_stopping = 0

    for epoch in range(1, int(config.EPOCH) + 1):

        epochs_before_early_stopping = epoch

        print('\nEpoch: {}'.format(epoch))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        history['Train Loss'].append(train_logs[loss_name])
        history['Train Acc'].append(train_logs['iou_score'])
        history['Valid Loss'].append(valid_logs[loss_name])
        history['Valid Acc'].append(valid_logs['iou_score'])

        if valid_logs[loss_name] < min_loss:
            min_loss = valid_logs[loss_name]
            score = valid_logs['iou_score']

            save_checkpoint(model, experiment_path, epoch, min_loss, score)
            current_early_stop_patience = 0
        else:
            current_early_stop_patience += 1

            if current_early_stop_patience >= config.EARLY_STOP_PATIENCE:
                print(f"Early stop on epoch {epoch}")
                break

        # if epoch % 2 == 0:
        print("current_early_stop_patience", current_early_stop_patience)

        if scheduler is not None:
            scheduler.step()

    draw_result(range(epochs_before_early_stopping), history['Train Loss'], history['Valid Loss'], history['Train Acc'],
                history['Valid Acc'])


def draw_result(lst_iter, train_loss, val_loss, train_acc, val_acc):
    fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True, sharex=True)

    axis[0].plot(lst_iter, train_loss, '-b', label='Training loss')
    axis[0].plot(lst_iter, val_loss, '-g', label='Validation loss')
    axis[0].set_title("Training and Validation loss")
    axis[0].legend()

    axis[1].plot(lst_iter, train_acc, '-b', label='Training acc')
    axis[1].plot(lst_iter, val_acc, '-g', label='Validation acc')
    axis[1].set_title("Training and Validation acc")
    axis[1].legend()

    # save image
    plt.savefig("result.png")  # should before show method

    # show
    # plt.show()


def get_train_valid_epoch(model):
    # loss = smp.utils.losses.JaccardLoss()
    # class_weights = [0.5, 0.5]
    # loss = losses.CrossentropyND(torch.FloatTensor(class_weights).cuda())
    # loss = losses.CrossentropyND()

    # loss = losses2.Combo_Loss()
    # loss_name = "SSLoss"

    # loss = dice_loss.SSLoss()
    # loss_name = "SSLoss"
    # setattr(loss, '__name__', loss_name)

    loss_name = "bce_with_logits_loss"
    loss = smp.utils.losses.BCEWithLogitsLoss()

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

    return train_epoch, valid_epoch, scheduler, loss_name


def get_train_valid_loaders(pretrain_prepocessing):
    train_dataset = DatasetWrapper(
        config.X_TRAIN_DIR,
        config.Y_TRAIN_DIR,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(pretrain_prepocessing),
        classes=config.CLASSES,
    )

    valid_dataset = DatasetWrapper(
        config.X_VALID_DIR,
        config.Y_VALID_DIR,
        augmentation=get_validation_transformation(),
        preprocessing=get_preprocessing(pretrain_prepocessing),
        classes=config.CLASSES,
    )

    train_loader = DataLoader(train_dataset, batch_size=int(config.BATCH_SIZE), shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

    return train_loader, valid_loader


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
