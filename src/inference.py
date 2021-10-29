import os
import numpy as np
import cv2
import shutil
import torch
import seaborn as sns
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from dataset_wrapper import DatasetWrapper
from augmentation import *
from my_utils import get_all_files_in_folder

import config


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def inference(data_file, classes_file, weights_dir):
    CLASSES = config.CLASSES
    # data_dict = get_data(data_file)

    # create folder for saving predictions
    pred_save_dir = Path(data_dict['prediction_inference_dir'])
    if pred_save_dir.exists() and pred_save_dir.is_dir():
        shutil.rmtree(pred_save_dir)
    Path(pred_save_dir).mkdir(parents=True, exist_ok=True)

    output_dir_images = pred_save_dir.joinpath('inference').joinpath('images')
    if output_dir_images.exists() and output_dir_images.is_dir():
        shutil.rmtree(output_dir_images)
    Path(output_dir_images).mkdir(parents=True, exist_ok=True)

    output_dir_masks = pred_save_dir.joinpath('inference').joinpath('masks')
    if output_dir_masks.exists() and output_dir_masks.is_dir():
        shutil.rmtree(output_dir_masks)
    Path(output_dir_masks).mkdir(parents=True, exist_ok=True)

    output_dir_masks_rgb = pred_save_dir.joinpath('inference').joinpath('masks_rgb')
    if output_dir_masks_rgb.exists() and output_dir_masks_rgb.is_dir():
        shutil.rmtree(output_dir_masks_rgb)
    Path(output_dir_masks_rgb).mkdir(parents=True, exist_ok=True)

    output_dir_vis = pred_save_dir.joinpath('inference').joinpath('visualization')
    if output_dir_vis.exists() and output_dir_vis.is_dir():
        shutil.rmtree(output_dir_vis)
    Path(output_dir_vis).mkdir(parents=True, exist_ok=True)

    # palette_rgb = [[31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40], [148, 103, 189], [140, 86, 75],
    #                [227, 119, 194], [127, 127, 127], [188, 189, 34], [23, 190, 207], [31, 119, 180], [255, 127, 14], ]

    palette = sns.color_palette(palette='bright', n_colors=len(CLASSES))
    palette_rgb = []
    for p in palette:
        palette_rgb.append([int(np.clip(x * 255, 0, 255)) for x in p])

    # DATA_DIR = data_dict['data_dir']
    inference_dir = data_dict['inference_dir']

    ENCODER = data_dict['encoder']
    ENCODER_WEIGHTS = 'imagenet'
    DEVICE = data_dict['device']

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # load best saved checkpoint
    best_model = torch.load(weights_dir)

    # # # create test dataset
    # test_dataset = DatasetWrapper(
    #     x_test_dir,
    #     '',
    #     all_classes=CLASSES,
    #     augmentation=get_validation_augmentation(),
    #     preprocessing=get_preprocessing(preprocessing_fn),
    #     classes=CLASSES,
    # )

    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    #
    # loss = smp.utils.losses.JaccardLoss()
    # metrics = [
    #     smp.utils.metrics.IoU(threshold=0.4),
    # ]

    # evaluate model on test set
    # test_epoch = smp.utils.train.ValidEpoch(
    #     model=best_model,
    #     loss=loss,
    #     metrics=metrics,
    #     device=DEVICE,
    #     verbose=True,
    # )

    # logs = test_epoch.run(test_dataloader)

    # test_dataset_vis = DatasetWrapper(
    #     x_test_dir,
    #     '',
    #     augmentation=get_validation_augmentation(),
    #     preprocessing=get_preprocessing(preprocessing_fn),
    #     all_classes=CLASSES,
    #     classes=CLASSES,
    # )

    images = get_all_files_in_folder(Path(inference_dir), ['*'])
    print()

    for im in tqdm(images):

        image_vis = cv2.imread(str(im), cv2.IMREAD_UNCHANGED)
        img_pred = image_vis.copy()

        augmenting = get_validation_augmentation()
        augmented = augmenting(image=img_pred)['image']

        preprocessing = get_preprocessing(preprocessing_fn)
        preprocessed = preprocessing(image=augmented)['image']

        x_tensor = torch.from_numpy(preprocessed).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = pr_mask.argmax(dim=1).detach().cpu().numpy().squeeze()

        # gt_mask = gt_mask.argmax(axis=0)

        # gt_mask_to_show = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=int)
        pr_mask_to_show = np.zeros((pr_mask.shape[0], pr_mask.shape[1], 3), dtype=int)

        # gt_mask_to_show.fill(255)
        pr_mask_to_show.fill(255)

        class_indexes = [index for index, value in enumerate(CLASSES)]
        for c, rgb in zip(class_indexes, palette_rgb):
            # gt_mask_to_show[(gt_mask == c), :] = rgb
            pr_mask_to_show[(pr_mask == c), :] = rgb

        # add resize because we have padding in our augmentation and the output size of image is different from the original
        h, w = image_vis.shape[:2]
        # gt_mask_to_show = cv2.resize(gt_mask_to_show.astype('float'), (w, h), interpolation=cv2.INTER_AREA)
        pr_mask_to_show = cv2.resize(pr_mask_to_show.astype('float'), (w, h), interpolation=cv2.INTER_AREA)
        pr_mask = cv2.resize(pr_mask.astype('float'), (w, h), interpolation=cv2.INTER_AREA)

        # visualization
        img = np.concatenate((image_vis, pr_mask_to_show), axis=1)

        cv2.imwrite(str(output_dir_images.joinpath(str(im.name))), image_vis)
        cv2.imwrite(str(output_dir_masks.joinpath(str(im.name))), pr_mask)
        cv2.imwrite(str(output_dir_masks_rgb.joinpath(str(im.name))), pr_mask_to_show)
        cv2.imwrite(str(output_dir_vis.joinpath(str(im.name))), img)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default='data.txt', help='default %(default)s')
    parser.add_argument("--classes_file", default='classes.txt', help='default %(default)s')
    parser.add_argument("--weights", default='weights/best_model.pth', help='default %(default)s')

    args = parser.parse_args()

    args.data_file = 'data.txt'
    args.classes_file = 'classes.txt'
    args.weights = 'weights/best_model.pth'

    inference(args.data_file, args.classes_file, args.weights)

    # python3  src/segmentation_inference.py --data_file data.txt --classes_file classes.txt --weights weights/best_model.pth
