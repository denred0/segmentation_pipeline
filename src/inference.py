import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm

from augmentation import get_validation_transformation, get_preprocessing
from my_utils import get_all_files_in_folder, recreate_folders, get_model_and_preprocessing

import config


def inference_main(images_dir: Path, root_output_dir: Path, weights_path: str) -> None:
    recreate_folders(root_output_dir, ["images", "masks", "masks_rgb", "visualization", "with_confidence"])
    model, pretrain_prepocessing = get_model_and_preprocessing(mode="eval", weights_path=weights_path)
    inference_images(images_dir, root_output_dir, model, pretrain_prepocessing)


def inference_images(images_dir: Path, root_output_dir: Path, model, pretrain_prepocessing) -> None:
    images = get_all_files_in_folder(images_dir, ['*'])

    sm = torch.nn.Softmax()

    for image_path in tqdm(images):
        image_original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image_original_height, image_original_width = image_original.shape[:2]

        x_tensor = process_image_to_tensor(image_original, pretrain_prepocessing)
        predicted_mask_tensor = model.predict(x_tensor)
        predicted_mask_probabilities = sm(predicted_mask_tensor).max(dim=1)[0].detach().cpu().numpy().squeeze()
        predicted_mask_classes = predicted_mask_tensor.argmax(dim=1).detach().cpu().numpy().squeeze()

        predicted_mask_rgb = np.zeros((predicted_mask_classes.shape[0], predicted_mask_classes.shape[1], 3), dtype=int)

        predicted_mask_classes = cv2.resize(predicted_mask_classes.astype('float'),
                                            (image_original_width, image_original_height)).astype('int')
        predicted_mask_probabilities = cv2.resize(predicted_mask_probabilities.astype('float'),
                                                  (image_original_width, image_original_height)).astype('float')

        predicted_mask_rgb = cv2.resize(predicted_mask_rgb.astype('float'),
                                        (image_original_width, image_original_height))

        # fill predicted_mask_rgb with colors of classes
        predicted_mask_classes[(predicted_mask_probabilities < config.THRESHOLD_INFERENCE)] = 0
        for class_ind, class_color_rgb in zip(config.CLASS_INDEXES, config.PALETTE):
            predicted_mask_rgb[(predicted_mask_classes == class_ind), :] = class_color_rgb

        # get visualization of prediction
        image_for_visualization = get_image_for_visualization(image_original, predicted_mask_classes)

        image_with_confidence = image_for_visualization.copy()
        image_with_confidence[(predicted_mask_probabilities < config.THRESHOLD_INFERENCE),
        :] = config.NOT_CONFIDENT_COLOR

        # save results of prediction
        cv2.imwrite(str(root_output_dir.joinpath("images").joinpath(str(image_path.name))), image_original)
        cv2.imwrite(str(root_output_dir.joinpath("masks").joinpath(str(image_path.name))), predicted_mask_classes)
        cv2.imwrite(str(root_output_dir.joinpath("masks_rgb").joinpath(str(image_path.name))), predicted_mask_rgb)
        cv2.imwrite(str(root_output_dir.joinpath("visualization").joinpath(str(image_path.name))),
                    image_for_visualization)
        cv2.imwrite(str(root_output_dir.joinpath("with_confidence").joinpath(str(image_path.name))),
                    image_with_confidence)


def get_image_for_visualization(image, predicted_mask):
    original_image_with_mask = image.copy()
    image_for_visualization = image.copy()

    for class_ind, class_color_rgb in zip(config.CLASS_INDEXES, config.PALETTE):
        if class_ind == 0:
            continue
        else:
            original_image_with_mask[(predicted_mask == class_ind), :] = class_color_rgb

    alpha = 0.3
    cv2.addWeighted(original_image_with_mask, alpha, image_for_visualization, 1 - alpha, 0, image_for_visualization)

    return image_for_visualization


def process_image_to_tensor(image, pretrain_prepocessing):
    valid_transforms = get_validation_transformation()
    image_transformed = valid_transforms(image=image)['image']

    preprocessing = get_preprocessing(pretrain_prepocessing)
    image_preprocessed = preprocessing(image=image_transformed)['image']

    x_tensor = torch.from_numpy(image_preprocessed).to(config.DEVICE).unsqueeze(0)

    return x_tensor


if __name__ == '__main__':
    images_dir = "data/train/test"
    # images_dir = "data/inference/input"
    root_output_dir = "data/inference/output"
    weights_path = "logs/FPN_inceptionv4/exp_64/e7_loss_0.0952_iou_score_0.9094.pth"

    inference_main(Path(images_dir), Path(root_output_dir), weights_path)

    # python3  src/segmentation_inference.py --data_file data.txt --classes_file classes.txt --weights weights/best_model.pth
