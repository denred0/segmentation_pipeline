import os
import shutil
import cv2
import numpy as np
from pathlib import Path, PosixPath

from random import shuffle
from typing import List, Tuple
from tqdm import tqdm

from my_utils import get_all_files_in_folder, seed_everything, recreate_folders
import config


def train_val_test_split_one_class(input_dir: Path, output_dir_root: Path, val_fraction: float,
                                   test_fraction: float) -> None:
    seed_everything(config.SEED)

    images_paths = get_all_files_in_folder(input_dir.joinpath("images"), ["*" + config.IMAGE_EXTENSION_OUTPUT])
    shuffle(images_paths)

    train_size, val_size, test_size = get_sizes_from_fraction(images_paths, val_fraction, test_fraction)

    train_images = images_paths[:train_size]
    copy_image_and_mask_to_split_folder(output_dir_root, train_images, "train")

    val_images = images_paths[train_size:(train_size + val_size)]
    copy_image_and_mask_to_split_folder(output_dir_root, val_images, "val")

    test_images = images_paths[(train_size + val_size):]
    copy_image_and_mask_to_split_folder(output_dir_root, test_images, "test")


def copy_image_and_mask_to_split_folder(root_output_dir: Path, images_to_move: List, schema: str) -> None:
    for img_path in tqdm(images_to_move, desc="Move to " + schema):
        file_name = str(img_path).split("/")[-1]
        root_path = "/".join(str(img_path).split("/")[:-2])

        shutil.copy(img_path, root_output_dir.joinpath(schema))
        shutil.copy(Path(root_path).joinpath("masks").joinpath(file_name),
                    root_output_dir.joinpath(schema + "annot"))


def copy_images_to_appropriate_class(input_dir: str, output_classes_dir: str) -> None:
    masks_paths = get_all_files_in_folder(Path(input_dir).joinpath("masks"), ["*" + config.IMAGE_EXTENSION_OUTPUT])

    for mask_path in tqdm(masks_paths):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        label = config.CLASSES[sorted(np.unique(mask))[1]]

        shutil.copy(mask_path, Path(output_classes_dir).joinpath(label).joinpath("masks"))
        shutil.copy(Path(os.sep.join(str(mask_path).split(os.sep)[:-2])).joinpath("images").joinpath(mask_path.name),
                    Path(output_classes_dir).joinpath(label).joinpath("images"))


def get_sizes_from_fraction(dataset: List, val_fraction: float, test_fraction: float) -> Tuple[int, int, int]:
    full_len = len(dataset)
    test_val_len = int(full_len * (val_fraction + test_fraction))
    train_len = full_len - test_val_len

    test_fraction_from_val = test_fraction / (val_fraction + test_fraction)
    test_len = int(test_val_len * test_fraction_from_val)
    val_len = test_val_len - test_len

    return train_len, val_len, test_len


def analyze_split_proportion(data_dir: str) -> None:
    all_labels_dict = {}

    train_masks_paths = get_all_files_in_folder(Path(data_dir).joinpath("trainannot"),
                                                ["*" + config.IMAGE_EXTENSION_OUTPUT])
    train_labels_dict = {}
    for train_mask_path in tqdm(train_masks_paths, desc="Analyze train"):
        mask = cv2.imread(str(train_mask_path), cv2.IMREAD_GRAYSCALE)

        for label in np.unique(mask):
            train_labels_dict[label] = train_labels_dict.get(label, 0) + 1
            all_labels_dict[label] = all_labels_dict.get(label, 0) + 1

    val_masks_paths = get_all_files_in_folder(Path(data_dir).joinpath("valannot"),
                                              ["*" + config.IMAGE_EXTENSION_OUTPUT])
    val_labels_dict = {}
    for val_mask_path in tqdm(val_masks_paths, desc="Analyze val"):
        mask = cv2.imread(str(val_mask_path), cv2.IMREAD_GRAYSCALE)

        for label in np.unique(mask):
            val_labels_dict[label] = val_labels_dict.get(label, 0) + 1
            all_labels_dict[label] = all_labels_dict.get(label, 0) + 1

    test_masks_paths = get_all_files_in_folder(Path(data_dir).joinpath("testannot"),
                                               ["*" + config.IMAGE_EXTENSION_OUTPUT])
    test_labels_dict = {}
    for test_mask_path in tqdm(test_masks_paths, desc="Analyze test"):
        mask = cv2.imread(str(test_mask_path), cv2.IMREAD_GRAYSCALE)

        for label in np.unique(mask):
            test_labels_dict[label] = test_labels_dict.get(label, 0) + 1
            all_labels_dict[label] = all_labels_dict.get(label, 0) + 1

    for split_type, data_dict in zip(["train", "val", "test"], [train_labels_dict, val_labels_dict, test_labels_dict]):
        print(f"\n{split_type} classes proportion:")
        for (key, value) in data_dict.items():
            print(f"Class {config.CLASSES[key]}: {round(data_dict[key] / all_labels_dict[key] * 100, 1)}%")


if __name__ == "__main__":
    input_dir = "data/train_val_test_split/input"

    one_class_per_image = True

    if one_class_per_image:
        output_dir = "data/train_val_test_split/output/train_val_test_split_one_class_per_image/train_val_test"
        recreate_folders(Path(output_dir), ["train", "trainannot", "val", "valannot", "test", "testannot"])

        output_classes_dir = "data/train_val_test_split/output/train_val_test_split_one_class_per_image/classes"
        recreate_folders(Path(output_classes_dir), config.CLASSES[1:])

        for label in config.CLASSES[1:]:
            recreate_folders(Path(output_classes_dir).joinpath(label), ["images", "masks"])

        copy_images_to_appropriate_class(input_dir, output_classes_dir)

        for label in config.CLASSES[1:]:
            input_dir_class = "data/train_val_test_split/output/train_val_test_split_one_class_per_image/classes" + os.sep + label
            train_val_test_split_one_class(Path(input_dir_class), Path(output_dir), val_fraction=0.1, test_fraction=0.1)

        analyze_split_proportion(output_dir)

    else:
        output_dir = "data/train_val_test_split/output/train_val_test_split_one_class"
        recreate_folders(Path(output_dir), ["train", "trainannot", "val", "valannot", "test", "testannot"])
        train_val_test_split_one_class(Path(input_dir), Path(output_dir), val_fraction=0.1, test_fraction=0.1)

        analyze_split_proportion(output_dir)
