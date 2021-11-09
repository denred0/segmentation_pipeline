import os

import cv2
import numpy as np
import shutil

from pathlib import Path
from tqdm import tqdm

from my_utils import get_all_files_in_folder, recreate_folders


def create_examples_emply_mask(input_dir: str, output_dir: str, image_ext: str) -> None:
    images = get_all_files_in_folder(Path(input_dir).joinpath("images"), ["*.png"])

    for image_path in tqdm(images):
        image = cv2.imread(str(image_path))

        h, w = image.shape[:2]

        mask = np.zeros((h, w), dtype=int)

        cv2.imwrite(str(Path(output_dir).joinpath("images").joinpath(str(image_path.stem) + "_empty_mask." + image_ext)),
                    image)
        cv2.imwrite(str(Path(output_dir).joinpath("masks").joinpath(str(image_path.stem) + "_empty_mask." + image_ext)),
                    mask)


if __name__ == "__main__":
    input_dir = "data/create_examples_empty_mask/input"
    output_dir = "data/create_examples_empty_mask/output"
    image_ext = "png"

    recreate_folders(Path(output_dir), ["images", "masks"])
    create_examples_emply_mask(input_dir, output_dir, image_ext)
