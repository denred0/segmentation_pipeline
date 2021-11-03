import os
import json
import cv2
from pathlib import Path
from PIL import Image
import os.path as osp

import numpy as np
import shutil

from tqdm import tqdm

import config
from my_utils import get_all_files_in_folder


def convert_json_to_masks(input_dir, output_dir, input_img_ext, output_img_ext):
    # create folders
    dirpath = output_dir
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

    folders_to_recreate = ["images", "masks", "masks_rgb", "masks_vis"]
    for folder in folders_to_recreate:
        dirpath = output_dir.joinpath(folder)
        Path(dirpath).mkdir(parents=True, exist_ok=True)

    files_json = get_all_files_in_folder(input_dir, ['*.json'])

    for f_json in tqdm(files_json):

        orig_image_path = input_dir.joinpath(f_json.stem + input_img_ext)
        orig_image = cv2.imread(str(orig_image_path), cv2.IMREAD_COLOR)
        orig_image_height, orig_image_width = orig_image.shape[:2]
        mask_image = np.zeros((orig_image_height, orig_image_width, 3), dtype=np.uint8)

        overlay = orig_image.copy()
        output = orig_image.copy()

        with open(str(f_json)) as label_json:
            labels = json.load(label_json)["shapes"]

        for label in labels:
            class_index = config.CLASSES.index(label["label"])
            points = label["points"]

            if label["shape_type"] == "polygon" or label["shape_type"] == "linestrip":
                contour = [np.array(points, dtype=np.int32)]
                cv2.drawContours(mask_image, [contour[0]], 0, (class_index, class_index, class_index), -1)
                cv2.drawContours(overlay, [contour[0]], 0, config.PALETTE[class_index], -1)
            elif label["shape_type"] == "rectangle":
                cv2.rectangle(mask_image, (int(points[0][0]), int(points[0][1])),
                              (int(points[1][0]), int(points[1][1])),
                              (class_index, class_index, class_index), -1)
                cv2.rectangle(overlay, (int(points[0][0]), int(points[0][1])),
                              (int(points[1][0]), int(points[1][1])),
                              (class_index, class_index, class_index), -1)

        alpha = 0.3
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        cv2.imwrite(str(output_dir.joinpath('images').joinpath(f_json.stem + output_img_ext)), orig_image)
        cv2.imwrite(str(output_dir.joinpath('masks').joinpath(f_json.stem + output_img_ext)), mask_image)
        cv2.imwrite(str(output_dir.joinpath('masks_vis').joinpath(f_json.stem + output_img_ext)), output)

        seg_img = Image.fromarray(mask_image[:, :, 0]).convert('P')  # P for png
        seg_img.putpalette(np.array(config.PALETTE, dtype=np.uint8))
        seg_img.save(output_dir.joinpath('masks_rgb').joinpath(f_json.stem + output_img_ext))


if __name__ == '__main__':
    input_dir = Path("data/create_masks_from_json/images_and_jsons")
    output_dir = Path("data/create_masks_from_json/output")
    input_img_ext = config.IMAGE_EXTENSION_INPUT
    output_img_ext = config.IMAGE_EXTENSION_OUTPUT

    convert_json_to_masks(input_dir=input_dir, output_dir=output_dir, input_img_ext=input_img_ext,
                          output_img_ext=output_img_ext)
