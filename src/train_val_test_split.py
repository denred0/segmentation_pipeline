import shutil
from pathlib import Path, PosixPath

from random import shuffle
from typing import List, Tuple
from tqdm import tqdm

from my_utils import get_all_files_in_folder, seed_everything
import config


def train_val_test_split(dataset_dir: Path, val_fraction: float, test_fraction: float) -> None:
    """
    :param dataset_dir: root directory of dataset
    :param val_part: fraction of val dataset
    :param test_fraction: fraction of test dataset
    :return: None
    """

    seed_everything(config.SEED)

    # recreate output dirs
    output_dirs = ["train", "trainannot", "val", "valannot", "test", "testannot"]
    for directory in output_dirs:
        output_dir = Path("data").joinpath("train_val_test_split").joinpath("output").joinpath(directory)
        if output_dir.exists() and output_dir.is_dir():
            shutil.rmtree(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    images_paths = get_all_files_in_folder(dataset_dir.joinpath("images"), ["*.jpg"])
    shuffle(images_paths)

    train_size, val_size, test_size = get_sizes_from_fraction(images_paths, val_fraction, test_fraction)

    train_images = images_paths[:train_size]
    move_to_folder(train_images, "train")

    val_images = images_paths[train_size:(train_size + val_size)]
    move_to_folder(val_images, "val")

    test_images = images_paths[(train_size + val_size):]
    move_to_folder(test_images, "test")


def move_to_folder(images_to_move: List, schema: str) -> None:
    for img_path in tqdm(images_to_move, desc="Move to " + schema):
        file_name = str(img_path).split("/")[-1]
        root_path = "/".join(str(img_path).split("/")[:-2])

        shutil.copy(img_path, "data/train_val_test_split/output/" + schema)
        shutil.copy(Path(root_path).joinpath("masks").joinpath(file_name),
                    "data/train_val_test_split/output/" + schema + "annot")


def get_sizes_from_fraction(dataset: List, val_fraction: float, test_fraction: float) -> Tuple[int, int, int]:
    """
    Получаем из датасета размеры большой и маленькой выборки
    :param dataset: любой объект, имеющий метод len()
    :param fraction: желательно передавать в пределах 0 < x <= 0.5
    :return: два значения: размер большой выборки, размер маленькой выборки
    """
    full_len = len(dataset)
    test_val_len = int(full_len * (val_fraction + test_fraction))
    train_len = full_len - test_val_len

    test_fraction_from_val = test_fraction / (val_fraction + test_fraction)
    test_len = int(test_val_len * test_fraction_from_val)
    val_len = test_val_len - test_len

    return train_len, val_len, test_len


if __name__ == "__main__":
    dataset_dir = "data/train_val_test_split/input"

    train_val_test_split(Path(dataset_dir), val_fraction=0.2, test_fraction=0.1)
