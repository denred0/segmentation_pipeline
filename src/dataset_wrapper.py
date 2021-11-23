import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

import numpy as np
import cv2
import config


class DatasetWrapper(BaseDataset):
    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.images_ids = os.listdir(images_dir)
        self.images_paths = [os.path.join(images_dir, image_id) for image_id in self.images_ids]
        self.masks_paths = [os.path.join(masks_dir, image_id) for image_id in self.images_ids]

        # convert str names to class values on masks
        self.class_values = [config.ALL_CLASSES.index(cls.lower()) for cls in classes]
        # self.class_values = list(range(len(classes)))

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_paths[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_paths[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.images_ids)
