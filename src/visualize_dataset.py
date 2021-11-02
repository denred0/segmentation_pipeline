import matplotlib.pyplot as plt
from dataset_wrapper import DatasetWrapper
from augmentation import get_training_augmentation, get_validation_augmentation, get_preprocessing

import config


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def visualize_dataset():
    dataset = DatasetWrapper(config.X_TRAIN_DIR,
                             config.Y_TRAIN_DIR,
                             all_classes=config.CLASSES,
                             classes=[config.CLASSES[0]])
    for i in range(3):
        image, mask = dataset[i]  # get some sample
        visualize(image=image, mask=mask.squeeze())


def visualize_dataset_augmented():
    augmented_dataset = DatasetWrapper(
        config.X_TRAIN_DIR,
        config.Y_TRAIN_DIR,
        augmentation=get_training_augmentation(),
        all_classes=config.CLASSES,
        classes=[config.CLASSES[0]]
    )

    # same image with different random transforms
    for i in range(3):
        image, mask = augmented_dataset[1]
        visualize(image=image, mask=mask.squeeze(-1))


if __name__ == "__main__":
    # visualize_dataset()
    visualize_dataset_augmented()
