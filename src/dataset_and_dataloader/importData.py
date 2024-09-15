import cv2
import os


def importData(data_path: str) -> tuple:
    fruit_images = {}
    folders_to_skip = {"test", "train"}

    for dirname, _, filenames in os.walk(data_path):
        folder_name = os.path.basename(dirname)

        if folder_name in folders_to_skip:
            continue

        if folder_name and folder_name not in fruit_images:
            fruit_images[folder_name] = []

        for filename in filenames:
            image_path = os.path.join(dirname, filename)
            image = cv2.imread(image_path)
            fruit_images[folder_name].append(image)

    return fruit_images
