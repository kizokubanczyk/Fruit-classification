from torch.utils.data import Dataset
import os
import cv2
import pandas as pd

class  ClassificationDataset(Dataset):
    def __init__ (self, image_data_dir, transform=None):

        self.fruit_images = {}
        self.fruits_failed_to_load = 0
        self.folders_to_skip = {"test", "train"}

        for dirname, _, filenames in os.walk(image_data_dir):
            folder_name = os.path.basename(dirname)

            if folder_name in self.folders_to_skip:
                continue

            if folder_name and folder_name not in self.fruit_images:
                self.fruit_images[folder_name] = []

            for filename in filenames:
                image_path = os.path.join(dirname, filename)
                image = cv2.imread(image_path)

                self.fruit_images[folder_name].append(image)

    def __len__(self):
         return sum(len(values) for values in self.fruit_images.values())

    def __getitem__(self, index) -> tuple[str, pd.Series]:
        index_sum = 0
        key_number = 0

        for list_of_fruit in self.fruit_images.values():
            previous_sum = index_sum
            index_sum += len(list_of_fruit)

            if index_sum > index:
                keys = list(self.fruit_images.keys())
                correct_key = keys[key_number]
                element_index = index - previous_sum
                break

            key_number += 1

        return correct_key, list_of_fruit[element_index] # lable, sample


