from torch.utils.data import Dataset
import yaml
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

class  ClassificationDataset(Dataset):
    def __init__ (self, image_train_dir,  transform=None):
        self.image_train_dir = image_train_dir
        self.transform = transform

        self.fruit_images = []
        self.fruit_labels = []

        for dirname, _, filenames in os.walk(self.image_train_dir):
            folder_name = os.path.basename(dirname)

            if folder_name in "train":
                continue

            for filename in filenames:
                image_path = os.path.join(dirname, filename)
                image = cv2.imread(image_path)
                self.fruit_images.append(image)
                self.fruit_labels.append(folder_name)

    def __len__(self):
         return len(self.fruit_images)

    def __getitem__(self, idx):
        return self.fruit_images[idx]


