from random import random

from torch.utils.data import Dataset
import torch
from torchvision import transforms
from PIL import Image


class  ClassificationDataset(Dataset):
    def __init__(self, image_directory_paths, labels, transform=None, augmentation_multiplier=None):
        self.image_paths = image_directory_paths
        self.labels = labels
        self.transform = transform

        self.label_to_index = {label: idx for idx, label in enumerate(set(labels))}
        #if augmentation_multiplier is not None:
            #if augmentation_multiplier > 1:
                #self.image_paths = self.image_paths * augmentation_multiplier
                #self.labels = self.labels * augmentation_multiplier

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.label_to_index[label]
        label = torch.tensor(label, dtype=torch.long)

        return image, label

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.RandomRotation(degrees=90),  # Losowy obrót do 30 stopni,
    transforms.RandomHorizontalFlip(p=0.5),  # 50% szans na odwrócenie poziome
    #transforms.Normalize(mean=[0.5], std=[0.5]) # 90%
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

def createDataSet(sample: str, label: str, augmentation_multiplier: int) -> ClassificationDataset:
    dataset = ClassificationDataset(
        image_directory_paths=sample,
        labels=label,
        transform=transform,
        augmentation_multiplier=augmentation_multiplier
    )
    return dataset
