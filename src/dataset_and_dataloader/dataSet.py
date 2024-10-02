from torch.utils.data import Dataset
import torch
from torchvision import transforms
from PIL import Image

def transform_label_to_index(labels_form_class: list[str]) -> list[int]:

    all_labels = ['apple', 'avocado', 'banana', 'cherry', 'kiwi', 'mango', 'orange', 'pinenapple', 'strawberries', 'watermelon']
    label_to_index = []

    for label_form_class in labels_form_class:
        index = 0
        for label_form_all_labels in all_labels:
            if label_form_class == label_form_all_labels:
                label_to_index.append(index)
            index += 1
        if len(label_to_index) == 0 and index == 10:
            label_to_index .append(index)

    return label_to_index

def transform_all_label_to_index()-> dict[str, list[int]]:
    all_labels = ['apple', 'avocado', 'banana', 'cherry', 'kiwi', 'mango', 'orange', 'pinenapple', 'strawberries', 'watermelon']
    labels_and_indexes = {}
    index = 0

    for label in all_labels:
        labels_and_indexes[label] = index
        index += 1

    return labels_and_indexes


class  ClassificationDataset(Dataset):
    def __init__(self, image_directory_paths, labels, transform=None, augmentation_multiplier=None):
        self.image_paths = image_directory_paths
        self.labels = labels
        self.transform = transform

        if augmentation_multiplier is not None:
            if augmentation_multiplier > 1:
                self.image_paths = self.image_paths * augmentation_multiplier
                self.labels = self.labels * augmentation_multiplier

        self.labels_to_index = transform_label_to_index(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels_to_index[idx]
        label = torch.tensor(label, dtype=torch.long)

        return image, label


def transform(additional_augmentations:bool) -> transforms.Compose:
    if additional_augmentations:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=90),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return transform


def createDataSet(sample: str, label: str, augmentation_multiplier: int, additional_augmentations:bool) -> ClassificationDataset:

    dataset = ClassificationDataset(
        image_directory_paths=sample,
        labels=label,
        transform=transform(additional_augmentations=additional_augmentations)                                                                        ,
        augmentation_multiplier=augmentation_multiplier
    )
    return dataset
