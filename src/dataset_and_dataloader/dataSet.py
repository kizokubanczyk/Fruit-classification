from torch.utils.data import Dataset
import cv2
import torch
from torchvision import transforms
from PIL import Image


class  ClassificationDataset(Dataset):
    def __init__ (self, image_directory_paths: list[str], labels: list[str], transform = None):

        if len(image_directory_paths) != len(labels):
            raise ValueError("The number of images and labels must be the same")

        self.image_directory_paths = image_directory_paths
        self.labels = labels
        self.transform = transform


    def __len__(self):
         return (len(self.image_directory_paths))

    def __getitem__(self, index) -> tuple[Image.Image, torch.Tensor]:
        image_path = self.image_directory_paths[index]
        label = self.labels[index]

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"The image {image_path} was not found")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Konwersja BGR do RGB
        image = Image.fromarray(image)  # Konwersja z numpy.ndarray na PIL.Image

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label)  # Konwersja etykiety na tensor PyTorch

        return image, label


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


def createDataSet(X_train: list[str], X_val: list[str], X_test: list[str],
                  y_train: list[str], y_val: list[str], y_test: list[str]) -> tuple[
    ClassificationDataset, ClassificationDataset, ClassificationDataset]:

    train_dataset = ClassificationDataset(
        image_directory_paths = X_train,
        labels = y_train,
        transform = transform
    )

    val_dataset = ClassificationDataset(
        image_directory_paths = X_val,
        labels = y_val,
        transform = transform
    )

    test_dataset = ClassificationDataset(
        image_directory_paths = X_test,
        labels = y_test,
        transform = transform
    )

    return train_dataset, val_dataset, test_dataset



