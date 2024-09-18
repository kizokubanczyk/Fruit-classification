from torch.utils.data import Dataset
import torch
from torchvision import transforms
from PIL import Image


class  ClassificationDataset(Dataset):
    def __init__(self, image_directory_paths, labels, transform=None):
        self.image_paths = image_directory_paths
        self.labels = labels
        self.transform = transform

        # Utwórz słownik mapujący etykiety na liczby całkowite
        self.label_to_index = {label: idx for idx, label in enumerate(set(labels))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Wczytaj obraz jako PIL.Image
        image = Image.open(image_path).convert('RGB')
        print(image)

        if self.transform:
            image = self.transform(image)

        # Mapowanie etykiety na liczbę całkowitą
        label = self.label_to_index[label]

        # Konwersja etykiety na tensor
        label = torch.tensor(label, dtype=torch.long)

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



