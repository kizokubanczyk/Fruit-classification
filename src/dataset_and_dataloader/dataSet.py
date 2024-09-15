from torch.utils.data import Dataset
import cv2
import pandas as pd
from torchvision import transforms


class  ClassificationDataset(Dataset):
    def __init__ (self, image_directory_paths: list[str], labels: list[str], transform = None):

        self.image_directory_paths = image_directory_paths
        self.labels = labels
        self.transform = transform


    def __len__(self):
         return (len(self.image_directory_paths))

    def __getitem__(self, index) -> tuple[str, pd.Series]:

        image_path = self.image_directory_paths[index]
        label = self.labels[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Konwersja BGR do RGB

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


def createDataSet(X_train: str, X_val: str, X_test:str, y_train:str, y_val:str, y_test:str) ->tuple [
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



