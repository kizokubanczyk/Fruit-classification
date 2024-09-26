import torch

import torch.nn as nn
import torch.optim as optim
from keras.src.utils.module_utils import tensorflow
from sympy.physics.units import inches
from tensorflow.python.keras.engine.data_adapter import DataAdapter
from torch.utils.data import DataLoader
from torchvision.models.quantization import resnet18

from src.model import covalent_network
from torchvision.models import mobilenet_v2

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import models
import torch.nn.functional as F
import yaml
from src.model.early_stopping import EarlyStopping
from src.save_results import  save_accuracy


num_classes = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True).to(device) #bratch size 16, lr = 0,0001, images 256 x 256
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10).to(device)

early_stopping = EarlyStopping(patience=8, min_delta=0.01)

best_val_Accuracy = None

def train_model(train_dataLoader,val_dataLoader) -> None:

    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    num_epochs = 1000

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_dataLoader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_dataLoader)}')

        running_loss = 0.0
        correct = 0
        total = 0

        model.eval()

        with torch.no_grad():
            for images, labels in val_dataLoader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(probabilities, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(val_dataLoader)
        accuracy = 100 * correct / total

        print(f'Validation Loss: {avg_loss}')
        print(f'Validation Accuracy: {accuracy}%')

        global best_val_Accuracy

        if best_val_Accuracy is None:
            best_val_Accuracy = accuracy
        elif accuracy > best_val_Accuracy:
            best_val_Accuracy = accuracy

        early_stopping(val_loss=avg_loss, )

        if early_stopping.early_stop:

            with (open("../config.yaml", 'r') as file):
                config_data = yaml.safe_load(file)
                path_file = config_data.get('path_to_scores')
                save_accuracy(best_val_Accuracy, "val accuracy", path_file)

            print("Early stopping triggered")
            # tu dojdzie do zapisu
            break

def test_model(test_dataLoader) -> None:

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_dataLoader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total

    with (open("../config.yaml", 'r') as file):
        config_data = yaml.safe_load(file)
        path_file = config_data.get('path_to_scores')
        save_accuracy(accuracy, "test accuracy", path_file)

    print(f'Accuracy of the model on the test images: {accuracy}%')



def classify_external_image(external_dataLoader) -> None:
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in external_dataLoader:

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total


        print(f'Accuracy of the model on the external images: {accuracy}%')