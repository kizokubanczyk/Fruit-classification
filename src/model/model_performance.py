import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
import yaml
from src.model.early_stopping import EarlyStopping
from src.file_operations.save_results import  save_score, save_external_images, save_Confusion_Matrix
from src.dataset_and_dataloader.dataSet import transform_all_label_to_index
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

num_classes = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
early_stopping = EarlyStopping(patience=8, min_delta=0.01)
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

def train_model(train_dataLoader:DataLoader, val_dataLoader:DataLoader) -> None:

    cuda_available = torch.cuda.is_available()
    print(f"Is CUDA available: {cuda_available}")

    if cuda_available:
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device index: {current_device}")
    else:
        print("CUDA is not available, so no current device to display.")

    print(torch.cuda.get_device_name(0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    num_epochs = 1000
    print(f"numer of epochs: {num_epochs}")

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

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_dataLoader)}')

        running_loss = 0.0
        correct = 0
        total = 0
        model.train()

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

        early_stopping(accuracy=accuracy, model=model)

        if early_stopping.early_stop:
            break
        scheduler.step(avg_loss)

def test_model(test_dataLoader: DataLoader) -> None:

    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with open("../config.yaml", 'r') as file:
        config_data = yaml.safe_load(file)
        model_path = config_data.get('path_to_best_model')

    if model_path is not None:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

    with torch.no_grad():
        for images, labels in test_dataLoader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

    cm = confusion_matrix(all_labels, all_predictions)
    print(f'Confusion Matrix:\n{cm}')

    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    print(f'Precision of the model: {precision * 100:.2f}%')
    print(f'Recall of the model: {recall * 100:.2f}%')
    print(f'F1 Score of the model: {f1 * 100:.2f}%')

    with open("../config.yaml", 'r') as file:
        config_data = yaml.safe_load(file)
        path_file = config_data.get('path_to_scores')

    save_Confusion_Matrix(cm, path_file)
    save_score(f"{round(accuracy, 2)}%", "Accuracy on the test dataset", path_file)
    save_score(f"{round(precision * 100, 2)}%", "Precision on the test dataset", path_file)
    save_score(f"{round(recall * 100, 2)}%", "Recall on the test dataset", path_file)
    save_score(f"{round(f1 * 100, 2)}%", "F1 Score on the test dataset", path_file)

def classify_external_image(external_dataLoader: DataLoader) -> None:

    with open("../config.yaml", 'r') as file:
        config_data = yaml.safe_load(file)
        model_path = config_data.get('path_to_best_model')

    if model_path is not None:
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        all_label_as_index = transform_all_label_to_index()
        headers = []

        with torch.no_grad():
            for images, _ in external_dataLoader:
                images = images.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                for i in range(images.size(0)):
                    predicted_label = predicted[i].item()
                    for key_as_label, value_as_label in all_label_as_index.items():
                        if value_as_label == predicted_label:
                            headers.append(key_as_label)

                with (open("../config.yaml", 'r') as file):
                    config_data = yaml.safe_load(file)
                    path_file = config_data.get('path_to_exteranl_images')
                    save_external_images(images, headers, path_file)