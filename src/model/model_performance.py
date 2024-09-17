import torch
import torch.nn as nn
import torch.optim as optim
from covalent_network import CNN


num_classes = 10  # Przykład; zmień zgodnie z liczbą klas w Twoim problemie
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(num_classes=num_classes).to(device)  # Inicjalizacja modelu

def train_model(train_loader):

    # Ustalamy funkcję straty i optymalizator
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Możesz dostosować learning rate

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()  # Przełącz model w tryb treningowy
        running_loss = 0.0

        for images, labels in train_loader:  # Przechodzenie przez batch danych
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Wyzerowanie gradientów

            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Obliczenie straty
            loss.backward()  # Backward pass
            optimizer.step()  # Optymalizacja wag

            running_loss += loss.item()  # Zbieranie statystyk

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')