import torch
import torch.nn as nn
import torch.optim as optim
from src.model.covalent_network import CNN


num_classes = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(num_classes=num_classes).to(device)  # Inicjalizacja modelu

def train_model(train_loader,val_dataLoader):


    criterion = nn.CrossEntropyLoss() # jest używane podczas trenowania modelu do obliczania straty, co pozwala optymalizatorowi na aktualizację wag modelu.

    optimizer = optim.Adam(model.parameters(), lr=0.001)  #Inicjalizuje optymalizator Adam, który będzie używany do aktualizacji wag modelu. Optymalizator Adam
    # jest popularnym algorytmem optymalizacji w uczeniu głębokim.
    #model.parameters(): Pobiera parametry modelu (wagi i biasy), które będą aktualizowane podczas treningu.
    #lr=0.001 -  Wartośćtego parametru określa, jak dużą   korektę  wagi modelu należy zastosowaćpo każdej iteracji (czyli pokażdej aktualizacji gradientu).
    # W skrócie, learning rate kontroluje, jak szybki jest proces  uczenia się modelu.

    num_epochs = 10 #Liczba epok, czyli ile razy model będzie przechodził przez cały zbiór danych treningowych.

    for epoch in range(num_epochs):
        model.train()  # Przełącz model w tryb treningowy
        running_loss = 0.0

        for images, labels in train_loader:  # Przechodzenie przez batch danych
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  #Gradienty są miarą tego, jak funkcja straty zmienia się w odpowiedzi na zmiany wag modelu.
            # W procesie optymalizacji, gradienty są używane do aktualizacji wag modelu w kierunku, który minimalizuje funkcję straty.
            # Wyzerowanie gradientów przed każdym krokiem optymalizacji jest niezbędne, aby gradienty obliczone dla bieżącego batcha danych były dokładne i
            # nie były wpływane przez wcześniejsze obliczenia. Bez tego kroku gradienty mogą się kumulować, co prowadzi do nieprawidłowych aktualizacji wag i
            # problemów z trenowaniem modelu.

            outputs = model(images)  # Przekazuje obrazy przez model, aby uzyskać przewidywania. (Forward pass)
            #Model przechodzi przez warstwy i oblicza prognozy na podstawie bieżących wag.

            loss = criterion(outputs, labels)  # Obliczenie straty
            loss.backward()  # Oblicza gradienty strat względem wag modelu (backward pass).
            optimizer.step()  # Aktualizuje wagi modelu na podstawie obliczonych gradientów.

            running_loss += loss.item()  # Zbieranie statystyk


        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # Nie obliczaj gradientów podczas walidacji
            for images, labels in val_dataLoader:

                images, labels = images.to(device), labels.to(device) # images i labels są przenoszone na urządzenie (CPU lub GPU), na którym działa model (device).


                outputs = model(images) #Forward pass, Model generuje przewidywania (outputs) na podstawie przekazanych obrazów.

                loss = criterion(outputs, labels) #Gradienty są śledzone: Podobnie jak podczas forward pass, operacje związane z obliczaniem straty (criterion)
                # również są śledzone dla gradientów.

                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1) #zwraca indeksy klas o największym prawdopodobieństwie dla każdego obrazu. predicted to tensor z
                # przewidywanymi klasami.

                total += labels.size(0)

                correct += (predicted == labels).sum().item() #total to łączna liczba próbek przetworzonych w batchach.correct
                # to liczba poprawnych przewidywań, gdzie predicted == labels porównuje przewidywania z prawdziwymi
                # etykietami, a.sum().item() zlicza liczbę poprawnych przewidywań.

        avg_loss = running_loss / len(val_dataLoader)
        accuracy = 100 * correct / total # Po przetworzeniu wszystkich batchy, avg_loss oblicza średnią  stratę na  podstawie
        #całkowitej straty(total_loss)  i liczby próbek(total).accuracy  oblicza dokładność modelu jako stosunek poprawnych
        #przewidywań(correct) do całkowitej liczby  próbek(total).
        print(f'Validation Loss: {avg_loss}')
        print(f'Validation Accuracy: {accuracy}%')

        #f1 score .. accuracy f1 score itp

def test_model(test_dataLoader):

    correct = 0
    total = 0

    with torch.no_grad():  # Nie obliczaj gradientów podczas testowania
        for images, labels in test_dataLoader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total}%')