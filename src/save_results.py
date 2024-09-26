

def save_accuracy(score: float, description: str, file_path: str) -> None:
    with open(file_path, "a") as file:
        print('save')
        file.write(description + ": " + str(score) + "\n")

def initialize_file(file_path: str) -> None:
    with open(file_path, "w") as file:
        pass  # Otwieramy plik w trybie 'w', aby go wyczyścić



