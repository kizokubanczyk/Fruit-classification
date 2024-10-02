import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def save_score(score: float, description: str, file_path: str) -> None:
    with open(file_path, "a") as file:
        file.write(description + ": " + str(score) + "\n")

def initialize_file(file_path: str) -> None:
    open(file_path, "w").close()

def save_external_images(images, headers, folder_path: str) -> None:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i, image in enumerate(images):
        image_np = image.cpu().numpy().transpose((1, 2, 0))
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        image_np = (image_np * 255).astype(np.uint8)

        plt.figure()
        plt.imshow(image_np)
        plt.text(100, -10, headers[i], color='white', fontsize=15, backgroundcolor='black')

        file_path = os.path.join(folder_path, f'image_{i + 1}.jpeg')
        plt.axis('off')
        plt.savefig(file_path, format='jpeg', bbox_inches='tight', pad_inches=0)
        plt.close()

def save_best_model(model: torch.nn.Module, file_path:str) -> None:
    torch.save(model.state_dict(), file_path)

def save_Confusion_Matrix(Confusion_Matrix:confusion_matrix,  folder_path: str) -> None:
    with open(folder_path, 'a') as f:
        f.write("\nConfusion Matrix:\n")
        for row in Confusion_Matrix:
            f.write(' '.join(map(str, row)) + '\n')

