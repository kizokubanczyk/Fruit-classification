import torch
import yaml
from src.file_operations.save_results import save_best_model

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.best_val_Accuracy = None
        self.best_model = None

    def __call__(self, accuracy:float, model: torch.nn.Module)->None:

        score = accuracy

        if self.best_val_Accuracy is None and self.best_model is None:
            self.best_val_Accuracy = score
            self.best_model = model
        elif score > self.best_val_Accuracy + self.min_delta:
            self.best_val_Accuracy = score
            self.counter = 0
            self.best_model = model
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                with open("../config.yaml", 'r') as file:
                    config_data = yaml.safe_load(file)
                    path_file = config_data.get('path_to_best_model')
                    save_best_model(model=self.best_model, file_path=path_file)
