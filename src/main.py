import click
import yaml
from dataset_and_dataloader.dataSet import createDataSet
from dataset_and_dataloader import importData
from dataset_and_dataloader.splitData import  split

from dataset_and_dataloader.dataLoader import createDataLoader
from src.model.model_performance import train_model


@click.command()
@click.option('--config', '-c', default='../config.yaml', help='Path to the configuration file')
def run(config) -> None:
    with (open(config, 'r') as file):
        config_data = yaml.safe_load(file)

        image_directory_paths, labels = importData.importData(config_data.get('path_to_all_data'))

        X_train, X_val, X_test, y_train, y_val, y_test = split(image_directory_paths, labels)

        train_dataset, val_dataset, test_dataset  = createDataSet(X_train, X_val, X_test, y_train, y_val, y_test)

        #print(train_dataset.__getitem__(0)[0][0])

        train_dataLoader = createDataLoader(train_dataset, 64, True)
        val_dataLoader = createDataLoader(val_dataset, 64, True)
        test_dataLoader = createDataLoader(test_dataset, 64, True)

        train_model(train_dataLoader,val_dataLoader)


if __name__ == "__main__":
    run()