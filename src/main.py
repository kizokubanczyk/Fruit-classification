import click
import yaml
from dataset_and_dataloader.dataSet import createDataSet
from dataset_and_dataloader import importData
from dataset_and_dataloader.splitData import  split
from dataset_and_dataloader.dataLoader import createDataLoader


@click.command()
@click.option('--config', '-c', default='../config.yaml', help='Path to the configuration file')
def run(config) -> None:
    with (open(config, 'r') as file):
        config_data = yaml.safe_load(file)

        image_directory_paths, labels = importData.importData(config_data.get('path_to_all_data'))

        X_train, X_val, X_test, y_train, y_val, y_test = split(image_directory_paths, labels)

        train_dataset, val_dataset, test_dataset  = createDataSet(X_train, X_val, X_test, y_train, y_val, y_test)

        train_dataLoader = createDataLoader(train_dataset, 32, True)
        val_dataLoader = createDataLoader(val_dataset, 32, True)
        test_dataLoader = createDataLoader(test_dataset, 32, True)



if __name__ == "__main__":
    run()