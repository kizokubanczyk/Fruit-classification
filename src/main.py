import click
import yaml
from dataset_and_dataloader.dataSet import createDataSet
from dataset_and_dataloader import importData
from dataset_and_dataloader.splitData import  split

from dataset_and_dataloader.dataLoader import createDataLoader
from src.model.model_performance import train_model, test_model, classify_external_image
from src.save_results import initialize_file


@click.command()
@click.option('--config', '-c', default='../config.yaml', help='Path to the configuration file')
def run(config) -> None:
    with (open(config, 'r') as file):
        config_data = yaml.safe_load(file)

        initialize_file(config_data.get('path_to_scores'))

        image_directory_paths, labels = importData.importData(config_data.get('path_to_train_and_test_data'))

        X_train, X_val, X_test, y_train, y_val, y_test = split(image_directory_paths, labels)

        train_dataset = createDataSet(X_train, y_train, 4)
        val_dataset = createDataSet(X_val, y_val, 0 )
        test_dataset = createDataSet(X_test, y_test, 0)

        print(train_dataset.__len__())

        #print(train_dataset.__getitem__(0)[0][0])

        train_dataLoader = createDataLoader(train_dataset, 8, True)

        val_dataLoader = createDataLoader(val_dataset, 8, False)
        test_dataLoader = createDataLoader(test_dataset, 8, False)

        train_model(train_dataLoader,val_dataLoader)
        test_model(test_dataLoader)

        external_image_directory_paths, external_labels = importData.importExternalData(config_data.get('path_to_external_data'))

        external_dataset = createDataSet(external_image_directory_paths, external_labels, 0)
        external_dataLoader = createDataLoader(external_dataset, 3, False)
        classify_external_image(external_dataLoader)


if __name__ == "__main__":
    run()

