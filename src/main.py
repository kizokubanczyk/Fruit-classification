import click
import yaml
from src.importData_and_dataLoader.importData import importData


@click.command()
@click.option('--config', '-c', default='../config.yaml', help='Path to the configuration file')
def run(config) -> None:
    with (open(config, 'r') as file):
        config_data = yaml.safe_load(file)
        config_path = '../config.yaml'

        path_train_data = config_data.get('path_train_data')
        path_test_data = config_data.get('path_test_data')
        path_predict_data = config_data.get('path_predict_data')


        importData(config_path, path_train_data)




if __name__ == "__main__":
    run()