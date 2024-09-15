import click
import yaml
from dataset_and_dataloader.dataSet import ClassificationDataset
from dataset_and_dataloader import importData
from dataset_and_dataloader.splitData import  split



@click.command()
@click.option('--config', '-c', default='../config.yaml', help='Path to the configuration file')
def run(config) -> None:
    with (open(config, 'r') as file):
        config_data = yaml.safe_load(file)

        images, labels = importData.importData(config_data.get('path_to_all_data'))



if __name__ == "__main__":
    run()