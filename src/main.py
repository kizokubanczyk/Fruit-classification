import click
import yaml
from dataset_and_dataloader.dataset import ClassificationDataset

@click.command()
@click.option('--config', '-c', default='../config.yaml', help='Path to the configuration file')
def run(config) -> None:
    with (open(config, 'r') as file):
        config_data = yaml.safe_load(file)

        path_data_train = config_data.get('path_data_train_for_main')
        classificationDataset = ClassificationDataset(path_data_train)


if __name__ == "__main__":
    run()