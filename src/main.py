import click
import yaml
from dataset_and_dataloader.dataSet import ClassificationDataset

@click.command()
@click.option('--config', '-c', default='../config.yaml', help='Path to the configuration file')
def run(config) -> None:
    with (open(config, 'r') as file):
        config_data = yaml.safe_load(file)

        path_to_all_data = config_data.get('path_to_all_data')
        classificationDataset = ClassificationDataset(path_to_all_data)
        print(classificationDataset.__getitem__(400))



if __name__ == "__main__":
    run()