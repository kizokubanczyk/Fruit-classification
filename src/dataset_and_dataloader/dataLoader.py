from torch.utils.data import DataLoader, Dataset


def createDataLoader(dataSet: Dataset, batch_size: int, shuffle: bool) -> DataLoader:

    dataloader = DataLoader(
        dataset = dataSet,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return dataloader
