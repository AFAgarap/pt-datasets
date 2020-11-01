import torch


def create_dataloader(
    dataset: object, batch_size: int = 16, shuffle: bool = True, num_workers: int = 0
) -> object:
    """
    Returns a data loader object, ready to be used by a model.

    Parameters
    ----------
    dataset: object
        The dataset from `torchvision.datasets`.
    batch_size: int
        The mini-batch size for the data loading. Default is [16].
    shuffle: bool
        Whether to shuffle dataset or not. Default is [True].
    num_workers: int
        The number of subprocesses to use for data loading. Default is [0].

    Returns
    -------
    data_loader: torch.utils.dataloader.DataLoader
        The data loader object to be used by a model.
    """
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader
