import torch


def create_dataset(
    features: torch.Tensor, labels: torch.Tensor
) -> torch.utils.data.TensorDataset:
    """
    Returns a torch.utils.data.TensorDataset object
    to be used for a data loader object.

    Parameters
    ----------
    features: torch.Tensor
        The input features.
    labels: torch.Tensor
        The input labels.

    Returns
    -------
    dataset: torch.utils.data.TensorDataset
        The Tensor Dataset object.
    """
    dataset = torch.utils.data.TensorDataset(features, labels)
    return dataset
