import torch


def create_dataset(
    features: torch.Tensor, labels: torch.Tensor
) -> torch.utils.data.TensorDataset:
    dataset = torch.utils.data.TensorDataset(features, labels)
    return dataset
