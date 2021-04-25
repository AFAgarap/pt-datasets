import numpy as np
from pt_datasets import create_dataloader
from pt_datasets.load_dataset import load_mnist, load_fashion_mnist


def test_load_mnist():
    train_data, test_data = load_mnist()
    assert (len(train_data), len(test_data)) == (60_000, 10_000)
    assert train_data.data.numpy().min() == 0
    assert train_data.data.numpy().max() == 255
    assert test_data.data.numpy().min() == 0
    assert test_data.data.numpy().max() == 255
    train_loader = create_dataloader(train_data, batch_size=32)
    for index, (batch_features, batch_labels) in enumerate(train_loader):
        if index == 1:
            break
    assert batch_features.shape == (32, 1, 28, 28)
    assert batch_labels.shape == (32,)
    assert batch_features.numpy().min() == 0.0
    assert batch_features.numpy().max() == 1.0
    train_data, test_data = load_mnist(normalize=True)
    train_loader = create_dataloader(train_data, batch_size=32)
    for index, (batch_features, batch_labels) in enumerate(train_loader):
        if index == 1:
            break
    assert np.isclose(batch_features.numpy().min(), -0.42421296)
    assert np.isclose(batch_features.numpy().max(), 2.8214867)
