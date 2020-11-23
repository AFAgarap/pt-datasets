# PyTorch Datasets utility repository
# Copyright (C) 2020  Abien Fred Agarap
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Function for loading datasets"""
import os
from pathlib import Path
from typing import Tuple

import gdown
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torchvision

from pt_datasets.utils import preprocess_data, read_data, vectorize_text

__author__ = "Abien Fred Agarap"


def load_dataset(
    name: str = "mnist",
    data_folder: str = "~/torch_datasets",
    vectorizer: str = "tfidf",
) -> Tuple[object, object]:
    """
    Returns a tuple of torchvision dataset objects.

    Parameters
    ----------
    name: str
        The name of the dataset to load. Current choices:
            1. mnist (MNIST)
            2. fashion_mnist (FashionMNIST)
            3. emnist (EMNIST/Balanced)
            4. cifar10 (CIFAR10)
            5. svhn (SVHN)
            6. malimg (Malware Image classification)
            7. ag_news (AG News)
    data_folder: str
        The path to the folder for the datasets.

    Returns
    -------
    Tuple[object, object]
        A tuple consisting of the training dataset and the test dataset.
    """
    supported_datasets = [
        "mnist",
        "fashion_mnist",
        "emnist",
        "cifar10",
        "svhn",
        "malimg",
        "ag_news",
    ]

    name = name.lower()

    _supported = "Supported datasets: mnist, fashion_mnist, emnist, cifar10, svhn, malimg, ag_news."
    assert (
        name in supported_datasets
    ), f"[ERROR] Dataset {name} is not supported. {_supported}"

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    if name == "mnist":
        train_dataset = torchvision.datasets.MNIST(
            root=data_folder, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=data_folder, train=False, download=True, transform=transform
        )
    elif name == "fashion_mnist":
        train_dataset = torchvision.datasets.FashionMNIST(
            root=data_folder, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root=data_folder, train=False, download=True, transform=transform
        )
    elif name == "emnist":
        train_dataset = torchvision.datasets.EMNIST(
            root=data_folder,
            train=True,
            split="balanced",
            download=True,
            transform=transform,
        )
        test_dataset = torchvision.datasets.EMNIST(
            root=data_folder,
            train=False,
            split="balanced",
            download=True,
            transform=transform,
        )
    elif name == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_folder, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_folder, train=False, download=True, transform=transform
        )
    elif name == "svhn":
        train_dataset = torchvision.datasets.SVHN(
            root=data_folder, split="train", download=True, transform=transform
        )
        test_dataset = torchvision.datasets.SVHN(
            root=data_folder, split="test", download=True, transform=transform
        )
    elif name == "malimg":
        train_dataset, test_dataset = load_malimg()
    elif name == "ag_news":
        train_dataset, test_dataset = load_agnews(vectorizer)
    return (train_dataset, test_dataset)


def load_malimg(
    test_size: float = 0.3, seed: int = 42
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Returns a tuple of tensor datasets for the
    training and test splits of MalImg dataset.

    Parameters
    ----------
    test_size: float
        The size of the test set.
    seed: int
        The random seed to use for splitting.

    Returns
    -------
    Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        train_dataset
            The training set of MalImg dataset.
        test_dataset
            The test set of MalImg dataset.
    """
    download_url = ("https://drive.google.com/uc?id=1Y6Ha5Jir8EI726KdwAKWHVU-oWmPqNDT",)
    malimg_filename = "malimg_dataset_32x32.npy"
    dataset_path = os.path.join(str(Path.home()), "datasets")
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    if not os.path.isfile(os.path.join(dataset_path, malimg_filename)):
        gdown.download(
            download_url, os.path.join(dataset_path, malimg_filename), quiet=True
        )
    dataset = np.load(os.path.join(dataset_path, malimg_filename), allow_pickle=True)
    train_data, test_data = train_test_split(
        dataset, test_size=test_size, random_state=seed
    )
    train_features, train_labels = train_data[:, : (32 ** 2)], train_data[:, -1]
    test_features, test_labels = test_data[:, : (32 ** 2)], test_data[:, -1]
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_features), torch.from_numpy(train_labels)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(test_features), torch.from_numpy(test_labels)
    )
    return train_dataset, test_dataset


def load_agnews(
    vectorization_mode: str = "tfidf", seed: int = 42
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    path = str(Path.home())
    path = os.path.join(path, "torch_datasets")
    train_path = os.path.join(path, "ag_news.train")
    test_path = os.path.join(path, "ag_news.test")
    train_dataset, test_dataset = (read_data(train_path), read_data(test_path))
    train_texts, train_labels = (
        list(train_dataset.keys()),
        list(train_dataset.values()),
    )
    test_texts, test_labels = (list(test_dataset.keys()), list(test_dataset.values()))
    train_texts, train_labels = preprocess_data(train_texts, train_labels)
    test_texts, test_labels = preprocess_data(test_texts, test_labels)
    train_vectors = vectorize_text(train_texts, vectorization_mode)
    test_vectors = vectorize_text(test_texts, vectorization_mode)
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_vectors), torch.from_numpy(train_labels)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(test_vectors), torch.from_numpy(test_labels)
    )
    return train_dataset, test_dataset
