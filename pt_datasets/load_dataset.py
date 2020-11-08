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

__author__ = "Abien Fred Agarap"


def load_dataset(
    name: str = "mnist", data_folder: str = "~/torch_datasets"
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
    data_folder: str
        The path to the folder for the datasets.

    Returns
    -------
    Tuple[object, object]
        A tuple consisting of the training dataset and the test dataset.
    """
    supported_datasets = ["mnist", "fashion_mnist", "emnist", "cifar10", "svhn"]

    name = name.lower()

    assert (
        name in supported_datasets
    ), f"[ERROR] Dataset {name} is not supported. Supported datasets: mnist, fashion_mnist, emnist, cifar10, svhn."

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
    return (train_dataset, test_dataset)


def load_malimg(
    test_size: float = 0.3, seed: int = 42
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
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
    return train_features, train_labels
