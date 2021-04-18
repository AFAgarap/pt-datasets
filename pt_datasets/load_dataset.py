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
from sklearn.datasets import fetch_20newsgroups, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torchvision

from pt_datasets.AGNews import AGNews
from pt_datasets.COVID19Dataset import BinaryCOVID19Dataset, MultiCOVID19Dataset
from pt_datasets.download_covid_dataset import (
    download_binary_covid19_dataset,
    download_covidx5_dataset,
)
from pt_datasets.TwentyNewsgroups import TwentyNewsgroups
from pt_datasets.utils import preprocess_data, read_data, vectorize_text, unzip_dataset

__author__ = "Abien Fred Agarap"


SUPPORTED_DATASETS = [
    "mnist",
    "fashion_mnist",
    "emnist",
    "cifar10",
    "svhn",
    "malimg",
    "ag_news",
    "20newsgroups",
    "kmnist",
    "wdbc",
    "binary_covid",
    "multi_covid",
]


def load_dataset(
    name: str = "mnist",
    normalize: bool = True,
    augment: bool = False,
    data_folder: str = "~/datasets",
    vectorizer: str = "tfidf",
    ngram_range: Tuple = (1, 5),
    return_vectorizer: bool = False,
    image_size: int = 64,
    preprocessed_covidx: bool = False,
    preprocessing_bsize: int = 2048,
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
            8. 20newsgroups (20 Newsgroups text classification)
            9. kmnist (KMNIST)
            10. wdbc (Wiscosin Diagnostic Breast Cancer classification)
            11. binary_covid (Binary COVID19)
            12. multi_covid (Multi COVID19)
    normalize: bool
        Whether to normalize images or not.
    augment: bool
        Whether to perform image augmentation or not.
        This is only used for *MNIST datasets.
    data_folder: str
        The path to the folder for the datasets.
    vectorizer: str
        The vectorization method to use.
        Options: [tfidf (default) | ngrams]
        This is only used for datasets [name = ag_news | 20newsgroups].
    ngram_range: Tuple
        The lower and upper bound of ngram range to use.
        Default: [(3, 3)]
    return_vectorizer: bool
        Whether to return the vectorizer object or not.
        This is only used for datasets [name = ag_news | 20newsgroups].
    image_size: int
        The image size to use for COVID19 datasets.
    preprocessed_covidx: bool
        Whether to use the preprocessed COVID19 datasets or not.
        This requires the use of `modules/export_covid19_dataset`
        in the package repository.
    preprocessing_bsize: int
        The batch size to use for preprocessing the COVID19 dataset.

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
        "20newsgroups",
        "kmnist",
        "wdbc",
        "binary_covid",
        "multi_covid",
    ]

    name = name.lower()

    _supported = f"Supported datasets: {supported_datasets}"
    assert (
        name in supported_datasets
    ), f"[ERROR] Dataset {name} is not supported. {_supported}"

    train_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    if augment and name in ["mnist", "fashion_mnist", "emnist", "kmnist"]:
        train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    elif (
        not augment
        and normalize
        and name in ["mnist", "fashion_mnist", "emnist", "kmnist"]
    ):
        train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    if name == "mnist":
        train_dataset = torchvision.datasets.MNIST(
            root=data_folder, train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=data_folder, train=False, download=True, transform=test_transform
        )
    elif name == "fashion_mnist":
        train_dataset = torchvision.datasets.FashionMNIST(
            root=data_folder, train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root=data_folder, train=False, download=True, transform=test_transform
        )
    elif name == "emnist":
        train_dataset = torchvision.datasets.EMNIST(
            root=data_folder,
            train=True,
            split="balanced",
            download=True,
            transform=train_transform,
        )
        test_dataset = torchvision.datasets.EMNIST(
            root=data_folder,
            train=False,
            split="balanced",
            download=True,
            transform=test_transform,
        )
    elif name == "cifar10":
        if normalize:
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_folder, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_folder, train=False, download=True, transform=transform
        )
    elif name == "svhn":
        train_dataset = torchvision.datasets.SVHN(
            root=data_folder, split="train", download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.SVHN(
            root=data_folder, split="test", download=True, transform=test_transform
        )
    elif name == "malimg":
        train_dataset, test_dataset = load_malimg()
    elif name == "ag_news":
        if return_vectorizer:
            train_dataset = AGNews(
                train=True, return_vectorizer=return_vectorizer, ngram_range=ngram_range
            )
            test_dataset = AGNews(train=False, ngram_range=ngram_range)
            vectorizer = train_dataset.vectorizer
        else:
            (train_dataset, test_dataset) = (
                AGNews(train=True, ngram_range=ngram_range),
                AGNews(train=False, ngram_range=ngram_range),
            )
    elif name == "20newsgroups":
        if return_vectorizer:
            train_dataset = TwentyNewsgroups(
                return_vectorizer=return_vectorizer, ngram_range=ngram_range
            )
            test_dataset = TwentyNewsgroups(train=False, ngram_range=ngram_range)
            vectorizer = train_dataset.vectorizer
        else:
            (train_dataset, test_dataset) = (
                TwentyNewsgroups(ngram_range=ngram_range),
                TwentyNewsgroups(train=False, ngram_range=ngram_range),
            )
    elif name == "kmnist":
        train_dataset = torchvision.datasets.KMNIST(
            root=data_folder, train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.KMNIST(
            root=data_folder, train=False, download=True, transform=test_transform
        )
    elif name == "wdbc":
        train_dataset, test_dataset = load_wdbc()
    elif name == "binary_covid":
        train_dataset, test_dataset = load_binary_covid19(
            transform=None,
            size=image_size,
            preprocessed=preprocessed_covidx,
            preprocessing_bsize=preprocessing_bsize,
        )
    elif name == "multi_covid":
        train_dataset, test_dataset = load_multi_covid19(
            transform=None,
            size=image_size,
            preprocessed=preprocessed_covidx,
            preprocessing_bsize=preprocessing_bsize,
        )
    return (
        (train_dataset, test_dataset, vectorizer)
        if return_vectorizer
        else (train_dataset, test_dataset)
    )


def load_mnist(
    data_folder: str = "~/datasets", augment: bool = False, normalize: bool = False
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Loads the MNIST training and test datasets.

    Parameters
    ----------
    data_folder: str
        The path to the folder for the datasets.
    augment: bool
        Whether to perform data augmentation or not.
    normalize: bool
        Whether to normalize data or not.

    Returns
    -------
    train_dataset: torch.utils.data.Dataset
        The training set.
    test_dataset: torch.utils.data.Dataset
        The test set.
    """
    train_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    if augment:
        train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    elif normalize:
        train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    train_dataset = torchvision.datasets.MNIST(
        root=data_folder, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_folder, train=False, download=True, transform=test_transform
    )
    return train_dataset, test_dataset


def load_fashion_mnist(
    data_folder: str = "~/dataset", augment: bool = False
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Loads the Fashion-MNIST training and test datasets.

    Parameters
    ----------
    data_folder: str
        The path to the folder for the datasets.
    augment: bool
        Whether to perform data augmentation or not.

    Returns
    -------
    train_dataset: torch.utils.data.Dataset
        The training set.
    test_dataset: torch.utils.data.Dataset
        The test set.
    """
    train_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    if augment:
        train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    train_dataset = torchvision.datasets.FashionMNIST(
        root=data_folder, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root=data_folder, train=False, download=True, transform=test_transform
    )
    return train_dataset, test_dataset


def load_cifar10(data_folder: str = "~/datasets", normalize: bool = False):
    """
    Loads the CIFAR10 training and test datasets.

    Parameter
    ---------
    data_folder: str
        The path to the folder for the datasets.
    normalize: bool
        Whether to normalize the dataset or not.

    Returns
    -------
    train_dataset: torch.utils.data.Dataset
        The training set.
    test_dataset: torch.utils.data.Dataset
        The test set.
    """
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    if normalize:
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_folder, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_folder, train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


def load_svhn(
    data_folder: str = "~/datasets",
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Loads the SVHN training and test datasets.

    Parameter
    ---------
    data_folder: str
        The path to the folder for the datasets.

    Returns
    -------
    Tuple
        train_dataset: torch.utils.data.Dataset
            The training set.
        test_dataset: torch.utils.data.Dataset
            The test set.
    """
    train_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_dataset = torchvision.datasets.SVHN(
        root=data_folder, split="train", download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.SVHN(
        root=data_folder, split="test", download=True, transform=test_transform
    )
    return train_dataset, test_dataset


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
    download_url = "https://drive.google.com/uc?id=1Y6Ha5Jir8EI726KdwAKWHVU-oWmPqNDT"
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


def load_wdbc(test_size: float = 3e-1, seed: int = 42):
    """
    Loads the Wisconsin Diagnostic Breast Cancer dataset.

    Parameters
    ----------
    test_size: float
        The size of the test set.
    seed: int
        The random seed to use for reproducibility.
    """
    features, labels = load_breast_cancer(return_X_y=True)
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=test_size, random_state=seed, shuffle=True
    )
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.fit_transform(test_features)
    train_features = train_features.astype("float32")
    test_features = test_features.astype("float32")
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_features), torch.from_numpy(train_labels)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(test_features), torch.from_numpy(test_labels)
    )
    return train_dataset, test_dataset


def load_binary_covid19(
    transform: torchvision.transforms,
    size: int = 64,
    preprocessed: bool = False,
    preprocessing_bsize: int = 2048,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Returns a tuple of the tensor datasets for the
    train and test sets of the COVID19 binary classification dataset.

    Parameters
    ----------
    transform: torchvision.transform
        The transformation pipeline to use for image preprocessing.
    size: int
        The size to use for image resizing.
    preprocessed: bool
        Whether to load preprocessed dataset or not.
    preprocessing_bsize: int
        The batch size to use for preprocessing the dataset.

    Returns
    -------
    train_data: torch.utils.data.TensorDataset
        The training set of Binary COVID19 dataset.
    test_data: torch.utils.data.TensorDataset
        The test set of Binary COVID19 dataset.
    """
    dataset_path = os.path.join(str(Path.home()), "datasets")
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    if not os.path.exists(os.path.join(dataset_path, "BinaryCOVID19Dataset")):
        download_binary_covid19_dataset()
        unzip_dataset(os.path.join(dataset_path, "BinaryCOVID19Dataset.tar.xz"))
    (train_data, test_data) = (
        BinaryCOVID19Dataset(
            train=True,
            preprocessed=preprocessed,
            size=size,
            preprocessing_bsize=preprocessing_bsize,
        ),
        BinaryCOVID19Dataset(
            train=False,
            preprocessed=preprocessed,
            size=size,
            preprocessing_bsize=preprocessing_bsize,
        ),
    )
    return train_data, test_data


def load_multi_covid19(
    transform: torchvision.transforms,
    size: int = 64,
    preprocessed: bool = False,
    preprocessing_bsize: int = 2048,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Returns a tuple of the tensor datasets for the
    train and test sets of the COVID19 multi-classification dataset.

    Parameters
    ----------
    transform: torchvision.transform
        The transformation pipeline to use for image preprocessing.
    size: int
        The size to use for image resizing.
    preprocessed: bool
        Whether to load preprocessed dataset or not.
    preprocessing_bsize: int
        The batch size to use for preprocessing the dataset.

    Returns
    -------
    train_data: torch.utils.data.TensorDataset
        The training set of COVIDx5 dataset.
    test_data: torch.utils.data.TensorDataset
        The test set of COVIDx5 dataset.
    """
    dataset_path = os.path.join(str(Path.home()), "datasets")
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    if not os.path.exists(os.path.join(dataset_path, "MultiCOVID19Dataset")):
        download_covidx5_dataset()
        unzip_dataset(os.path.join(dataset_path, "MultiCOVID19Dataset.tar.xz"))
    (train_data, test_data) = (
        MultiCOVID19Dataset(
            train=True,
            preprocessed=preprocessed,
            size=size,
            preprocessing_bsize=preprocessing_bsize,
        ),
        MultiCOVID19Dataset(
            train=False,
            preprocessed=preprocessed,
            size=size,
            preprocessing_bsize=preprocessing_bsize,
        ),
    )
    return train_data, test_data
