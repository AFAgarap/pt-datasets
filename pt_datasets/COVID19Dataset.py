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
"""COVID19 dataset classes"""
import os
from pathlib import Path
import pickle
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision

from . import create_dataloader
from pt_datasets.utils import read_metadata, load_image

__author__ = "Abien Fred Agarap"


DATASET_DIR = os.path.join(str(Path.home()), "datasets")
BINARY_COVID19_DIR = os.path.join(DATASET_DIR, "BinaryCOVID19Dataset")
MULTI_COVID19_DIR = os.path.join(DATASET_DIR, "MultiCOVID19Dataset")
TRAIN_METADATA = "train_split.txt"
TEST_METADATA = "test_split.txt"


class BinaryCOVID19Dataset(torch.utils.data.Dataset):
    """
    Dataset class for the COVID19 binary classification dataset.
    """

    def __init__(
        self,
        train: bool = True,
        transform: torchvision.transforms = None,
        size: int = 64,
        preprocessed: bool = False,
        preprocessing_bsize: int = 2048,
    ):
        """
        Builds the COVID19 binary classification dataset.

        Parameter
        ---------
        train: bool
            Whether to load the training set or not.
        transform: torchvision
            The transformation pipeline to use for image preprocessing.
        size: int
            The size to use for resizing images.
        preprocessed: bool
            Whether to load preprocessed dataset or not.
        preprocessing_bsize: int
            The mini-batch size to use preprocessing the dataset.
        """
        if preprocessed:
            if not os.path.isfile(os.path.join(BINARY_COVID19_DIR, f"train_{size}.pt")):
                print(
                    "[INFO] No preprocessed training dataset found. Preprocessing now..."
                )
                preprocess_dataset(
                    train=True,
                    size=size,
                    export_dir=BINARY_COVID19_DIR,
                    batch_size=preprocessing_bsize,
                )
            if not os.path.isfile(os.path.join(BINARY_COVID19_DIR, f"test_{size}.pt")):
                print("[INFO] No preprocessed test dataset found. Preprocessing now...")
                preprocess_dataset(
                    train=False,
                    size=size,
                    export_dir=BINARY_COVID19_DIR,
                    batch_size=preprocessing_bsize,
                )
            if train:
                if size > 64:
                    dataset = load_pickle(
                        os.path.join(BINARY_COVID19_DIR, f"train_{size}.pt")
                    )
                else:
                    dataset = torch.load(
                        os.path.join(BINARY_COVID19_DIR, f"train_{size}.pt")
                    )
            else:
                if size > 64:
                    dataset = load_pickle(
                        os.path.join(BINARY_COVID19_DIR, f"test_{size}.pt")
                    )
                else:
                    dataset = torch.load(
                        os.path.join(BINARY_COVID19_DIR, f"test_{size}.pt")
                    )
            self.data = dataset[0]
            self.labels = dataset[1]
        else:
            if train:
                path = os.path.join(BINARY_COVID19_DIR, "data/train")
                self.annotations = read_metadata(
                    os.path.join(BINARY_COVID19_DIR, TRAIN_METADATA)
                )
                self.root_dir = path
            else:
                path = os.path.join(BINARY_COVID19_DIR, "data/test")
                self.annotations = read_metadata(
                    os.path.join(BINARY_COVID19_DIR, TEST_METADATA)
                )
                self.root_dir = path
        self.classes = ["negative", "positive"]
        self.transform = transform
        self.size = size
        self.preprocessed = preprocessed

    def __len__(self):
        return len(self.data) if self.preprocessed else len(self.annotations)

    def __getitem__(self, idx) -> Tuple or Dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.preprocessed:
            image = self.data[idx]
            if self.transform:
                image = self.transform(image)
            label = self.labels[idx].astype("int64")
            return (image, label)
        else:
            image_name = os.path.join(self.root_dir, self.annotations[idx][1])
            image = load_image(image_name, self.size)
            if self.transform:
                image = self.transform(image)
            label = self.annotations[idx][2]
            label = 0 if label == "negative" else 1
            sample = {"image": image, "label": label}
            return sample


class MultiCOVID19Dataset(torch.utils.data.Dataset):
    """
    Dataset class for the COVID19 multi-classification dataset.
    """

    def __init__(
        self,
        train: bool = True,
        transform: torchvision.transforms = None,
        size: int = 64,
        preprocessed: bool = False,
        preprocessing_bsize: int = 2048,
    ):
        """
        Builds the COVID19 multi-classification dataset.

        Parameter
        ---------
        train: bool
            Whether to load the training set or not.
        transform: torchvision
            The transformation pipeline to use for image preprocessing.
        size: int
            The size to use for resizing images.
        preprocessed: bool
            Whether to load preprocessed dataset or not.
        preprocessing_bsize: int
            The mini-batch size to use preprocessing the dataset.
        """
        if preprocessed:
            if not os.path.isfile(os.path.join(MULTI_COVID19_DIR, f"train_{size}.pt")):
                print(
                    "[INFO] No preprocessed training dataset found. Preprocessing now..."
                )
                preprocess_dataset(
                    train=True,
                    size=size,
                    export_dir=MULTI_COVID19_DIR,
                    batch_size=preprocessing_bsize,
                )
            if not os.path.isfile(os.path.join(MULTI_COVID19_DIR, f"test_{size}.pt")):
                print("[INFO] No preprocessed test dataset found. Preprocessing now...")
                preprocess_dataset(
                    train=False,
                    size=size,
                    export_dir=MULTI_COVID19_DIR,
                    batch_size=preprocessing_bsize,
                )
            if train:
                if size > 64:
                    dataset = load_pickle(
                        os.path.join(MULTI_COVID19_DIR, f"train_{size}.pt")
                    )
                else:
                    dataset = torch.load(
                        os.path.join(MULTI_COVID19_DIR, f"train_{size}.pt")
                    )
            else:
                if size > 64:
                    dataset = load_pickle(
                        os.path.join(MULTI_COVID19_DIR, f"test_{size}.pt")
                    )
                else:
                    dataset = torch.load(
                        os.path.join(MULTI_COVID19_DIR, f"test_{size}.pt")
                    )
            self.data = dataset[0]
            self.labels = dataset[1]
        if train:
            path = os.path.join(MULTI_COVID19_DIR, "data/train")
            self.annotations = read_metadata(
                os.path.join(MULTI_COVID19_DIR, TRAIN_METADATA)
            )
            self.root_dir = path
        else:
            path = os.path.join(MULTI_COVID19_DIR, "data/test")
            self.annotations = read_metadata(
                os.path.join(MULTI_COVID19_DIR, TEST_METADATA)
            )
            self.root_dir = path
        self.classes = ["normal", "non-COVID-19 pneumonia", "COVID-19 pneumonia"]
        self.transform = transform
        self.size = size
        self.preprocessed = preprocessed

    def __len__(self):
        return len(self.data) if self.preprocessed else len(self.annotations)

    def __getitem__(self, idx) -> Dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.preprocessed:
            image = self.data[idx]
            if self.transform:
                image = self.transform(image)
            label = self.labels[idx].astype("int64")
            return (image, label)
        else:
            image_name = os.path.join(self.root_dir, self.annotations[idx][1])
            image = load_image(image_name, self.size)
            if self.transform:
                image = self.transform(image)
            label = self.annotations[idx][2]
            if label == "normal":
                label = 0
            elif label == "pneumonia":
                label = 1
            elif label == "COVID-19":
                label = 2
            sample = {"image": image, "label": label}
            return sample


def unpack_examples(data_loader: torch.utils.data.DataLoader) -> Tuple[List, List]:
    """
    Unpacks examples from a data loader,
    and returns them as tuples of list pairs.

    Parameter
    ---------
    data_loader: torch.utils.data.DataLoader
        The data loader object that contains the features-labels pairs.

    Returns
    -------
    Tuple[List, List]
        features: List
            The dataset features.
        labels: List
            The dataset labels.
    """
    features, labels = [], []
    for index, example in enumerate(data_loader):
        start_time = time.time()
        features.append(example.get("image"))
        labels.append(example.get("label"))
        duration = time.time() - start_time
        print(f"[INFO] Processing batch {index + 1} took {duration:.6f}s.")
    return features, labels


def vectorize_examples(
    features: List,
    labels: List,
    dataset_size: int,
    batch_size: int = 512,
    image_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns a the vectorized features-labels pairs.

    Parameters
    ----------
    features: List
        The dataset features.
    labels: List
        The dataset labels.
    dataset_size: int
        The size of the dataset.
    batch_size: int
        The mini-batch used for unpacking the examples.
    image_size: int
        The number of channels in the dataset features.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        array: np.ndarray
            The vectorized features.
        labels_array: np.ndarray
            The vectorized labels.
    """
    num_channels = 3
    array = np.zeros((dataset_size, num_channels, image_size, image_size))
    labels_array = np.zeros((dataset_size))
    for index, (row, label) in enumerate(zip(features, labels)):
        offset = index * batch_size
        array[offset : offset + batch_size] = row
        labels_array[offset : offset + batch_size] = label
    labels_array = labels_array.astype("int64")
    array = array.astype("float32")
    return array, labels_array


def export_dataset(dataset: Tuple[np.ndarray, np.ndarray], filename: str) -> None:
    """
    Exports the vectorized dataset to a `.pt` file.

    Parameters
    ----------
    dataset: Tuple[np.ndarray, np.ndarray]
        The tuple of vectorized features and labels.
    filename: str
        The path where to save the dataset.
    """
    if not filename.endswith(".pt"):
        filename = f"{filename}.pt"
    if dataset[0].shape[2] > 64:
        with open(filename, "wb") as tensor_file:
            pickle.dump(dataset, tensor_file)
    else:
        torch.save(dataset, filename)


def preprocess_dataset(
    export_dir: str, train: bool = False, size: int = 64, batch_size: int = 2048
) -> None:
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    if train:
        print("[INFO] Loading dataset...")
        if "BinaryCOVID19Dataset" in export_dir:
            train_data = BinaryCOVID19Dataset(
                train=True, size=size, transform=transform
            )
        elif "MultiCOVID19Dataset" in export_dir:
            train_data = MultiCOVID19Dataset(train=True, size=size, transform=transform)
        print("[INFO] Creating data loader...")
        train_loader = create_dataloader(
            train_data, batch_size=batch_size, num_workers=4
        )
        print("[INFO] Unpacking examples...")
        train_features, train_labels = unpack_examples(train_loader)
        print("[INFO] Vectorizing examples...")
        train_features, train_labels = vectorize_examples(
            train_features,
            train_labels,
            dataset_size=len(train_data),
            batch_size=batch_size,
            image_size=size,
        )
        train_dataset = (train_features, train_labels)
        print(
            "[INFO] Exporting dataset to {}".format(
                os.path.join(export_dir, f"train_{size}.pt")
            )
        )
        export_dataset(train_dataset, os.path.join(export_dir, f"train_{size}.pt"))
    else:
        print("[INFO] Loading dataset...")
        if "BinaryCOVID19Dataset" in export_dir:
            test_data = BinaryCOVID19Dataset(
                train=False, size=size, transform=transform
            )
        elif "MultiCOVID19Dataset" in export_dir:
            test_data = MultiCOVID19Dataset(train=False, size=size, transform=transform)
        print("[INFO] Creating data loader...")
        test_loader = create_dataloader(test_data, batch_size=batch_size, num_workers=4)
        print("[INFO] Unpacking examples...")
        test_features, test_labels = unpack_examples(test_loader)
        print("[INFO] Vectorizing examples...")
        test_features, test_labels = vectorize_examples(
            test_features,
            test_labels,
            dataset_size=len(test_data),
            batch_size=batch_size,
            image_size=size,
        )
        test_dataset = (test_features, test_labels)
        print(
            "[INFO] Exporting dataset to {}".format(
                os.path.join(export_dir, f"test_{size}.pt")
            )
        )
        export_dataset(test_dataset, os.path.join(export_dir, f"test_{size}.pt"))


def load_pickle(
    filename: str
) -> Tuple[np.ndarray or torch.Tensor, np.ndarray or torch.Tensor]:
    """
    Loads the pickled dataset.

    Parameter
    ---------
    filename: str
        The path to the pickled dataset.

    Returns
    -------
    Tuple[np.ndarray or torch.Tensor, np.ndarray or torch.Tensor]
        The first element is the features tensor.
        The second element is the labels tensor.
    """
    with open(filename, "rb") as tensor_file:
        dataset = pickle.load(tensor_file)
    return dataset
