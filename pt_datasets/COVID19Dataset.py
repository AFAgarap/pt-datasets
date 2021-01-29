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


class COVID19Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        classes: str = "binary",
        transform: torchvision.transforms = None,
        train: bool = True,
    ):
        """
        Builds the preprocessed COVID19 dataset.

        Parameters
        ----------
        classes: str
            The dataset to load, either "binary" or "multi".
        transform: torchvision.transforms
            The transformation pipeline to use for image preprocessing.
        train: bool
            Whether to load the training set or not.
        """
        if classes == "binary":
            self.classes = ["negative", "positive"]
            if train:
                dataset = torch.load(os.path.join(BINARY_COVID19_DIR, "train.pt"))
            else:
                dataset = torch.load(os.path.join(BINARY_COVID19_DIR, "test.pt"))
        elif classes == "multi":
            self.classes = ["normal", "pneumonia", "COVID-19"]
            if train:
                dataset = torch.load(os.path.join(MULTI_COVID19_DIR, "train.pt"))
            else:
                dataset = torch.load(os.path.join(MULTI_COVID19_DIR, "test.pt"))
        self.data = dataset[0]
        self.labels = dataset[1]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx].astype("int64")
        return (image, label)


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
        preproceseed: bool
            Whether to load preprocessed dataset or not.
        """
        if preprocessed:
            if not os.path.isfile(os.path.join(BINARY_COVID19_DIR, f"train_{size}.pt")):
                print(
                    "[INFO] No preprocessed training dataset found. Preprocessing now..."
                )
                preprocess_dataset(train=True, size=size)
            if not os.path.isfile(os.path.join(BINARY_COVID19_DIR, f"test_{size}.pt")):
                print("[INFO] No preprocessed test dataset found. Preprocessing now...")
                preprocess_dataset(train=False, size=size)
            if train:
                dataset = torch.load(
                    os.path.join(BINARY_COVID19_DIR, f"train_{size}.pt")
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
        """
        if preprocessed:
            if not os.path.isfile(os.path.join(MULTI_COVID19_DIR, f"train_{size}.pt")):
                print(
                    "[INFO] No preprocessed training dataset found. Preprocessing now..."
                )
                preprocess_dataset(train=True, size=size)
            if not os.path.isfile(os.path.join(MULTI_COVID19_DIR, f"test_{size}.pt")):
                print("[INFO] No preprocessed test dataset found. Preprocessing now...")
                preprocess_dataset(train=False, size=size)
            if train:
                dataset = torch.load(
                    os.path.join(MULTI_COVID19_DIR, f"train_{size}.pt")
                )
            else:
                dataset = torch.load(
                    os.path.join(MULTI_COVID19_DIR, f"test_{size}.pt")
                )
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
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx) -> Dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()
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


def export_dataset(dataset: np.ndarray, filename: str) -> None:
    if not filename.endswith(".pt"):
        filename = f"{filename}.pt"
    torch.save(dataset, filename)


def preprocess_dataset(
    train: bool = False, size: int = 64, batch_size: int = 2048
) -> None:
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    if train:
        print("[INFO] Loading dataset...")
        train_data = BinaryCOVID19Dataset(train=True, size=size, transform=transform)
        print("[INFO] Creating data loader...")
        train_loader = create_dataloader(train_data, batch_size=batch_size)
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
                os.path.join(BINARY_COVID19_DIR, f"train_{size}.pt")
            )
        )
        export_dataset(
            train_dataset, os.path.join(BINARY_COVID19_DIR, f"train_{size}.pt")
        )
    else:
        print("[INFO] Loading dataset...")
        test_data = BinaryCOVID19Dataset(train=False, size=size, transform=transform)
        print("[INFO] Creating data loader...")
        test_loader = create_dataloader(test_data, batch_size=batch_size)
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
                os.path.join(BINARY_COVID19_DIR, f"test_{size}.pt")
            )
        )
        export_dataset(
            test_dataset, os.path.join(BINARY_COVID19_DIR, f"test_{size}.pt")
        )
