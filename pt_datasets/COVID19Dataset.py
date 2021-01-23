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
from typing import Dict

import torch
import torchvision

from pt_datasets.utils import read_metadata, load_image

__author__ = "Abien Fred Agarap"


DATASET_DIR = os.path.join(str(Path.home()), "torch_datasets")
DATASET_PATH = os.path.join(DATASET_DIR, "data")
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
        """
        BINARY_COVID19_DIR = os.path.join(DATASET_DIR, "BinaryCOVID19Dataset")
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
        """
        MULTI_COVID19_DIR = os.path.join(DATASET_DIR, "MultiCOVID19Dataset")
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
