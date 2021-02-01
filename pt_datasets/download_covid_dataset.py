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
"""Download functions for COVID19 datasets"""
import os
from pathlib import Path

import gdown

__author__ = "Abien Fred Agarap"


BINARY_COVID19_DATASET = (
    "https://drive.google.com/uc?id=1xfuoZmwjf5wu_H2xgqUevs9rjW1R-Sel"
)
MULTI_COVID19_DATASET = (
    "https://drive.google.com/uc?id=1zjtn3ZRZTIYXMo0NnWNHdOQtpw6IUvUl"
)


def download_binary_covid19_dataset():
    """
    Downloads the COVID19 Binary classification dataset.
    """
    path = os.path.join(str(Path.home()), "datasets")
    filename = "BinaryCOVID19Dataset.tar.xz"
    print("[INFO] Downloading the dataset...")
    gdown.download(BINARY_COVID19_DATASET, os.path.join(path, filename))


def download_covidx5_dataset():
    """
    Downloads the COVID19 multi-classification dataset.
    """
    path = os.path.join(str(Path.home()), "datasets")
    filename = "MultiCOVID19Dataset.tar.xz"
    print("[INFO] Downloading the dataset...")
    gdown.download(MULTI_COVID19_DATASET, os.path.join(path, filename))
