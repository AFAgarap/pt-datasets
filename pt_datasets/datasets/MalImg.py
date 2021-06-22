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
"""Class for Malware Image dataset"""
import os
from pathlib import Path
from typing import Any, Tuple

import gdown
import numpy as np
from sklearn.model_selection import train_test_split
import torch


class MalImg(torch.utils.data.Dataset):
    _download_url = "https://drive.google.com/uc?id=1Y6Ha5Jir8EI726KdwAKWHVU-oWmPqNDT"
    _filename = "malimg_dataset_32x32.npy"

    def __init__(self, train: bool = True, download: bool = True):
        """
        Loads the Malware Image dataset.

        Parameters
        ----------
        train: bool
            Whether to load the training split or not.
        download: bool
            Whether to download the dataset or not.
        """
        super().__init__()
        dataset_path = os.path.join(str(Path.home()), "datasets")
        if not os.path.exists(dataset_path):
            os.mkdir(dataset_path)
        if download or not os.path.isfile(os.path.join(dataset_path, MalImg._filename)):
            gdown.download(
                MalImg._download_url,
                os.path.join(dataset_path, MalImg._filename),
                quiet=True,
            )
        dataset = np.load(os.path.join(dataset_path, MalImg._filename))
        if train:
            train_data, _ = train_test_split(
                dataset, test_size=3e-1, random_state=torch.random.initial_seed()
            )
            features, labels = train_data[:, : (32 ** 2)], train_data[:, -1]
        else:
            _, test_data = train_test_split(
                dataset, test_size=3e-1, random_state=torch.random.initial_seed()
            )
            features, labels = test_data[:, : (32 ** 2)], test_data[:, -1]
        self.features = features
        self.labels = labels

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        features, labels = self.features[index], self.labels[index]
        return (features, labels)

    def __len__(self) -> int:
        pass
