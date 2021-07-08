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
import torchvision


class MalImg(torch.utils.data.Dataset):
    _download_url = "https://drive.google.com/uc?id=1ljOv9NnEsyOPTVC4RFgUGYDy189pNl2S"
    _filename = "malimg_dataset_32x32.npy"

    def __init__(
        self,
        train: bool = True,
        download: bool = True,
        transform: torchvision.transforms = None,
    ):
        """
        Loads the Malware Image dataset.

        Parameters
        ----------
        train: bool
            Whether to load the training split or not.
        download: bool
            Whether to download the dataset or not.
        transform: torchvision.transform
            The transformation pipeline to use for image preprocessing.
        """
        super().__init__()
        self.classes = (
            "Adialer.C",
            "Adialer.C",
            "Adialer.C",
            "Allaple.L",
            "Alueron.gen!J",
            "Autorun.K",
            "C2Lop.P",
            "C2Lop.gen!G",
            "Dialplatform.B",
            "Dontovo.A",
            "Fakerean",
            "Instantaccess",
            "Lolyda.AA 1",
            "Lolyda.AA 2",
            "Lolyda.AA 3",
            "Lolyda.AT",
            "Malex.gen!J",
            "Obfuscator.AD",
            "Rbot!gen",
            "Skintrim.N",
            "Swizzor.gen!E",
            "Swizzor.gen!I",
            "VB.AT",
            "Wintrim.BX",
            "Yuner.A",
        )
        dataset_path = os.path.join(str(Path.home()), "datasets")
        if not os.path.exists(dataset_path):
            os.mkdir(dataset_path)
        if download or not os.path.isfile(os.path.join(dataset_path, MalImg._filename)):
            gdown.download(
                MalImg._download_url,
                os.path.join(dataset_path, MalImg._filename),
                quiet=True,
            )
        dataset = np.load(
            os.path.join(dataset_path, MalImg._filename), allow_pickle=True
        )
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
        self.data = features.reshape(-1, 32, 32)
        self.targets = labels
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        features = self.data[index]
        if self.transform:
            features = self.transform(features)
        labels = self.targets[index].astype("int64")
        return (features, labels)

    def __len__(self) -> int:
        return len(self.data)
