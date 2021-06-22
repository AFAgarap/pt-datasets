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

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        pass

    def __len__(self) -> int:
        pass
