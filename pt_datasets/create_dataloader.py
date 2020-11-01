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
"""Function for creating a data loader"""
import torch

__author__ = "Abien Fred Agarap"


def create_dataloader(
    dataset: object, batch_size: int = 16, shuffle: bool = True, num_workers: int = 0
) -> object:
    """
    Returns a data loader object, ready to be used by a model.

    Parameters
    ----------
    dataset: object
        The dataset from `torchvision.datasets`.
    batch_size: int
        The mini-batch size for the data loading. Default is [16].
    shuffle: bool
        Whether to shuffle dataset or not. Default is [True].
    num_workers: int
        The number of subprocesses to use for data loading. Default is [0].

    Returns
    -------
    data_loader: torch.utils.dataloader.DataLoader
        The data loader object to be used by a model.
    """
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader
