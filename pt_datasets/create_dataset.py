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
"""Function for creating Tensor datasets"""
import torch


def create_dataset(
    features: torch.Tensor, labels: torch.Tensor
) -> torch.utils.data.TensorDataset:
    """
    Returns a torch.utils.data.TensorDataset object
    to be used for a data loader object.

    Parameters
    ----------
    features: torch.Tensor
        The input features.
    labels: torch.Tensor
        The input labels.

    Returns
    -------
    dataset: torch.utils.data.TensorDataset
        The Tensor Dataset object.
    """
    if not isinstance(features, torch.Tensor):
        features = torch.from_numpy(features)
    if not isinstance(labels, torch.Tensor):
        labels = torch.from_numpy(labels)
    dataset = torch.utils.data.TensorDataset(features, labels)
    return dataset
