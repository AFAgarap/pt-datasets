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
"""Class for IMDB dataset"""
import csv

import torch

__author__ = "Abien Fred Agarap"


class IMDB(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self):
        pass

    def __len__(self):
        pass

    @staticmethod
    def load_data(data: str = "data/IMDB Dataset.csv"):
        dataset = {}
        with open(data, "r") as file:
            reader = csv.reader(file)
            for index, row in enumerate(reader):
                if index == 0:
                    pass
                else:
                    if row[1] == "positive":
                        label = 1
                    elif row[1] == "negative":
                        label = 0
                    dataset[row[0]] = label
        return dataset
