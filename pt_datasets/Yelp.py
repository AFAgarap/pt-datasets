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
"""Class for Yelp dataset"""
import csv
from typing import Tuple

import torch

__author__ = "Abien Fred Agarap"


class Yelp(torch.utils.data.Dataset):
    def __init__(
        self,
        train: bool = True,
        vectorizer: str = "tfidf",
        ngram_range: Tuple = (3, 3),
        return_vectorizer: bool = False,
    ):
        """
        Loads the Yelp dataset.

        Parameters
        ----------
        train: bool
            Whether to load the training set or not.
        vectorizer: str
            The vectorizer to use, options: [tfidf (default) | ngrams]
        ngram_range: Tuple
            The lower and upper bound of ngram range to use.
            Default: [(3, 3)]
        return_vectorizer: bool
            Whether to return the vectorizer object or not.
        """
        super().__init__()
        self.classes = ["Negative", "Positive"]

    def __getitem__(self):
        pass

    def __len__(self):
        pass

    @staticmethod
    def load_data(data: str = "~/datasets/yelp.csv"):
        dataset = {}
        with open(data, "r") as file:
            reader = csv.reader(file)
            for index, row in enumerate(reader):
                if index == 0:
                    pass
                else:
                    dataset[row[0]] = int(row[1])
        return dataset
