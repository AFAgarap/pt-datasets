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
"""Class for 20Newsgroups dataset"""
from typing import Any, Tuple

from sklearn.datasets import fetch_20newsgroups
import torch

from pt_datasets.utils import preprocess_data, vectorize_text

__author__ = "Abien Fred Agarap"


class TwentyNewsgroups(torch.utils.data.Dataset):
    def __init__(
        self,
        train: bool = True,
        vectorizer: str = "tfidf",
        ngram_range: Tuple = (3, 3),
        return_vectorizer: bool = False,
    ):
        """
        Loads the 20 Newsgroups dataset.

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
        if train:
            self.dataset = fetch_20newsgroups(
                subset="train", remove=("headers", "foooters", "quotes")
            )
            (features, labels) = preprocess_data(self.dataset.data, self.dataset.target)
            if return_vectorizer:
                features, vectorizer = vectorize_text(
                    features,
                    vectorizer=vectorizer,
                    return_vectorizer=return_vectorizer,
                    ngram_range=ngram_range,
                )
                self.vectorizer = vectorizer
            else:
                features = vectorize_text(
                    features, vectorizer=vectorizer, ngram_range=ngram_range
                )
        else:
            self.dataset = fetch_20newsgroups(
                subset="test", remove=("headers", "footers", "quotes")
            )
            (features, labels) = preprocess_data(self.dataset.data, self.dataset.target)
            features = vectorize_text(
                features, vectorizer=vectorizer, ngram_range=ngram_range
            )
        self.classes = self.dataset.target_names
        self.data = features
        self.targets = labels

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        text, target = self.data[index], self.targets[index]
        return (text, target)

    def __len__(self) -> int:
        return len(self.data)
