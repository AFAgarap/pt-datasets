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
from typing import Tuple

from sklearn.model_selection import train_test_split
import torch

from pt_datasets.utils import preprocess_data, vectorize_text

__author__ = "Abien Fred Agarap"


class IMDB(torch.utils.data.Dataset):
    def __init__(
        self,
        train: bool = True,
        vectorizer: str = "tfidf",
        ngram_range: Tuple = (3, 3),
        return_vectorizer: bool = False,
    ):
        super().__init__()
        self.classes = ["Negative", "Positive"]
        dataset = IMDB.load_data()
        (texts, labels) = (list(dataset.keys()), list(dataset.values()))
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=3e-1, random_state=42, shuffle=True
        )
        if train:
            features, labels = preprocess_data(train_texts, train_labels)
            if return_vectorizer:
                features, vectorizer = vectorize_text(
                    features,
                    vectorizer,
                    return_vectorizer=return_vectorizer,
                    ngram_range=ngram_range,
                )
                self.vectorizer = vectorizer
            else:
                features = vectorize_text(
                    features,
                    vectorizer,
                    return_vectorizer=return_vectorizer,
                    ngram_range=ngram_range,
                )
        else:
            features, labels = preprocess_data(test_texts, test_labels)
            features = vectorize_text(features, vectorizer, ngram_range=ngram_range)

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
