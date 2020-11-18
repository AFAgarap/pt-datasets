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
import csv
from typing import Dict

__author__ = "Abien Fred Agarap"


def read_data(corpus_file: str, label_column: int = 0, document_start: int = 2) -> Dict:
    """
    Returns a <key, value> pair of the loaded dataset
    where the key is the text data and the value is the data label.
    Parameters
    ----------
    corpus_file: str
        The filename of the dataset to load.
    label_column: int
        The column number of the dataset label (zero-indexed).
    document_start: int
        The number of columns in the dataset.
    Returns
    -------
    dataset: Dict
        The <key, value> pair representing the text data and their labels.
    """
    dataset = dict()
    with open(corpus_file, "r", encoding="utf-8") as text_data:
        if corpus_file.endswith(".csv"):
            text_data = csv.reader(text_data, delimiter=",")
            for index, line in enumerate(text_data):
                text = line[-1]
                label = int(line[label_column])
                dataset[text] = label
        else:
            for line in text_data:
                columns = line.strip().split(maxsplit=document_start)
                text = columns[-1]
                label = int(columns[label_column].strip("__label__"))
                dataset[text] = label
    return dataset
