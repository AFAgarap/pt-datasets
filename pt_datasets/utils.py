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
import os
from pathlib import Path
import string
import tarfile
from typing import Dict, List, Tuple
from zipfile import ZipFile

import cv2
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import torch

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


def preprocess_data(texts: List, labels: List) -> Tuple[List, np.ndarray]:
    """
    Loads the dataset from file, and returns the processed dataset.

    Parameters
    ----------
    texts: List
        The texts to vectorize.
    labels: List
        The corresponding labels for texts.

    Returns
    -------
    Tuple[List, np.ndarray]
        texts: List
            The preprocessed text features.
        labels: np.ndarray
            The corresponding labels for texts.
    """
    texts = list(
        map(
            lambda text: text.translate(str.maketrans("", "", string.punctuation)),
            texts,
        )
    )
    texts = list(
        map(
            lambda text: " ".join([word for word in text.split() if len(word) > 3]),
            texts,
        )
    )
    texts = list(map(lambda text: text.lower(), texts))
    texts = list(map(lambda text: text.split(), texts))
    en_stopwords = nltk.corpus.stopwords.words("english")
    texts = list(
        map(lambda text: [word for word in text if word not in en_stopwords], texts)
    )
    texts = list(map(lambda text: " ".join(text), texts))
    labels = np.array(labels, dtype=np.float32)
    labels -= 1
    return (texts, labels)


def vectorize_text(
    texts: List,
    vectorizer: str = "tfidf",
    ngram_range: Tuple = (3, 3),
    max_features: int = 2000,
    return_vectorizer: bool = False,
) -> np.ndarray:
    """
    Returns the n-Grams or TF-IDF vector representation of the text.

    Parameters
    ----------
    texts: List
        The texts to vectorize.
    vectorizer: str
        The vectorizer to use.
    ngram_range: Tuple
        The lower and upper boundary of the range
        of n-values for different n-grams to be extracted.
    max_features: int
        The maximum number of features to keep.
    return_vectorizer: bool
        Whether to return the vectorizer object or not.

    Returns
    -------
    vectors: np.ndarray
        The vector representation of the text.
    """
    supported_vectorizers = ["ngrams", "tfidf"]
    assert vectorizer in supported_vectorizers, f"{vectorizer} is not supported."

    if vectorizer == "tfidf":
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            max_df=0.5,
            smooth_idf=True,
            stop_words="english",
        )
    elif vectorizer == "ngrams":
        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            max_df=0.5,
            stop_words="english",
        )
    vectors = vectorizer.fit_transform(texts)
    vectors = vectors.toarray()
    vectors = vectors.astype(np.float32)
    return (vectors, vectorizer) if return_vectorizer else vectors


def unzip_dataset(dataset_filename: str) -> None:
    """
    Extracts the contents of a `.zip` or `.tar` file.

    Parameter
    ---------
    dataset_filename: str
        The path to the compressed dataset.
    """
    print(f"[INFO] Extracting {dataset_filename}...")
    if dataset_filename.endswith(".zip"):
        with ZipFile(dataset_filename, "r") as zip_object:
            zip_object.extractall(os.path.join(str(Path.home()), "datasets"))
    elif dataset_filename.endswith(".tar.xz"):
        with tarfile.open(dataset_filename, "r") as tar_object:
            tar_object.extractall(os.path.join(str(Path.home()), "datasets"))


def read_metadata(metadata_file: str) -> List:
    """
    Returns a nested list that consists of the following
    metadata for the dataset,
    ID, filename, class, source

    Parameter
    ---------
    metadata_filename: str
        The path to the dataset metadata.

    Returns
    -------
    data: List
        The metadata for the dataset.
    """
    with open(metadata_file, "r") as file:
        data = file.readlines()
    for index in range(len(data)):
        data[index] = data[index].strip("\n").split()
        if data[index][0] == "COVID":
            data[index] = [
                f"{data[index][0]} {data[index][1]}",
                data[index][2],
                data[index][3],
            ]
    return data


def crop_top(image: np.ndarray, percent: float = 8e-2) -> np.ndarray:
    """
    Returns an image whose top has been cropped out.

    Parameters
    ----------
    image: np.ndarray
        The image whose top will be cropped out.
    percent: float
        The percentage of top to crop.

    Returns
    -------
    image: np.ndarray
        The top cropped image.
    """
    offset = int(image.shape[0] * percent)
    return image[offset:]


def central_crop(image: np.ndarray) -> np.ndarray:
    """
    Returns a crop of the image center.
    Code from Wang et al. (2020):
    https://github.com/lindawangg/COVID-Net

    Parameter
    ---------
    image: np.ndarray
        The image whose central crop will be returned.

    Returns
    -------
    image: np.ndarray
        The central cropped image.
    """
    size = min(image.shape[0], image.shape[1])
    offset_height = int((image.shape[0] - size) / 2)
    offset_width = int((image.shape[1] - size) / 2)
    return image[
        offset_height : offset_height + size, offset_width : offset_width + size
    ]


def load_image(filename: str, size: Tuple = 224) -> torch.Tensor:
    """
    Loads the image from file.

    Parameters
    ----------
    filename: str
        The image data to load.
    size: int
        The size to use for the loaded image.

    Returns
    -------
    image: torch.Tensor:
        The loaded image.
    """
    image = cv2.imread(filename)
    image = crop_top(image)
    image = central_crop(image)
    image = cv2.resize(image, (size, size))
    return image
