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
import string
from typing import Dict, List, Tuple

import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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
