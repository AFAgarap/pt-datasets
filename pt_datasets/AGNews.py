import os
from pathlib import Path
from typing import Any, Tuple

import torch

from pt_datasets.utils import read_data, preprocess_data, vectorize_text


class AGNews(torch.utils.data.Dataset):
    def __init__(
        self,
        train: bool = True,
        vectorizer: str = "tfidf",
        return_vectorizer: bool = False,
    ):
        """
        Loads the AG News dataset.

        Parameters
        ----------
        train: bool
            Whether to load the training set or not.
        vectorizer: str
            The vectorizer to use, options: [tfidf (default) | ngrams]
        return_vectorizer: bool
            Whether to return the vectorizer object or not.
        """
        path = str(Path.home())
        path = os.path.join(path, "datasets")
        if train:
            path = os.path.join(path, "ag_news.train")
            dataset = read_data(path)
            features, labels = (list(dataset.keys()), list(dataset.values()))
            features, labels = preprocess_data(features, labels)
            if return_vectorizer:
                features, vectorizer = vectorize_text(
                    features, vectorizer, return_vectorizer=return_vectorizer
                )
            else:
                features = vectorize_text(
                    features, vectorizer, return_vectorizer=return_vectorizer
                )
        else:
            path = os.path.join(path, "ag_news.test")
            dataset = read_data(path)
            features, labels = (list(dataset.keys()), list(dataset.values()))
            features, labels = preprocess_data(features, labels)
            features = vectorize_text(features, vectorizer)
        self.data = features
        self.targets = labels
        self.classes = ["World", "Sports", "Business", "Sci/Tech"]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        text, target = self.data[index], self.targets[index]
        return (text, target)

    def __len__(self) -> int:
        return len(self.data)
