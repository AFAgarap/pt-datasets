import os
from pathlib import Path

import torch

from pt_datasets.utils import read_data


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
        else:
            path = os.path.join(path, "ag_news.test")
            dataset = read_data(path)
            features, labels = (list(dataset.keys()), list(dataset.values()))
