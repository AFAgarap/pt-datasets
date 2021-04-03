from typing import Any, Tuple

from sklearn.datasets import fetch_20newsgroups
import torch

from pt_datasets.utils import preprocess_data, vectorize_text


class TwentyNewsgroups(torch.utils.data.Dataset):
    def __init__(
        self,
        train: bool = True,
        vectorizer: str = "tfidf",
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
        return_vectorizer: bool
            Whether to return the vectorizer object or not.
        """
        if train:
            self.train_set = fetch_20newsgroups(
                subset="train", remove=("headers", "foooters", "quotes")
            )
            (features, labels) = preprocess_data(
                self.train_set.data, self.train_set.target
            )
            if return_vectorizer:
                features, vectorizer = vectorize_text(features, vectorizer=vectorizer)
            else:
                features = vectorize_text(features, vectorizer=vectorizer)
        else:
            self.test_set = fetch_20newsgroups(
                subset="test", remove=("headers", "footers", "quotes")
            )
            (features, labels) = preprocess_data(
                self.test_set.data, self.test_set.target
            )
            features = vectorize_text(features, vectorizer=vectorizer)
        self.classes = self.train_set.target_names
        self.data = features
        self.targets = labels

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        pass
