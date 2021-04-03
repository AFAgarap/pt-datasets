from sklearn.datasets import fetch_20newsgroups
import torch

from pt_datasets.utils import preprocess_data, vectorize_text


class TwentyNewsgroups(torch.utils.data.Dataset):
    def __init__(self, vectorizer: str = "tfidf", return_vectorizer: bool = False):
        """
        Loads the 20 Newsgroups dataset.

        Parameters
        ----------
        vectorizer: str
            The vectorizer to use, options: [tfidf (default) | ngrams]
        return_vectorizer: bool
            Whether to return the vectorizer object or not.
        """
        self.train_set = fetch_20newsgroups(
            subset="train", remove=("headers", "foooters", "quotes")
        )
        self.test_set = fetch_20newsgroups(
            subset="test", remove=("headers", "footers", "quotes")
        )
        (train_features, train_labels) = preprocess_data(
            self.train_set.data, self.train_set.target
        )
        (test_features, test_labels) = preprocess_data(
            self.test_set.data, self.test_set.target
        )
        if return_vectorizer:
            train_features, vectorizer = vectorize_text(
                train_features, vectorizer=vectorizer
            )
        else:
            train_features = vectorize_text(train_features, vectorizer=vectorizer)
        test_features = vectorize_text(test_features, vectorizer=vectorizer)
        self.classes = self.train_set.target_names