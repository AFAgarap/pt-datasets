from sklearn.datasets import fetch_20newsgroups
import torch


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
        self.classes = self.train_set.target_names
