from sklearn.datasets import fetch_20newsgroups
import torch


class TwentyNewsgroups(torch.utils.data.Dataset):
    def __init__(self):
        self.train_set = fetch_20newsgroups(
            subset="train", remove=("headers", "foooters", "quotes")
        )
        self.test_set = fetch_20newsgroups(
            subset="test", remove=("headers", "footers", "quotes")
        )
        self.classes = self.train_set.target_names
