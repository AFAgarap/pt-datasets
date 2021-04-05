import torch


class AGNews(torch.utils.data.Dataset):
    def __init__(
        self,
        train: bool = True,
        vectorizer: str = "tfidf",
        return_vectorizer: bool = False,
    ):
        pass
