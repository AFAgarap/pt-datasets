from typing import Any, Tuple

from sklearn.datasets import load_breast_cancer
import torch


class WDBC(torch.utils.data.Dataset):
    def __init__(self, train: bool = True):
        """
        Loads the WDBC dataset.

        Parameter
        ---------
        train: bool
            Whether to load the training set or not.
        """
        super().__init__()
        data = load_breast_cancer()
        self.classes = data.target_names

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        pass

    def __len__(self):
        pass
