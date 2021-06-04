from typing import Any, Tuple

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

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        pass

    def __len__(self):
        pass
