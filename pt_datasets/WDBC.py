from typing import Any, Tuple

import torch


class WDBC(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        pass

    def __len__(self):
        pass
