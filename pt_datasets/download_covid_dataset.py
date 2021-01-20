import os
from pathlib import Path

import gdown


BINARY_COVID19_DATASET = (
    "https://drive.google.com/uc?id=1rcYxrMmxtX3JSW9eHT20sTzmRHWrmGc2"
)


def download_binary_covid19_dataset():
    """
    Downloads the COVID19 Binary classification dataset
    """
    path = os.path.join(str(Path.home()), "torch_datasets")
    filename = "BinaryCOVID19Dataset.zip"
    gdown.download(BINARY_COVID19_DATASET, os.path.join(path, filename))
