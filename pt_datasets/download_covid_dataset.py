import os
from pathlib import Path

import gdown


BINARY_COVID_DATASET = (
    "https://drive.google.com/uc?id=1rcYxrMmxtX3JSW9eHT20sTzmRHWrmGc2"
)


def download_binary_covid_dataset():
    """
    Downloads the COVID19 Binary classification dataset
    """
    path = os.path.join(str(Path.home()), "datasets")
    filename = "BinaryCOVIDDataset.zip"
    gdown.download(BINARY_COVID_DATASET, os.path.join(path, filename))
