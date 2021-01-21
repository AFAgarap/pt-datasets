import os
from pathlib import Path

import gdown


BINARY_COVID19_DATASET = (
    "https://drive.google.com/uc?id=1sGHrm76Rt_qRoHo-nU3A4iK8RMuTRMB6"
)


def download_binary_covid19_dataset():
    """
    Downloads the COVID19 Binary classification dataset
    """
    path = os.path.join(str(Path.home()), "datasets")
    filename = "BinaryCOVID19Dataset.tar.xz"
    print("[INFO] Downloading the dataset...")
    gdown.download(BINARY_COVID19_DATASET, os.path.join(path, filename))
