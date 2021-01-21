import os
from pathlib import Path
from typing import Dict

import torch

from pt_datasets.utils import read_metadata, load_image


DATASET_DIR = os.path.join(str(Path.home()), "torch_datasets/BinaryCOVID19Dataset")
DATASET_PATH = os.path.join(DATASET_DIR, "data")
TRAIN_METADATA = os.path.join(DATASET_DIR, "train_split.txt")
TEST_METADATA = os.path.join(DATASET_DIR, "test_split.txt")


class BinaryCOVID19Dataset(torch.utils.data.Dataset):
    """
    Dataset class for the COVID19 binary classification dataset.
    """

    def __init__(self, train: bool = True, transform=None):
        """
        Builds the COVID19 binary classification dataset.

        Parameter
        ---------
        train: bool
            Whether to load the training set or not.
        """
        if train:
            path = os.path.join(DATASET_PATH, "train")
            self.annotations = read_metadata(TRAIN_METADATA)
            self.root_dir = path
        else:
            path = os.path.join(DATASET_PATH, "test")
            self.annotations = read_metadata(TEST_METADATA)
            self.root_dir = path
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx) -> Dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = os.path.join(self.root_dir, self.annotations[idx][1])
        image = load_image(image_name, 128)
        if self.transform:
            image = self.transform(image)
        label = self.annotations[idx][2]
        label = 0 if label == "negative" else 1
        sample = {"image": image, "label": label}
        return sample
