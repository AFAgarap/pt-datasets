import time
from typing import List, Tuple

import numpy as np
import torch

from pt_datasets import load_dataset, create_dataloader


def unpack_examples(data_loader: torch.utils.data.Dataset) -> Tuple[List, List]:
    features = []
    labels = []
    for index, example in enumerate(data_loader):
        start_time = time.time()
        features.append(example.get("image"))
        labels.append(example.get("label"))
        duration = time.time() - start_time
        print(f"[INFO] Processing batch {index} took {duration:.6f}s")
    return features, labels


def vectorize_examples(
    features: List, labels: List, dataset_size: int, batch_size: int = 2048
) -> Tuple[np.ndarray, np.ndarray]:
    array = np.zeros((dataset_size, 3, 64, 64))
    for index, row in enumerate(features):
        offset = index * batch_size
        array[offset : offset + batch_size] = row
    labels = np.array(labels)
    array = array.astype("float32")
    return array, labels


def export_dataset(dataset: np.ndarray, filename: str) -> None:
    if not filename.endswith(".pt"):
        filename = f"{filename}.pt"
    torch.save(dataset, filename)


def main():
    batch_size = 2048
    train_data, test_data = load_dataset("multi_covid")
    train_loader = create_dataloader(train_data, batch_size=batch_size)
    train_features, train_labels = unpack_examples(train_loader)
    train_features, train_labels = vectorize_examples(
        train_features, train_labels, dataset_size=len(train_data)
    )
    train_dataset = (train_features, train_labels)
    export_dataset(train_dataset, "train.pt")


if __name__ == "__main__":
    main()
