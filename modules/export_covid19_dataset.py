import argparse
import os
import time
from typing import List, Tuple

import numpy as np
import torch

from pt_datasets import load_dataset, create_dataloader


__author__ = "Abien Fred Agarap"


def parse_args():
    parser = argparse.ArgumentParser(description="COVID19 dataset exporter")
    group = parser.add_argument_group("Parameters")
    group.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="multi_covid",
        help="the COVID19 dataset to export, options: [binary_covd | multi_covid (default)]",
    )
    group.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=2048,
        help="the mini-batch size to use, default: [2048]",
    )
    group.add_argument(
        "-f",
        "--filename",
        type=str,
        default="torch_datasets/MultiCOVID19Dataset",
        help="the path where the datasets shall be exported.",
    )
    group.add_argument(
        "-s",
        "--size",
        type=int,
        default=64,
        help="the size to use for resizing the COVID19 data, default: [64]",
    )
    arguments = parser.parse_args()
    return arguments


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
    labels_array = np.zeros((dataset_size))
    for index, (row, label) in enumerate(zip(features, labels)):
        offset = index * batch_size
        array[offset : offset + batch_size] = row
        labels_array[offset : offset + batch_size] = label
    labels_array = labels_array.astype("uint8")
    array = array.astype("float32")
    return array, labels_array


def export_dataset(dataset: np.ndarray, filename: str) -> None:
    if not filename.endswith(".pt"):
        filename = f"{filename}.pt"
    torch.save(dataset, filename)


def main(arguments):
    batch_size = arguments.batch_size
    path = arguments.filename
    train_data, test_data = load_dataset("multi_covid")
    train_loader = create_dataloader(train_data, batch_size=batch_size)
    test_loader = create_dataloader(test_data, batch_size=len(test_data))
    train_features, train_labels = unpack_examples(train_loader)
    train_features, train_labels = vectorize_examples(
        train_features,
        train_labels,
        dataset_size=len(train_data),
        batch_size=batch_size,
    )
    train_dataset = (train_features, train_labels)
    export_dataset(train_dataset, os.path.join(path, "train.pt"))
    test_features, test_labels = unpack_examples(test_loader)
    test_features, test_labels = vectorize_examples(
        test_features, test_labels, dataset_size=len(test_data), batch_size=batch_size
    )
    test_dataset = (test_features, test_labels)
    export_dataset(test_dataset, os.path.join(path, "test.pt"))


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
