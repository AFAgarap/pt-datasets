import time

import numpy as np
import torch

from pt_datasets import load_dataset, create_dataloader


batch_size = 2048
train_data, test_data = load_dataset("multi_covid")
train_loader = create_dataloader(train_data, batch_size=batch_size)
test_loader = create_dataloader(test_data, batch_size=len(test_data))

train_labels = []
processed_train = []

for index, example in enumerate(train_loader):
    start_time = time.time()
    train_labels.append(example.get("label"))
    processed_train.append(example.get("image"))
    duration = time.time() - start_time
    print(f"[INFO] Processing batch {index} took {duration:.6f}s")

array = np.zeros((len(train_data), 3, 64, 64))
for index, row in enumerate(processed_train):
    offset = index * batch_size
    array[offset : offset + batch_size] = row

torch.save(array, "multi_covid_train.pt")
