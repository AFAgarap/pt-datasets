from typing import Any, Tuple

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
        features, labels = data.data, data.target
        train_features, test_features, train_labels, test_labels = train_test_split(
            features,
            labels,
            random_state=torch.random.initial_seed(),
            test_size=3e-1,
            shuffle=True,
        )
        scaler = StandardScaler()
        if train:
            train_features = scaler.fit_transform(train_features)
            train_features = train_features.astype("float32")
            self.data = train_features
            self.targets = train_labels
        else:
            test_features = scaler.fit_transform(test_features)
            test_features = test_features.astype("float32")
            self.data = test_features
            self.targets = test_labels

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        features, labels = self.data[index], self.targets[index]
        return (features, labels)

    def __len__(self):
        return len(self.data)
