import csv

import torch


class IMDB(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self):
        pass

    def __len__(self):
        pass

    @staticmethod
    def read_data(data: str = "data/IMDB Dataset.csv"):
        dataset = {}
        with open(data, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                dataset[row[1]] = row[0]
        return dataset
