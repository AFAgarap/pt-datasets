import torch
import torchvision


def load_dataset(
    name: str = "mnist", data_folder: str = "~/torch_datasets"
) -> Tuple[object, object]:
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    name = name.lower()

    if name == "mnist":
        train_dataset = torchvision.datasets.MNIST(
            root=data_folder, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=data_folder, train=False, download=True, transform=transform
        )
    elif name == "fashion_mnist":
        train_dataset = torchvision.datasets.FashionMNIST(
            root=data_folder, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root=data_folder, train=False, download=True, transform=transform
        )
    elif name == "emnist":
        train_dataset = torchvision.datasets.EMNIST(
            root=data_folder,
            train=True,
            split="balanced",
            download=True,
            transform=transform,
        )
        test_dataset = torchvision.datasets.EMNIST(
            root=data_folder,
            train=False,
            split="balanced",
            download=True,
            transform=transform,
        )
    elif name == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_folder, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_folder, train=False, download=True, transform=transform
        )
    elif name == "svhn":
        train_dataset = torchvision.datasets.SVHN(
            root=data_folder, split="train", download=True, transform=transform
        )
        test_dataset = torchvision.datasets.SVHN(
            root=data_folder, split="test", download=True, transform=transform
        )
    return (train_dataset, test_dataset)
