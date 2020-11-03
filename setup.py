from setuptools import setup

with open("requirements.txt", "r") as file:
    requirements = file.read().splitlines()

setup(
    name="pt-datasets",
    version="0.1.1",
    packages=["pt_datasets"],
    url="https://github.com/AFAgarap/pt-datasets",
    license="AGPL-3.0 License",
    author="Abien Fred Agarap",
    author_email="abienfred.agarap@gmail.com",
    long_description="PyTorch dataset loader for MNIST, Fashion-MNIST, EMNIST-Balanced, CIFAR10, and SVHN datasets",
    install_requires=requirements,
)
