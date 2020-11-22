from os import path
from setuptools import setup


this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pt-datasets",
    version="0.3.0",
    packages=["pt_datasets"],
    url="https://github.com/AFAgarap/pt-datasets",
    license="AGPL-3.0 License",
    author="Abien Fred Agarap",
    author_email="abienfred.agarap@gmail.com",
    description="PyTorch dataset loader for MNIST, Fashion-MNIST, EMNIST-Balanced, CIFAR10, SVHN, and MalImg datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy==1.19.0",
        "torchvision==0.7.0",
        "umap_learn==0.4.6",
        "torch==1.6.0",
        "MulticoreTSNE==0.1",
        "scikit_learn==0.23.2",
        "gdown==3.12.2",
    ],
)
