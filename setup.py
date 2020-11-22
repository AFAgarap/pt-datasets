from setuptools import setup


setup(
    name="pt-datasets",
    version="0.3.0",
    packages=["pt_datasets"],
    url="https://github.com/AFAgarap/pt-datasets",
    license="AGPL-3.0 License",
    author="Abien Fred Agarap",
    author_email="abienfred.agarap@gmail.com",
    description="PyTorch dataset loader for MNIST, Fashion-MNIST, EMNIST-Balanced, CIFAR10, SVHN, and MalImg datasets",
    long_description="""
        # PyTorch Datasets

        [![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
        [![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-377/)
        [![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-382/)

        ## Overview

        This repository is meant for easier and faster access to commonly used
        benchmark datasets. Using this repository, one can load the datasets in a
        ready-to-use fashion for PyTorch models. Additionally, this can be used to load
        the low-dimensional features of the aforementioned datasets, encoded using PCA,
        t-SNE, or UMAP.

        ## Datasets

        - MNIST
        - Fashion-MNIST
        - EMNIST-Balanced
        - CIFAR10
        - SVHN
        - MalImg
        - AG News

        ## Usage

        It is recommended to use a virtual environment to isolate the project dependencies.

        ```shell script
        $ virtualenv env --python=python3  # we use python 3
        $ pip install pt-datasets  # install the package
        ```

        We use the [`tsnecuda`](https://github.com/CannyLab/tsne-cuda) library for the
        CUDA-accelerated t-SNE encoder, which can be installed by following the
        [instructions](https://github.com/CannyLab/tsne-cuda/wiki/Installation) in its wiki.

        But there is also a provided script for installing `tsne-cuda` from source.

        ```shell script
        $ bash setup/install_tsnecuda
        ```

        Do note that this script has only been tested on an Ubuntu 20.04 LTS system
        with Nvidia GTX960M GPU.

        We can then use this package for loading ready-to-use data loaders,

        ```python
        from pt_datasets import load_dataset, create_dataloader

        # load the training and test data
        train_data, test_data = load_dataset(name="cifar10")

        # create a data loader for the training data
        train_loader = create_dataloader(
            dataset=train_data, batch_size=64, shuffle=True, num_workers=1
        )

        ...

        # use the data loader for training
        model.fit(train_loader, epochs=10)
        ```
    """,
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
