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
    long_description="PyTorch dataset loader for MNIST, Fashion-MNIST, EMNIST-Balanced, CIFAR10, SVHN, and MalImg datasets",
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
