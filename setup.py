import os
from os import path
from setuptools import setup


this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def _post_install():
    ag_news_setup = """
        echo "[INFO] Downloading AG News Dataset..."
        mkdir -p ~/torch_datasets
        wget --no-clobber -O ~/torch_datasets/ag_news.train https://raw.githubusercontent.com/AnubhavGupta3377/Text-Classification-Models-Pytorch/master/data/ag_news.train
        wget --no-clobber -O ~/torch_datasets/ag_news.test https://raw.githubusercontent.com/AnubhavGupta3377/Text-Classification-Models-Pytorch/master/data/ag_news.test
        echo "[SUCCESS] Done downloading AG News Dataset."
        """
    os.system(ag_news_setup)


setup(
    name="pt-datasets",
    version="0.9.3",
    packages=["pt_datasets"],
    url="https://github.com/AFAgarap/pt-datasets",
    license="AGPL-3.0 License",
    author="Abien Fred Agarap",
    author_email="abienfred.agarap@gmail.com",
    description="Library for loading PyTorch datasets and data loaders.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy==1.19.4",
        "torchvision==0.8.2",
        "umap_learn==0.4.6",
        "torch==1.7.1",
        "MulticoreTSNE==0.1",
        "scikit_learn==0.23.2",
        "gdown==3.12.2",
        "nltk==3.5",
    ],
)


_post_install()
