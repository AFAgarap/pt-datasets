# PyTorch Datasets utility repository
# Copyright (C) 2020  Abien Fred Agarap
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Setup file for pt-datasets"""
import os
from os import path
from pathlib import Path
from setuptools import setup

import gdown

__author__ = "Abien Fred Agarap"


this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def _post_install():
    ag_news_setup = """
        echo "[INFO] Downloading AG News Dataset..."
        mkdir -p ~/datasets
        wget --no-clobber -O ~/datasets/ag_news.train https://raw.githubusercontent.com/AnubhavGupta3377/Text-Classification-Models-Pytorch/master/data/ag_news.train
        wget --no-clobber -O ~/datasets/ag_news.test https://raw.githubusercontent.com/AnubhavGupta3377/Text-Classification-Models-Pytorch/master/data/ag_news.test
        echo "[SUCCESS] Done downloading AG News Dataset."
        """
    os.system(ag_news_setup)


def _download_imdb():
    imdb_dataset = "https://drive.google.com/uc?id=1VhjJUc6hlJfHmEln8b0xfxuzUC36bBHE"
    imdb_path = os.path.join(str(Path.home()), "datasets")
    filename = "IMDB Dataset.csv"
    print("[INFO] Downloading the IMDB dataset...")
    gdown.download(imdb_dataset, os.path.join(imdb_path, filename))


def _download_yelp():
    yelp_dataset = "https://drive.google.com/uc?id=1-RG97iRFppt3zK939cijAH8JQIhOyqu8"
    yelp_path = os.path.join(str(Path.home()), "datasets")
    filename = "yelp.csv"
    print("[INFO] Downloading the Yelp dataset...")
    gdown.download(yelp_dataset, os.path.join(yelp_path, filename))


setup(
    name="pt-datasets",
    version="0.13.0",
    packages=["pt_datasets"],
    url="https://github.com/AFAgarap/pt-datasets",
    license="AGPL-3.0 License",
    author="Abien Fred Agarap",
    author_email="abienfred.agarap@gmail.com",
    description="Library for loading PyTorch datasets and data loaders.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy==1.20.0",
        "torchvision==0.9.1",
        "umap_learn==0.4.6",
        "torch==1.8.1",
        "MulticoreTSNE==0.1",
        "scikit_learn==0.23.2",
        "gdown==3.12.2",
        "nltk==3.5",
        "cmake==3.18.0",
        "opencv-python==4.2.0.34",
        "imbalanced_learn==0.7.0",
    ],
)


_post_install()
_download_imdb()
_download_yelp()
