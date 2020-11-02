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
"""Function for encoding features into lower-dimensional space"""
import MulticoreTSNE
import numpy as np
from sklearn.decomposition import PCA
import tsnecuda
from umap import UMAP


def encode_features(
    features: np.ndarray,
    seed: int = 42,
    dim: int = 30,
    use_cuda: bool = True,
    encoder: str = "pca",
) -> np.ndarray:
    supported_encoders = ["pca", "umap", "tsne"]
    assert encoder in supported_encoders, f"Encoder [{encoder}] is not supported."
    if encoder == "pca":
        encoder = PCA(n_components=dim, random_state=seed)
    elif encoder == "tsne" and use_cuda:
        print("[INFO] CUDA-based t-SNE only supports encoding to 2 dimensions.")
        encoder = tsnecuda.TSNE(random_seed=seed)
    elif encoder == "tsne" and not use_cuda:
        encoder = MulticoreTSNE.MulticoreTSNE(
            n_jobs=4, random_state=seed, n_components=dim
        )
    encoded_features = encoder.fit_transform(features)
    return encoded_features
