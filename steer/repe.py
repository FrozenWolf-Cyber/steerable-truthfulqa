"""
RepE — Representation Engineering (arXiv 2310.01405).
PCA first component on (pos − neg) differences.
"""

import torch
from torch import Tensor
from sklearn.decomposition import PCA
from .base import VecSteer


class RepE(VecSteer):
    def __init__(self):
        super().__init__()
        self.pca = PCA(n_components=1)

    @torch.no_grad()
    def fit(self, pos_X: Tensor, neg_X: Tensor) -> "RepE":
        n = min(len(pos_X), len(neg_X))
        diff = (pos_X[:n] - neg_X[:n]).cpu().numpy()
        self.pca.fit(diff)
        self.steer_vec = torch.as_tensor(
            self.pca.components_[0], device=pos_X.device,
        )
        return self
