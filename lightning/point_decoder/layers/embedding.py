import torch
import torch.nn as nn
from typing import Callable
from .normalization import CustomGroupNorm1d


class PointEmbedding(nn.Module):

    expansion = 4

    def __init__(
        self,
        in_features: int,
        out_features: int,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = CustomGroupNorm1d
    ) -> None:
        super().__init__()
        assert out_features % 2 == 0

        self.in_features = in_features
        self.hidden_features = in_features * self.expansion
        self.out_features = out_features

        self.embedding = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_features),
            act_layer(),
            nn.Linear(self.hidden_features, self.out_features)
        )

    def forward(self, positions, features):
        assert positions.ndim == features.ndim == 3
        features = torch.cat([positions, features], dim=-1)
        return self.embedding(features)
