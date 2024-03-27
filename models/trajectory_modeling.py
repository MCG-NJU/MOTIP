# Copyright (c) RuopengGao. All Rights Reserved.
# About: Modeling the targets' trajectories.
#        In this streamlined version, we only use a simple FFN for this part.
import torch.nn as nn

from models.ffn import FFN


class TrajectoryAugmentation(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            dim_feedforward: int,
            dropout: float,
            device: str,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.device = device

        d_dim = 1 * self.hidden_dim
        ffn_dim = 1 * self.dim_feedforward

        self.d_dim = d_dim
        self.ffn_dim = ffn_dim

        # We only use a simple FFN for trajectory augmentation, as a streamlined baseline.
        # I think more complex models can be used here, such as transformers, in the future.
        self.trajectory_ffn = FFN(
            d_model=d_dim,
            d_ffn=ffn_dim,
            dropout=self.dropout,
            activation="GELU"
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, seq_features):
        seq_embeds = self.trajectory_ffn(seq_features)
        return seq_embeds
