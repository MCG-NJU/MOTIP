# Copyright (c) Ruopeng Gao. All Rights Reserved.
# About: 创建一个简单的 FFN Module
import torch
import torch.nn as nn

from .utils import get_activation_layer


class FFN(nn.Module):
    def __init__(self, d_model, d_ffn, dropout: float, activation: str = "ReLU"):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        # self.activation = nn.ReLU(inplace=True)
        self.activation = get_activation_layer(activation=activation)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, gate=None):
        tgt2 = self.linear2(
            self.dropout1(
                self.activation(
                    self.linear1(tgt)
                )
            )
        )
        if gate is None:    # without gated control:
            tgt = tgt + self.dropout2(tgt2)
        else:
            tgt = tgt + gate.tanh() * self.dropout2(tgt2)
        tgt = self.norm(tgt)
        return tgt
