# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, d_model, d_ffn, activation: nn.Module = nn.ReLU(inplace=True)):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = activation
        self.linear2 = nn.Linear(d_ffn, d_model)

    def forward(self, tgt):
        return self.linear2(
            self.activation(
                self.linear1(tgt)
            )
        )
