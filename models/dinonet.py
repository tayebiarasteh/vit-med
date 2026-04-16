"""
Created on Aug 26, 2025.
dinonet.py

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh/
"""



import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F


class DinoNet(nn.Module):
    def __init__(self, in_features=4096, hidden=512, out_features=12, dropout=0.3):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden)
        self.hn = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, out_features)

    def forward(self, x):
        x = self.in_norm(x)
        x = F.gelu(self.fc1(x))
        x = self.hn(x)
        x = self.drop(x)

        x = self.out(x)
        return x

