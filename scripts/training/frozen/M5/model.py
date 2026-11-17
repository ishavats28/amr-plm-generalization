"""Masked multiscale 1D CNN for residue-level frozen ProtBERT embeddings."""
import torch
from torch import nn
from torch.nn import functional as F


class MultiScaleResidueCNN(nn.Module):
    def __init__(self, input_dim=1024, kernels=(3, 5, 7), channels=128,
                 widths=(256, 64), dropout=0.30):
        super().__init__()
        if not kernels or any(k < 1 or k % 2 == 0 for k in kernels):
            raise ValueError("Convolution kernels must be positive odd integers")
        if len(widths) != 2:
            raise ValueError("Expected two classifier hidden widths")
        self.branches = nn.ModuleList([
            nn.Conv1d(input_dim, channels, kernel_size=k, padding=k // 2)
            for k in kernels
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(2 * len(kernels) * channels, widths[0]),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(widths[0], widths[1]),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(widths[1], 1),
        )

    def forward(self, embeddings, lengths):
        if embeddings.ndim != 3 or lengths.ndim != 1:
            raise ValueError("Require [B,L,D] embeddings and [B] lengths")
        if embeddings.shape[0] != lengths.shape[0]:
            raise ValueError("Batch/length mismatch")
        if not bool((lengths > 0).all()) or not bool((lengths <= embeddings.shape[1]).all()):
            raise ValueError("Invalid residue lengths")
        valid = torch.arange(embeddings.shape[1], device=embeddings.device)[None, :] < lengths[:, None]
        features = (embeddings * valid.unsqueeze(-1)).transpose(1, 2)
        conv_mask = valid.unsqueeze(1)
        summaries = []
        for layer in self.branches:
            activated = self.dropout(F.gelu(layer(features)))
            mean = activated.masked_fill(~conv_mask, 0.0).sum(-1) / lengths[:, None].to(activated.dtype)
            maximum = activated.masked_fill(~conv_mask, -torch.inf).amax(-1)
            summaries.extend((mean, maximum))
        return self.classifier(torch.cat(summaries, 1)).squeeze(-1)
