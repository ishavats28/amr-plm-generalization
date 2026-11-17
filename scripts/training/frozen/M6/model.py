"""M6: mask-aware two-layer Transformer attention head over frozen ProtBERT residues.

The ProtBERT embeddings are contextual and carry positional information. This
module intentionally matches the original M6 head: no additional position
embeddings, masked mean + learned attention pooling, binary logit output.
"""
import torch
from torch import nn


class ResidueTransformerAttention(nn.Module):
    def __init__(self, input_dim=1024, model_dim=256, heads=8, layers=2,
                 feedforward_dim=512, widths=(256, 64), dropout=0.30,
                 transformer_dropout=0.15):
        super().__init__()
        if model_dim < 1 or heads < 1 or model_dim % heads:
            raise ValueError('model_dim must be divisible by heads')
        if layers < 1 or len(widths) != 2:
            raise ValueError('invalid number of layers or classifier widths')
        self.input_projection = nn.Sequential(nn.Linear(input_dim, model_dim), nn.LayerNorm(model_dim))
        block = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=heads, dim_feedforward=feedforward_dim,
            dropout=transformer_dropout, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(model_dim)
        self.attention_pool = nn.Linear(model_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(2 * model_dim, widths[0]), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(widths[0], widths[1]), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(widths[1], 1),
        )

    def forward(self, embeddings, lengths):
        if embeddings.ndim != 3 or lengths.ndim != 1 or embeddings.shape[0] != lengths.shape[0]:
            raise ValueError('expected [B,L,D] embeddings and [B] lengths')
        if not bool(torch.all(lengths > 0).item()) or not bool(torch.all(lengths <= embeddings.shape[1]).item()):
            raise ValueError('invalid length range')
        valid = torch.arange(embeddings.shape[1], device=embeddings.device)[None, :] < lengths[:, None]
        x = embeddings.masked_fill(~valid.unsqueeze(-1), 0.0)
        x = self.final_norm(self.encoder(self.input_projection(x), src_key_padding_mask=~valid))
        mean = x.masked_fill(~valid.unsqueeze(-1), 0.0).sum(1) / lengths[:, None].to(x.dtype)
        score = self.attention_pool(x).squeeze(-1).masked_fill(~valid, torch.finfo(x.dtype).min)
        attention = torch.softmax(score, dim=1)
        attended = (x * attention.unsqueeze(-1)).sum(1)
        return self.classifier(torch.cat((mean, attended), dim=1)).squeeze(-1)
