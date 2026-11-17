"""M7: bidirectional LSTM then Transformer/attention over frozen ProtBERT tokens.

Inputs: padded [B,L,1024] precomputed residue features and true lengths.
Padding is excluded from recurrent states, attention keys, and pooled summaries.
Only this classification head is trainable; no ProtBERT weights are loaded here.
"""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ResidueBiLSTMTransformer(nn.Module):
    """Projection -> packed BiLSTM -> Transformer -> masked pooling -> MLP."""

    def __init__(
        self,
        input_dim: int = 1024,
        model_dim: int = 256,
        lstm_hidden_per_direction: int = 128,
        lstm_layers: int = 1,
        heads: int = 8,
        transformer_layers: int = 2,
        feedforward_dim: int = 512,
        head_widths=(256, 64),
        dropout: float = 0.30,
        transformer_dropout: float = 0.15,
        lstm_output_dropout: float = 0.15,
    ):
        super().__init__()
        if model_dim <= 0 or model_dim % heads:
            raise ValueError('model_dim must be positive and divisible by heads')
        if 2 * lstm_hidden_per_direction != model_dim:
            raise ValueError('Two LSTM directions must concatenate to model_dim')
        if lstm_layers < 1 or transformer_layers < 1 or len(head_widths) != 2:
            raise ValueError('Expected positive layer counts and two MLP widths')
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.bilstm = nn.LSTM(
            input_size=model_dim,
            hidden_size=lstm_hidden_per_direction,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.lstm_norm = nn.LayerNorm(model_dim)
        self.lstm_dropout = nn.Dropout(lstm_output_dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=transformer_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.attention_pool = nn.Linear(model_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(2 * model_dim, head_widths[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_widths[0], head_widths[1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_widths[1], 1),
        )

    def forward(self, embeddings: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3 or lengths.ndim != 1:
            raise ValueError('Expected embeddings [B,L,D], lengths [B]')
        if embeddings.shape[0] != lengths.shape[0]:
            raise ValueError('Batch dimensions do not match')
        if not bool(torch.all(lengths > 0).item()) or not bool(torch.all(lengths <= embeddings.shape[1]).item()):
            raise ValueError('All sequence lengths must lie within padded width')

        B, L, _ = embeddings.shape
        mask = torch.arange(L, device=embeddings.device)[None, :] < lengths[:, None]
        x = embeddings.masked_fill(~mask.unsqueeze(-1), 0.)
        x = self.input_projection(x)
        # Packing prevents backward LSTM states from seeing right padding.
        packed = pack_padded_sequence(
            x, lengths.detach().to('cpu'), batch_first=True, enforce_sorted=False,
        )
        h, _ = self.bilstm(packed)
        x, unpacked_lengths = pad_packed_sequence(
            h, batch_first=True, total_length=L,
        )
        if not bool(torch.equal(unpacked_lengths.to(lengths.device), lengths)):
            raise RuntimeError('Packed BiLSTM sequence lengths do not match')
        x = self.lstm_dropout(self.lstm_norm(x))
        x = x.masked_fill(~mask.unsqueeze(-1), 0.)
        x = self.encoder(x, src_key_padding_mask=~mask)
        x = self.final_norm(x)

        pooled_mean = x.masked_fill(~mask.unsqueeze(-1), 0.).sum(dim=1) / lengths.unsqueeze(-1).to(x.dtype)
        logits = self.attention_pool(x).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        attended = (x * weights.unsqueeze(-1)).sum(dim=1)
        return self.classifier(torch.cat([pooled_mean, attended], dim=-1)).squeeze(-1)
