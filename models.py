"""
models.py
PyTorch architectures for Convolutional, Recurrent, and Attention-based models
incorporating specific component toggles for ablation studies.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------------
# 0. Baseline: Bag-of-Words Logistic Regression
# -------------------------------------------------------------------------
class BagOfWordsLogReg(nn.Module):
    """Logistic regression on log-scaled word counts, fed the same token ids as the deep models."""

    def __init__(self, vocab_size: int, num_classes: int = 7, pad_idx: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.fc = nn.Linear(vocab_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        counts = torch.zeros(x.size(0), self.vocab_size, device=x.device)
        counts.scatter_add_(1, x, (x != self.pad_idx).float())
        return self.fc(torch.log1p(counts))


# -------------------------------------------------------------------------
# 1. Convolutional Model Family (1D CNN)
# -------------------------------------------------------------------------
class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_classes: int = 7,
        num_filters: int = 100,
        kernel_sizes: tuple = (2, 3, 4),
        dropout: float = 0.5,
        use_batch_norm: bool = True  # Ablation component
    ):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        if self.use_batch_norm:
            self.bns = nn.ModuleList([nn.BatchNorm1d(num_filters) for _ in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L] -> [B, D, L]
        emb = self.embedding(x).permute(0, 2, 1)
        pooled_outputs = []
        for i, conv in enumerate(self.convs):
            h = conv(emb)
            if self.use_batch_norm:
                h = self.bns[i](h)
            h = F.relu(h)
            pooled = F.max_pool1d(h, kernel_size=h.shape[2]).squeeze(2)
            pooled_outputs.append(pooled)
        out = self.dropout(torch.cat(pooled_outputs, dim=1))
        return self.fc(out)


# -------------------------------------------------------------------------
# 2. Recurrent Model Family (LSTM)
# -------------------------------------------------------------------------
class TextRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 7,
        num_layers: int = 1,
        bidirectional: bool = True,  # Ablation component
        dropout: float = 0.5
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        fc_in = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_in, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x != 0).unsqueeze(-1).float()
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        
        # Masked average pooling: pool strictly over real words, discarding <PAD>
        masked_out = out * mask
        sum_out = torch.sum(masked_out, dim=1)
        token_counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = sum_out / token_counts
        
        pooled = self.dropout(pooled)
        return self.fc(pooled)

# -------------------------------------------------------------------------
# 3. Attention-Based Model Family (Transformer Encoder)
# -------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TextTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 256,
        num_layers: int = 2,
        num_classes: int = 7,
        max_len: int = 35,
        dropout: float = 0.3,
        use_positional: bool = True  # Ablation component
    ):
        super().__init__()
        self.d_model = d_model
        self.use_positional = use_positional
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        # Explicitly disable nested tensors to bypass the missing MPS op
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            enable_nested_tensor=False
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mask out padding tokens (index 0) from self-attention
        padding_mask = (x == 0)
        emb = self.embedding(x) * math.sqrt(self.d_model)
        if self.use_positional:
            emb = self.pos_encoder(emb)
        encoded = self.transformer_encoder(emb, src_key_padding_mask=padding_mask)
        # Mean pool non-padding tokens
        mask_expanded = (~padding_mask).unsqueeze(-1).float()
        sum_embeddings = torch.sum(encoded * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_embeddings / sum_mask
        return self.fc(self.dropout(pooled))