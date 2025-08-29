"""
models/transformer_model.py

Small, modular Transformer-based model for behavioral data.
- Can accept time-series input: (batch, seq_len, feat_dim)
- Or static feature vectors: (batch, feat_dim) -> treated as seq_len = n_groups
- Produces a compact embedding vector per example (for fusion with GNN)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeaturePositionalEncoding(nn.Module):
    """
    Simple positional encoding to inject order info when we convert a vector into a sequence.
    This follows the formula from "Attention is All You Need".
    """
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # if odd, avoid shape mismatch
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        seq_groups=None,
        output_dim=64,
    ):
        """
        Args:
            input_dim (int): number of raw behavioral features per timestep or total features.
            d_model (int): transformer hidden dimension.
            nhead (int): number of attention heads.
            num_layers (int): number of TransformerEncoder layers.
            dim_feedforward (int): feedforward layer size inside Transformer.
            dropout (float): dropout.
            seq_groups (int|None): If None and input is (batch, feat), we convert to a sequence
                                   by splitting features into seq_groups tokens (rounded). If input
                                   is (batch, seq_len, feat_dim) pass seq_groups=None.
            output_dim (int): dimension of final embedding returned.
        """
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_groups = seq_groups  # optional grouping when input is 1D per sample

        # input projection: map raw features -> d_model
        self.input_proj = nn.Linear(input_dim, d_model) if self.seq_groups is None else nn.Linear(input_dim // (self.seq_groups or 1), d_model)

        self.pos_encoder = FeaturePositionalEncoding(d_model, max_len=256)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,  # easier: input as (batch, seq, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # pooling into a single vector
        self.pool = nn.AdaptiveAvgPool1d(1)  # pool along seq dim after permute

        # final MLP head
        self.mlp = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

        # small init
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _to_sequence(self, x):
        """
        Convert input to sequence form (batch, seq_len, token_dim)
        - If x is (batch, seq_len, feat_dim): project each timestep
        - If x is (batch, feat_dim): split features into seq_groups tokens
        """
        if x.dim() == 3:
            # time-series input: (batch, seq_len, feat_dim)
            seq = self.input_proj(x)  # projects feat_dim -> d_model
            return seq
        elif x.dim() == 2:
            # static vector per sample: split into tokens
            b, f = x.shape
            if self.seq_groups is None:
                # default split into sqrt-like groups
                g = int(max(1, round(math.sqrt(f))))
            else:
                g = int(self.seq_groups)
            # compute size per token (may not divide evenly)
            token_size = max(1, f // g)
            tokens = []
            for i in range(g):
                start = i * token_size
                end = start + token_size
                if i == g - 1:
                    # include remaining features in the last token
                    chunk = x[:, start:]
                    # if chunk smaller than token_size, pad with zeros
                    if chunk.shape[1] < token_size:
                        pad = torch.zeros(b, token_size - chunk.shape[1], device=x.device)
                        chunk = torch.cat([chunk, pad], dim=1)
                else:
                    chunk = x[:, start:end]
                tokens.append(chunk)
            token_tensor = torch.stack(tokens, dim=1)  # (batch, seq_len=g, token_size)
            seq = self.input_proj(token_tensor)  # (batch, seq_len, d_model)
            return seq
        else:
            raise ValueError("Input tensor must be 2D or 3D")

    def forward(self, x, src_key_padding_mask=None):
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch, feat) or (batch, seq_len, feat_dim)
            src_key_padding_mask: optional mask for padded tokens (batch, seq_len)
        Returns:
            embedding: (batch, output_dim)
        """
        # convert to seq
        seq = self._to_sequence(x)                    # (batch, seq_len, d_model)
        seq = self.pos_encoder(seq)                   # add positional info
        encoded = self.transformer_encoder(seq, src_key_padding_mask=src_key_padding_mask)  # (batch, seq_len, d_model)

        # pool across sequence dimension
        # transform to (batch, d_model, seq_len) for AdaptiveAvgPool1d
        pooled = self.pool(encoded.permute(0, 2, 1)).squeeze(-1)  # (batch, d_model)
        out = self.mlp(pooled)  # (batch, output_dim)
        return out

if __name__ == "__main__":
    # quick smoke test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    # Example A: time-series input (batch, seq_len, feat_dim)
    batch = 4
    seq_len = 6
    feat_dim = 8
    x_ts = torch.randn(batch, seq_len, feat_dim).to(device)
    model_ts = TransformerModel(input_dim=feat_dim, d_model=32, nhead=4, num_layers=2, output_dim=16).to(device)
    emb_ts = model_ts(x_ts)
    print("Time-series input -> embedding shape:", emb_ts.shape)  # expect (batch, 16)

    # Example B: static vector input (batch, feat_dim)
    feat_vec_dim = 16
    x_vec = torch.randn(batch, feat_vec_dim).to(device)
    # choose seq_groups so each token gets 4 features: token_size = feat_vec_dim / seq_groups
    model_vec = TransformerModel(input_dim=feat_vec_dim // 4, d_model=32, nhead=4, num_layers=2, seq_groups=4, output_dim=16).to(device)
    # Note: when seq_groups is provided, input_proj expects token_size = input_dim//seq_groups in constructor,
    # so be careful to pass correct input_dim. Alternatively for simplicity, call with seq_groups=None and the model
    # will choose grouping automatically.
    # For simpler usage, use seq_groups=None and pass raw feature dim:
    model_vec_auto = TransformerModel(input_dim=feat_vec_dim, d_model=32, nhead=4, num_layers=2, seq_groups=None, output_dim=16).to(device)
    emb_vec = model_vec_auto(x_vec)
    print("Vector input -> embedding shape:", emb_vec.shape)  # expect (batch, 16)

