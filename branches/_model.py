"""Canonical model definitions shared across branches.

These classes are reconstructed from saved ``state_dict`` checkpoints in several
branch scripts, so a single source of truth keeps every definition byte-for-byte
identical -- a drift here would make ``load_state_dict`` fail or silently load the
wrong weights. Import from here instead of redefining:

    from _model import SimplifiedHybridStockModel   # branch11/12/33
    from _model import PooledMultimodalRegressor     # branch32/33
"""
import torch
import torch.nn as nn


class SimplifiedHybridStockModel(nn.Module):
    """Branch-1 encoder. ``encode()`` returns the 128-d feature vector consumed
    downstream; ``forward()`` adds the scalar (standardized-High) head used only
    while training branch11."""

    def __init__(self, input_dim, hidden_dim=128, num_layers=1, num_heads=4,
                 dropout_rate=0.3, output_feature_dim=128):
        super().__init__()
        # GELU (not ReLU) so the projection keeps negative standardized values
        # instead of zeroing roughly half of every input.
        self.fusion = nn.Linear(input_dim, input_dim)
        self.activation = nn.GELU()
        self.bilstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        self.residual = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout_rate, batch_first=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        # Pool both the sequence mean (context) and the last step (recency) so a
        # next-day forecast is not blurred by averaging the whole window.
        self.fc = nn.Linear(hidden_dim * 2, output_feature_dim)
        self.out = nn.Linear(output_feature_dim, 1)

    def encode(self, x):
        x = self.activation(self.fusion(x))
        lstm_out, _ = self.bilstm(x)
        out = self.norm(self.residual(lstm_out))
        attention_output, _ = self.attention(out, out, out)
        pooled = torch.cat(
            [attention_output.mean(dim=1), attention_output[:, -1, :]], dim=-1
        )
        return self.fc(self.dropout(pooled))

    def forward(self, x):
        return self.out(self.encode(x)).squeeze(-1)


class PooledMultimodalRegressor(nn.Module):
    """One model for all stocks. Predicts standardized High; the caller unscales
    with each stock's High mean/scale, which is how the per-stock price level is
    restored (the features are per-stock standardized, so a shared head cannot
    recover the level on its own -- the same fix used in branch11)."""

    def __init__(self, stock_dim=128, tweet_dim=32, hidden=128, dropout=0.3):
        super().__init__()
        self.stock_proj = nn.Sequential(
            nn.Linear(stock_dim, 128), nn.GELU(), nn.Dropout(dropout)
        )
        self.tweet_proj = nn.Sequential(
            nn.Linear(tweet_dim, 64), nn.GELU(), nn.Dropout(dropout)
        )
        self.fuse = nn.Sequential(
            nn.Linear(128 + 64, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, x_stock, x_tweet):
        fused = torch.cat([self.stock_proj(x_stock), self.tweet_proj(x_tweet)], dim=-1)
        return self.fuse(fused).squeeze(-1)
