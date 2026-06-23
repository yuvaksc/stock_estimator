"""Branch 3 - Step 2: train the pooled multimodal regressor.

Pools the merged tensors of every stock into one train/val set, fits a sentiment
PCA (768 -> k dims, retaining 95% of the TRAIN variance) on the training rows
only, and trains ``PooledMultimodalRegressor`` to predict standardized ``High``
from [stock 128-d | sentiment PCA]. Predictions are unscaled per stock with the
branch11 High mean/scale, so one shared head still recovers each stock's price
level. Early-stops on pooled validation RMSE.

Input   pca_new_merged_tensors/*.pt, pca_best_model1.pth (High mean/scale)
Output  Results/pooled_model.pt
        {state, sentiment_pca_mean, sentiment_pca_components, sentiment_dim,
         stock_feature_dim, val_days, hidden, dropout, ...}
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from _metrics import format_metrics, regression_metrics
from _model import PooledMultimodalRegressor


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
SHOW_PROGRESS = sys.stdout.isatty()

DATA_DIR = "pca_new_merged_tensors"
RESULTS_DIR = "Results"
BRANCH1_CHECKPOINT = "pca_best_model1.pth"
POOLED_MODEL_PATH = os.path.join(RESULTS_DIR, "pooled_model.pt")
TARGET_COL = "High"
SEQ_LEN = 30                 # input window length, matches branch11
SENTIMENT_VARIANCE = 0.95    # PCA components retain this much train variance
os.makedirs(RESULTS_DIR, exist_ok=True)


# Per-stock High mean/scale from the branch11 checkpoint: the single source of
# truth shared with branch33 so standardize/unscale stay perfectly consistent.
if not os.path.exists(BRANCH1_CHECKPOINT):
    raise FileNotFoundError(f"Run branch11.py first; missing {BRANCH1_CHECKPOINT}")
branch1_ckpt = torch.load(BRANCH1_CHECKPOINT, map_location="cpu", weights_only=False)
feature_scalers = branch1_ckpt["feature_scalers"]
high_idx = branch1_ckpt["feature_cols"].index(TARGET_COL)

if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f"Run branch31.py first; missing {DATA_DIR}")
stock_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".pt"))
if not stock_files:
    raise RuntimeError(f"No merged tensors in {DATA_DIR}")

# Per-stock chronological split into pooled train / val. Test is already
# excluded upstream (branch12 exported train+val only).
tr_stock, tr_tweet, tr_ystd, tr_mean, tr_scale = [], [], [], [], []
va_stock, va_tweet, va_ystd = [], [], []
va_mean, va_scale, va_raw, va_prev = [], [], [], []
val_days = 30

for filename in stock_files:
    stock = filename[:-3]
    if stock not in feature_scalers:
        print(f"Skipping {stock}: no branch11 scaler")
        continue
    data = torch.load(os.path.join(DATA_DIR, filename), map_location="cpu", weights_only=False)
    if "previous_target" not in data:
        raise ValueError(f"{stock}: rerun branches 12 and 31")
    features = np.asarray(data["features"], dtype=np.float32)          # (N,128) standardized
    tweets = np.asarray(data["tweet_embeddings"], dtype=np.float32)    # (N,768) standardized
    target = np.asarray(data["target"], dtype=np.float32)              # (N,) raw High series
    previous = np.asarray(data["previous_target"], dtype=np.float32)   # (N,) raw High[t-1]
    val_days = int(data.get("val_days", 30))

    hm = float(feature_scalers[stock]["mean"][high_idx])
    hs = float(feature_scalers[stock]["scale"][high_idx])
    y_std = (target - hm) / hs

    split = len(features) - val_days
    if split <= SEQ_LEN:
        print(f"Skipping {stock}: too few samples")
        continue

    tr_stock.append(features[:split]); tr_tweet.append(tweets[:split]); tr_ystd.append(y_std[:split])
    tr_mean.append(np.full(split, hm, np.float32)); tr_scale.append(np.full(split, hs, np.float32))
    n_val = len(features) - split
    va_stock.append(features[split:]); va_tweet.append(tweets[split:]); va_ystd.append(y_std[split:])
    va_mean.append(np.full(n_val, hm, np.float32)); va_scale.append(np.full(n_val, hs, np.float32))
    va_raw.append(target[split:]); va_prev.append(previous[split:])

# Pool across stocks.
Xs_tr = np.concatenate(tr_stock); Xt_tr_full = np.concatenate(tr_tweet); y_tr = np.concatenate(tr_ystd)
m_tr = np.concatenate(tr_mean); s_tr = np.concatenate(tr_scale)
Xs_va = np.concatenate(va_stock); Xt_va_full = np.concatenate(va_tweet); y_va = np.concatenate(va_ystd)
m_va = np.concatenate(va_mean); s_va = np.concatenate(va_scale)
y_va_raw = np.concatenate(va_raw); y_va_prev = np.concatenate(va_prev)

# Sentiment PCA fit on POOLED TRAIN only (leakage-safe), then applied to both.
pca = PCA(n_components=SENTIMENT_VARIANCE, random_state=SEED).fit(Xt_tr_full)
Xt_tr = pca.transform(Xt_tr_full).astype(np.float32)
Xt_va = pca.transform(Xt_va_full).astype(np.float32)
k = pca.n_components_
print(f"Stocks pooled: {len(va_stock)} | train {len(Xs_tr)}, val {len(Xs_va)} samples")
print(f"Sentiment PCA: 768 -> {k} dims ({SENTIMENT_VARIANCE:.0%} train variance)")


def unscale(pred_std, mean, scale):
    return pred_std * scale + mean


train_loader = DataLoader(
    TensorDataset(
        torch.tensor(Xs_tr), torch.tensor(Xt_tr), torch.tensor(y_tr),
        torch.tensor(m_tr), torch.tensor(s_tr),
    ),
    batch_size=128, shuffle=True,
)
Xs_va_t = torch.tensor(Xs_va, device=device)
Xt_va_t = torch.tensor(Xt_va, device=device)

model = PooledMultimodalRegressor(stock_dim=Xs_tr.shape[1], tweet_dim=k).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

EPOCHS, PATIENCE = 150, 15
best_val_rmse = float("inf")
best_metrics = None
best_state = None
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    sq_err = 0.0
    n = 0
    bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False, disable=not SHOW_PROGRESS)
    for bs, bt, by, bm, bsc in bar:
        bs, bt, by = bs.to(device), bt.to(device), by.to(device)
        bm, bsc = bm.to(device), bsc.to(device)
        optimizer.zero_grad()
        pred = model(bs, bt)
        loss = criterion(pred, by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        with torch.no_grad():
            pr = unscale(pred.detach(), bm, bsc)
            ar = unscale(by, bm, bsc)
            sq_err += torch.sum((pr - ar) ** 2).item()
            n += len(by)

    model.eval()
    with torch.no_grad():
        pred_va_std = model(Xs_va_t, Xt_va_t).cpu().numpy()
    pred_va_raw = unscale(pred_va_std, m_va, s_va)
    train_rmse = math.sqrt(sq_err / n)
    val_rmse = float(np.sqrt(np.mean((pred_va_raw - y_va_raw) ** 2)))
    m = regression_metrics(y_va_raw, pred_va_raw, y_va_prev)
    scheduler.step(val_rmse)

    print(
        f"Epoch {epoch}/{EPOCHS} - Train RMSE: {train_rmse:.4f} | "
        f"Val RMSE: {m['rmse']:.4f}  MAE: {m['mae']:.4f}  "
        f"MAPE: {m['mape']:.2f}%  Dir: {m['directional']:.2f}%"
    )

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_metrics = m
        patience_counter = 0
        best_state = {kk: vv.cpu().clone() for kk, vv in model.state_dict().items()}
        torch.save(
            {
                "state": best_state,
                "sentiment_pca_mean": pca.mean_.astype(np.float32),
                "sentiment_pca_components": pca.components_.astype(np.float32),
                "sentiment_dim": int(k),
                "stock_feature_dim": int(Xs_tr.shape[1]),
                "val_days": int(val_days),
                "hidden": 128,
                "dropout": 0.3,
            },
            POOLED_MODEL_PATH,
        )
        print(f"  saved best pooled model (Val RMSE {val_rmse:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}; best Val RMSE {best_val_rmse:.4f}")
            break

print(f"\nTraining complete (best Val RMSE {best_val_rmse:.4f}). Best validation metrics:")
print(format_metrics(best_metrics))
print(f"Saved {POOLED_MODEL_PATH}")
