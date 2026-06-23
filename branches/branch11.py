"""Branch 1 - Step 1: train the historical-price encoder.

Reads the raw-dollar per-stock CSVs in ``stock_data/raw_pca_historical_csv``
(OHLCV + Movement_PerC + 5 PCA indicator latents) and trains
``SimplifiedHybridStockModel`` (Bi-LSTM + attention) to predict the next day's
*standardized* ``High`` from a 30-day window.

The split is chronological per stock: the last ``TEST_DAYS`` rows are held out
for branch33, the preceding ``VAL_DAYS`` are validation, the rest is training.
StandardScalers are fit on the training period only (no leakage) and stored in
the checkpoint -- they are the single source of truth used everywhere downstream
to standardize inputs and to unscale the predicted ``High`` back to dollars.

Outputs
    pca_best_model1.pth            best-by-val-RMSE weights + per-stock
                                   ``feature_scalers`` + config (feature_cols,
                                   seq_len, val_days, test_days).
    final_trained_stock_model.pth  the same best weights (state_dict only).

Run from the ``stock_estimator/`` directory:  python branches/branch11.py
"""
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from _metrics import format_metrics, regression_metrics
from _model import SimplifiedHybridStockModel


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Keep redirected logs (CI/background runs) clean: tqdm bars only render on a TTY.
SHOW_PROGRESS = sys.stdout.isatty()

DATA_DIR = "stock_data/raw_pca_historical_csv"
CHECKPOINT_PATH = "pca_best_model1.pth"
SEQ_LEN = 30
VAL_DAYS = 30
TEST_DAYS = 30
TARGET_COL = "High"
# The model SEES the target's own history ("High"): the most informative
# predictor of next-day High is recent High, so it is kept as an input column.
FEATURE_COLS = [
    "Movement_PerC", "Open", "High", "Low", "Close", "Volume",
    "latent_1", "latent_2", "latent_3", "latent_4", "latent_5",
]
TARGET_IDX = FEATURE_COLS.index(TARGET_COL)


def load_stock_data(stock):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{stock}.csv"))
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def create_sequences(feature_frame, raw_target):
    """Window the (scaled) feature columns; return raw-unit targets too.

    feature_frame holds the standardized FEATURE_COLS (which now include the
    target's own history). raw_target is the unscaled target series, kept so we
    can report RMSE/R2 in the same price-ratio space as the baselines.
    """
    features = feature_frame.values
    sequences, targets, previous_targets = [], [], []
    for i in range(len(feature_frame) - SEQ_LEN):
        target_index = i + SEQ_LEN
        sequences.append(features[i:target_index])
        targets.append(raw_target[target_index])
        previous_targets.append(raw_target[target_index - 1])
    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(previous_targets, dtype=np.float32),
    )


stocks = sorted(file[:-4] for file in os.listdir(DATA_DIR) if file.endswith(".csv"))
X_train_list, y_train_list, train_mean_list, train_scale_list = [], [], [], []
X_val_list, y_val_list, y_val_previous_list = [], [], []
val_mean_list, val_scale_list = [], []
feature_scalers = {}

for stock in tqdm(stocks, desc="Loading stock data", disable=not SHOW_PROGRESS):
    df = load_stock_data(stock)
    missing = set(FEATURE_COLS + ["Date", TARGET_COL]) - set(df.columns)
    if missing:
        raise ValueError(
            f"{stock}: missing columns {sorted(missing)}. "
            "Rerun pp/mergeLatent.py; partial-stock training is disabled."
        )

    train_end = len(df) - VAL_DAYS - TEST_DAYS
    if train_end <= SEQ_LEN:
        print(f"Skipping {stock}: not enough rows for train/validation/test")
        continue

    # Fit scaling statistics on the training period only (no leakage). Because
    # the target ("High") is part of FEATURE_COLS, the scaler also gives us the
    # per-stock target mean/scale we use to predict in standardized space.
    scaler = StandardScaler().fit(df.iloc[:train_end][FEATURE_COLS])
    scaled = df[FEATURE_COLS].copy()
    scaled.loc[:, FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])
    raw_target = df[TARGET_COL].to_numpy()
    target_mean = float(scaler.mean_[TARGET_IDX])
    target_scale = float(scaler.scale_[TARGET_IDX])
    feature_scalers[stock] = {
        "mean": scaler.mean_,
        "scale": scaler.scale_,
        "feature_cols": FEATURE_COLS,
    }

    train_frame = scaled.iloc[:train_end]
    val_frame = scaled.iloc[train_end - SEQ_LEN:train_end + VAL_DAYS]
    train_target = raw_target[:train_end]
    val_target = raw_target[train_end - SEQ_LEN:train_end + VAL_DAYS]

    X_train, y_train, _ = create_sequences(train_frame, train_target)
    X_val, y_val, y_val_previous = create_sequences(val_frame, val_target)
    if len(y_val) != VAL_DAYS:
        raise RuntimeError(f"{stock}: expected {VAL_DAYS} validation targets, got {len(y_val)}")

    X_train_list.append(X_train)
    y_train_list.append(y_train)
    train_mean_list.append(np.full(len(y_train), target_mean, dtype=np.float32))
    train_scale_list.append(np.full(len(y_train), target_scale, dtype=np.float32))
    X_val_list.append(X_val)
    y_val_list.append(y_val)
    y_val_previous_list.append(y_val_previous)
    val_mean_list.append(np.full(len(y_val), target_mean, dtype=np.float32))
    val_scale_list.append(np.full(len(y_val), target_scale, dtype=np.float32))

if not X_train_list:
    raise RuntimeError(f"No usable CSV files found in {DATA_DIR}")

X_train = np.concatenate(X_train_list)
y_train = np.concatenate(y_train_list)
train_mean = np.concatenate(train_mean_list)
train_scale = np.concatenate(train_scale_list)
X_val = np.concatenate(X_val_list)
y_val = np.concatenate(y_val_list)
y_val_previous = np.concatenate(y_val_previous_list)
val_mean = np.concatenate(val_mean_list)
val_scale = np.concatenate(val_scale_list)

# Train and evaluate in standardized target space, then unscale with the known
# per-stock statistics. This puts the per-stock price level back -- the level
# that per-stock standardization removes from the inputs and that a shared head
# cannot otherwise recover, which is why a raw-target head lost to a moving avg.
y_train_std = (y_train - train_mean) / train_scale

X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train_std_tensor = torch.tensor(y_train_std, dtype=torch.float32, device=device)
train_mean_tensor = torch.tensor(train_mean, dtype=torch.float32, device=device)
train_scale_tensor = torch.tensor(train_scale, dtype=torch.float32, device=device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
val_mean_tensor = torch.tensor(val_mean, dtype=torch.float32, device=device)
val_scale_tensor = torch.tensor(val_scale, dtype=torch.float32, device=device)

train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_std_tensor, train_mean_tensor, train_scale_tensor),
    batch_size=64, shuffle=True,
)
print(f"Training data: {len(X_train_tensor)} samples, {X_train.shape[2]} features")
print(f"Validation data: {len(X_val_tensor)} samples ({VAL_DAYS} per stock)")
print(f"Test data: reserved for branch33 ({TEST_DAYS} targets per stock)")


model = SimplifiedHybridStockModel(input_dim=X_train.shape[2]).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=4
)

EPOCHS = 150
EARLY_STOPPING_PATIENCE = 15
best_val_rmse = float("inf")
best_metrics = None
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_squared_error = 0.0
    train_samples = 0
    progress_bar = tqdm(
        train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False, disable=not SHOW_PROGRESS
    )
    for batch_x, batch_y_std, batch_mean, batch_scale in progress_bar:
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y_std)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # Track training error in raw target units for a like-for-like overfit
        # read against the raw validation RMSE.
        with torch.no_grad():
            pred_raw = predictions.detach() * batch_scale + batch_mean
            actual_raw = batch_y_std * batch_scale + batch_mean
            train_squared_error += torch.sum((pred_raw - actual_raw) ** 2).item()
            train_samples += len(batch_y_std)
        progress_bar.set_postfix(
            rmse=f"{math.sqrt(train_squared_error / train_samples):.4f}"
        )

    model.eval()
    with torch.no_grad():
        val_pred_std = model(X_val_tensor)
        val_pred_raw = (val_pred_std * val_scale_tensor + val_mean_tensor).cpu().numpy()

    train_rmse = math.sqrt(train_squared_error / train_samples)
    val_rmse = float(np.sqrt(np.mean((val_pred_raw - y_val) ** 2)))
    metrics = regression_metrics(y_val, val_pred_raw, y_val_previous)
    scheduler.step(val_rmse)

    print(
        f"Epoch {epoch}/{EPOCHS} - Train RMSE: {train_rmse:.4f} | "
        f"Val RMSE: {metrics['rmse']:.4f}  MAE: {metrics['mae']:.4f}  "
        f"MAPE: {metrics['mape']:.2f}%  Dir: {metrics['directional']:.2f}%"
    )

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_metrics = metrics
        patience_counter = 0
        torch.save(
            {
                "state_dict": model.state_dict(),
                "feature_scalers": feature_scalers,
                "seq_len": SEQ_LEN,
                "val_days": VAL_DAYS,
                "test_days": TEST_DAYS,
                "feature_cols": FEATURE_COLS,
            },
            CHECKPOINT_PATH,
        )
        print(f"  saved best model (Val RMSE {val_rmse:.4f})")
    else:
        patience_counter += 1

    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping at epoch {epoch}; best Val RMSE {best_val_rmse:.4f}")
        break

best_checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(best_checkpoint["state_dict"])
torch.save(model.state_dict(), "final_trained_stock_model.pth")
print(f"\nTraining complete (best Val RMSE {best_val_rmse:.4f}). Best validation metrics:")
print(format_metrics(best_metrics))
print("final_trained_stock_model.pth contains the best checkpoint weights")
