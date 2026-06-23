"""Branch 1 - Step 2: export the encoder's 128-d feature vectors.

Loads the encoder trained by branch11 and runs ``.encode()`` over every 30-day
window of the raw-dollar CSVs, turning each target day into a 128-d stock-feature
vector. The final ``test_days`` targets are excluded so the test split stays
unseen until branch33.

IMPORTANT: this must read the SAME source branch11 trained on
(``raw_pca_historical_csv``). Pointing it at the pre-normalized
``pca_historical_csv`` applies the raw scaler to already-normalized values and
silently corrupts every encoding.

Input   pca_best_model1.pth, stock_data/raw_pca_historical_csv/*.csv
Output  stock_data/pca_branch1_tensors/<stock>.pt
        {features (N,128), dates, target, previous_target, val_days}
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from _model import SimplifiedHybridStockModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SHOW_PROGRESS = sys.stdout.isatty()

merged_data_path = "stock_data/raw_pca_historical_csv"  # Raw-price CSVs: branch11 fit its scalers here, so branch12 must read the SAME source or the (mean,scale) standardization is applied to the wrong units.
branch1_tensor_dir = "stock_data/pca_branch1_tensors"           # Folder to save Branch 1 tensor files
os.makedirs(branch1_tensor_dir, exist_ok=True)

SEQ_LEN = 30
TARGET_COL = "High"
# Fallback only; the real feature list/order comes from the branch11 checkpoint.
DEFAULT_FEATURE_COLS = [
    "Movement_PerC", "Open", "High", "Low", "Close", "Volume",
    "latent_1", "latent_2", "latent_3", "latent_4", "latent_5",
]


def load_stock_data_with_date(stock):
    df = pd.read_csv(os.path.join(merged_data_path, f"{stock}.csv"))
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(by="Date", inplace=True)
    return df.reset_index(drop=True)


def create_sequences_with_date(feature_frame, raw_target, dates):
    """Window scaled features; keep raw target/previous_target for downstream."""
    features = feature_frame.values
    sequences, seq_dates, targets, previous_targets = [], [], [], []
    for i in range(len(feature_frame) - SEQ_LEN):
        j = i + SEQ_LEN
        sequences.append(features[i:j])
        seq_dates.append(dates[j])
        targets.append(raw_target[j])
        previous_targets.append(raw_target[j - 1])
    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(seq_dates),
        np.asarray(targets, dtype=np.float32),
        np.asarray(previous_targets, dtype=np.float32),
    )


CHECKPOINT_PATH = "pca_best_model1.pth"
if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Run branch11.py first; missing {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
if "feature_scalers" not in checkpoint:
    raise ValueError("Existing checkpoint predates leakage fix; rerun branch11.py")
if checkpoint.get("seq_len", SEQ_LEN) != SEQ_LEN:
    raise ValueError("Sequence length differs from the branch11 checkpoint")

feature_cols = checkpoint.get("feature_cols", DEFAULT_FEATURE_COLS)
feature_scalers = checkpoint["feature_scalers"]
TEST_DAYS = checkpoint.get("test_days", 30)
output_feature_dim = 128

model = SimplifiedHybridStockModel(
    input_dim=len(feature_cols), output_feature_dim=output_feature_dim
).to(device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
print(f"Loaded Branch 1 model from {CHECKPOINT_PATH} ({len(feature_cols)} features)")

# Process each stock and save the Branch 1 feature vectors as tensor files
stocks = [file.replace(".csv", "") for file in os.listdir(merged_data_path) if file.endswith(".csv")]
for stock in tqdm(stocks, desc="Saving Branch 1 Tensors", disable=not SHOW_PROGRESS):
    if stock not in feature_scalers:
        print(f"Skipping {stock}: scaler not found in checkpoint")
        continue

    df = load_stock_data_with_date(stock)
    scaler_state = feature_scalers[stock]
    # Cast to float first: Volume is int64, and assigning standardized floats into
    # an int column is a pandas FutureWarning that will eventually truncate (zeroing
    # the Volume feature). Casting up front keeps the standardized values intact.
    scaled = df[feature_cols].astype("float64")
    scaled.loc[:, feature_cols] = (
        df[feature_cols].values - np.asarray(scaler_state["mean"])
    ) / np.asarray(scaler_state["scale"])
    raw_target = df[TARGET_COL].to_numpy()
    dates = df["Date"].to_numpy()

    # Export train + validation; leave the final test targets untouched.
    keep = len(df) - TEST_DAYS
    X_seq, seq_dates, target, previous_target = create_sequences_with_date(
        scaled.iloc[:keep], raw_target[:keep], dates[:keep]
    )

    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    with torch.no_grad():
        feature_vectors = model.encode(X_tensor).cpu()  # (num_samples, output_feature_dim)

    stock_features = {
        "features": feature_vectors,
        "dates": seq_dates,
        "target": target,
        "previous_target": previous_target,
        "val_days": checkpoint.get("val_days", 30),
    }
    save_path = os.path.join(branch1_tensor_dir, f"{stock}.pt")
    torch.save(stock_features, save_path)

print(f"Saved Branch 1 tensors for {len(stocks)} stocks to {branch1_tensor_dir}")
