"""Preprocessing - reduce the indicator block to 5 PCA latents.

Despite the filename this uses PCA, not an autoencoder. Standardizes each stock's
indicator table and projects it down to ``LATENT_DIM`` components. Both the
scaler and the PCA are fit on the TRAINING period only (excluding the last
VAL_DAYS + TEST_DAYS), so no test-period information leaks into the latents.

Input   pp/indicators/indicators_<stock>.csv
Output  stock_data/pca_latent_indicators/<stock>.csv   (Date, latent_1..5)
Feeds   pp/mergeLatent.py
"""
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

input_dir = "pp/indicators"
output_dir = "stock_data/pca_latent_indicators"
os.makedirs(output_dir, exist_ok=True)

# Dimensionality for PCA
LATENT_DIM = 5
VAL_DAYS = 30
TEST_DAYS = 30

for file in os.listdir(input_dir):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(input_dir, file))
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)

        # Set "Date" as index (or keep separately if needed)
        df.set_index("Date", inplace=True)

        # Do not interpolate/backfill from future dates.
        df.ffill(inplace=True)
        df.fillna(0.0, inplace=True)

        date_index = df.index  # Save the Date column
        stock_id = file.replace(".csv", "")

        # Standardize features
        train_end = len(df) - VAL_DAYS - TEST_DAYS
        if train_end <= LATENT_DIM:
            print(f"Skipping {file}: not enough training rows")
            continue
        scaler = StandardScaler().fit(df.iloc[:train_end].values)
        X_scaled = scaler.transform(df.values)

        # Apply PCA
        pca = PCA(n_components=LATENT_DIM).fit(X_scaled[:train_end])
        latent_features = pca.transform(X_scaled)

        # Create latent DataFrame
        latent_df = pd.DataFrame(latent_features, columns=[f"latent_{i+1}" for i in range(LATENT_DIM)])
        latent_df.insert(0, "Date", date_index)

        # Save
        save_path = os.path.join(output_dir, f"{stock_id}.csv")
        latent_df.to_csv(save_path, index=False)
        print(f"Saved PCA latent features for stock: {stock_id} -> {save_path}")
