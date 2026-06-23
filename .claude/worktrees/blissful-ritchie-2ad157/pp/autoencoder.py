import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

input_dir = "pp/indicators"
output_dir = "stock_data/pca_latent_indicators"
os.makedirs(output_dir, exist_ok=True)

# Dimensionality for PCA
LATENT_DIM = 5

for file in os.listdir(input_dir):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(input_dir, file))

        # Set "Date" as index (or keep separately if needed)
        df.set_index("Date", inplace=True)

        # Interpolate and fill missing values
        df.interpolate(method="linear", inplace=True)
        df.fillna(method="bfill", inplace=True)
        df.fillna(method="ffill", inplace=True)

        date_index = df.index  # Save the Date column
        stock_id = file.replace(".csv", "")

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df.values)

        # Apply PCA
        pca = PCA(n_components=LATENT_DIM)
        latent_features = pca.fit_transform(X_scaled)

        # Create latent DataFrame
        latent_df = pd.DataFrame(latent_features, columns=[f"latent_{i+1}" for i in range(LATENT_DIM)])
        latent_df.insert(0, "Date", date_index)

        # Save
        save_path = os.path.join(output_dir, f"{stock_id}.csv")
        latent_df.to_csv(save_path, index=False)
        print(f"Saved PCA latent features for stock: {stock_id} -> {save_path}")
