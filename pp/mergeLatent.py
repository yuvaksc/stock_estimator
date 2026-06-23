"""Preprocessing - assemble the final raw-dollar feature CSVs.

Two stages:
  1. From ``stock_data/price/raw`` add ``Movement_PerC = (Close-Open)/Open`` and
     write the base files to ``stock_data/raw_pca_historical_csv``.
  2. Inner-join the 5 PCA indicator latents onto those files (dropping any
     previously merged ``latent_*`` columns first, so reruns are idempotent).

Output  stock_data/raw_pca_historical_csv/<stock>.csv
        (Date, Open, High, Low, Close, Adj Close, Volume, Movement_PerC,
         latent_1..5)  -- the single raw-dollar source every branch reads.
"""
import os
import pandas as pd

# Directories
historical_dir = "stock_data/price/raw"
output_dir = "stock_data/raw_pca_historical_csv"
os.makedirs(output_dir, exist_ok=True)

# Column names for historical data
historical_columns = ["Date", "Movement_PerC", "Open", "High", "Low", "Close", "Volume"]

# Process each CSV file
for file in os.listdir(historical_dir):
    if file.endswith(".csv"):
        stock_id = file.replace(".csv", "")
        csv_path = os.path.join(historical_dir, file)
        output_path = os.path.join(output_dir, f"{stock_id}.csv")

        try:
            # Read CSV file
            df = pd.read_csv(csv_path, parse_dates=["Date"])
            df["Movement_PerC"] = (
                (df["Close"] - df["Open"]) / df["Open"]
            )

            # Save as CSV
            df.to_csv(output_path, index=False)
            print(f"Converted {file} -> {output_path}")

        except Exception as e:
            print(f"Error processing {file}: {e}")

print("TXT to CSV conversion complete!")


# --- Stage 2: merge the PCA indicator latents onto the raw-dollar files ---
# Directories
historical_dir = "stock_data/raw_pca_historical_csv"   
indicators_dir = "stock_data/pca_latent_indicators"   
output_dir = "stock_data/raw_pca_historical_csv"  

# Process each historical file
for hist_file in os.listdir(historical_dir):
    if hist_file.endswith(".csv"):
        stock_id = hist_file.replace(".csv", "")  
        hist_path = os.path.join(historical_dir, hist_file)
        
        indicators_file = f"indicators_{stock_id}.csv"
        indicators_path = os.path.join(indicators_dir, indicators_file)
        
        if os.path.exists(indicators_path):  
            df_hist = pd.read_csv(hist_path, parse_dates=["Date"])
            df_indicators = pd.read_csv(indicators_path, parse_dates=["Date"])

            # Make repeated/interrupted runs safe. Remove any previously merged
            # latent columns (including pandas _x/_y suffixes) before merging.
            old_latent_cols = [
                col for col in df_hist.columns if col.startswith("latent_")
            ]
            if old_latent_cols:
                df_hist.drop(columns=old_latent_cols, inplace=True)

            df_merged = pd.merge(df_hist, df_indicators, on="Date", how="inner")

            merged_path = os.path.join(output_dir, f"{stock_id}.csv")
            df_merged.to_csv(merged_path, index=False)
            print(f"Merged {stock_id} -> {merged_path}")

        else:
            print(f"No indicators found for {stock_id}, skipping...")

print("Merging complete!")
