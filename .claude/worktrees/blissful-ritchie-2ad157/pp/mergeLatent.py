import os
import pandas as pd

# Directories
historical_dir = "stock_data/price/preprocessed"
output_dir = "stock_data/pca_historical_csv"
os.makedirs(output_dir, exist_ok=True)

# Column names for historical data
historical_columns = ["Date", "Movement_PerC", "Open", "High", "Low", "Close", "Volume"]

# Process each TXT file
for file in os.listdir(historical_dir):
    if file.endswith(".txt"):
        stock_id = file.replace(".txt", "")
        txt_path = os.path.join(historical_dir, file)
        csv_path = os.path.join(output_dir, f"{stock_id}.csv")

        try:
            # Read TXT file (no column names, whitespace-separated)
            df = pd.read_csv(txt_path, delimiter=r"\s+", header=None, names=historical_columns, parse_dates=["Date"])

            # Save as CSV
            df.to_csv(csv_path, index=False)
            print(f"✅ Converted {file} -> {csv_path}")

        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

print("🚀 TXT to CSV conversion complete!")



import os
import pandas as pd

# Directories
historical_dir = "stock_data/pca_historical_csv"   
indicators_dir = "stock_data/pca_latent_indicators"   
output_dir = "stock_data/pca_historical_csv"  

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

            df_merged = pd.merge(df_hist, df_indicators, on="Date", how="inner")

            merged_path = os.path.join(output_dir, f"{stock_id}.csv")
            df_merged.to_csv(merged_path, index=False)
            print(f"✅ Merged {stock_id} -> {merged_path}")

        else:
            print(f"No indicators found for {stock_id}, skipping...")

print("Merging complete!")
