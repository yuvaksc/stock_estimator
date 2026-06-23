"""Branch 3 - Step 1: merge stock features with sentiment, per stock.

For each stock, joins the branch12 128-d stock vectors with the branch21/22
768-d sentiment of the PRIOR calendar day (only sentiment available before the
target trading day), fits one StandardScaler on the training portion only, and
saves the scaled 896-d vector split back into its 128-d and 768-d halves along
with the raw-dollar targets and the scaler statistics.

Input   stock_data/pca_branch1_tensors/<stock>.pt,
        stock_data/new_sentiment_embeddings/sentiment_embeddings_<stock>.csv
Output  pca_new_merged_tensors/<stock>.pt
        {features (N,128), tweet_embeddings (N,768), target, previous_target,
         scaler_mean, scaler_scale, dates, val_days}
"""
import os
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler


tensor_data_path = "stock_data/pca_branch1_tensors"  
tweet_embeddings_path = "stock_data/new_sentiment_embeddings"  

output_tensor_path = "pca_new_merged_tensors"
os.makedirs(output_tensor_path, exist_ok=True)

stock_files = [f for f in os.listdir(tensor_data_path) if f.endswith(".pt")]

for stock_file in stock_files:
    stock_name = stock_file.replace(".pt", "")
    stock_name = stock_name.replace("branch1_features_", "")
    
    # Load tensor file (Branch 1 output)
    tensor_file_path = os.path.join(tensor_data_path, stock_file)

    tensor_data = torch.load(tensor_file_path) 
    feature_vectors = tensor_data["features"]  # Shape: (num_days, feature_dim)
    dates_tensor = tensor_data["dates"]  # Shape: (num_days, )
    target_tensor = tensor_data["target"]
    if "previous_target" not in tensor_data:
        raise ValueError(f"{stock_name}: rerun branch12.py to add previous-target metadata")
    previous_target_tensor = tensor_data["previous_target"]
    val_days = int(tensor_data.get("val_days", 30))
 

    # Load tweet embeddings CSV (Branch 2 output)
    tweet_file_path = os.path.join(tweet_embeddings_path, f"sentiment_embeddings_{stock_name}.csv")
    
    if not os.path.exists(tweet_file_path):
        print(f"Missing tweet embeddings for {stock_name}, skipping...")
        continue
    
    tweet_df = pd.read_csv(tweet_file_path)
    
    # Ensure the date column is in datetime format
    tweet_df["Date"] = pd.to_datetime(tweet_df["Date"])
    
    
    # Reindex explicitly so stock features, targets and tweets have identical order.
    tweet_df = tweet_df.drop_duplicates("Date", keep="last").set_index("Date")
    # Predict before the target trading day: only sentiment through the prior
    # completed calendar day is available.
    sentiment_dates = pd.to_datetime(dates_tensor) - pd.Timedelta(days=1)
    aligned_tweets = tweet_df.reindex(sentiment_dates)
    if aligned_tweets.isna().any().any():
        raise ValueError(f"Missing tweet embeddings for target dates in {stock_name}")
    tweet_embeddings = aligned_tweets.to_numpy(dtype=np.float32)

    final_features = np.hstack([np.asarray(feature_vectors), tweet_embeddings])

    final_scaler = StandardScaler()
    train_end = len(final_features) - val_days
    if train_end <= 0:
        raise ValueError(f"Not enough samples to reserve {val_days} validation days for {stock_name}")
    final_scaler.fit(final_features[:train_end])
    final_features_scaled = final_scaler.transform(final_features)

    branch1_128_scaled = final_features_scaled[:, :feature_vectors.shape[1]]
    bert_tweets_scaled = final_features_scaled[:, feature_vectors.shape[1]:]

    merged_tensor_data = {
        "features": branch1_128_scaled,  # 128-d Branch-1 stock vectors (scaled)
        "tweet_embeddings": bert_tweets_scaled,  # 768-d sentiment (scaled)
        "target": target_tensor,
        "previous_target": previous_target_tensor,
        "scaler_mean": final_scaler.mean_,
        "scaler_scale": final_scaler.scale_,
        "val_days": val_days,
    }

    
    output_file_path = os.path.join(output_tensor_path, f"{stock_name}.pt")
    torch.save(merged_tensor_data, output_file_path)
    
    print(f"Merged tensor saved for {stock_name}: {output_file_path}")
    

print("Processing complete! All merged tensors saved successfully.")
