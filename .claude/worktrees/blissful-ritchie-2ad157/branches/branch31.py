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
 

    tensor_df = pd.DataFrame({"Date": dates_tensor, "Feature_Vector": list(feature_vectors)})
    # Load tweet embeddings CSV (Branch 2 output)
    tweet_file_path = os.path.join(tweet_embeddings_path, f"sentiment_embeddings_{stock_name}.csv")
    
    if not os.path.exists(tweet_file_path):
        print(f"⚠️ Missing tweet embeddings for {stock_name}, skipping...")
        continue
    
    tweet_df = pd.read_csv(tweet_file_path)
    
    # Ensure the date column is in datetime format
    tweet_df["Date"] = pd.to_datetime(tweet_df["Date"])
    
    
    filtered_train_tweet_df = tweet_df[tweet_df["Date"].isin(tensor_df["Date"])]
    tweet_embeddings = [
        torch.tensor(row[1:].astype(float).values, dtype=torch.float32)
        for _, row in filtered_train_tweet_df.iterrows()
    ]
    tweet_embeddings = torch.stack(tweet_embeddings)
    # print(tweet_embeddings.shape, feature_vectors.shape, dates_tensor.shape, target_tensor.shape)

    merge_scaler = StandardScaler()
    final_features = np.hstack([feature_vectors, tweet_embeddings])

    final_scaler = StandardScaler()
    final_features_scaled = final_scaler.fit_transform(final_features)

    branch1_128_scaled = final_features_scaled[:, :feature_vectors.shape[1]]
    bert_tweets_scaled = final_features_scaled[:, feature_vectors.shape[1]:]

    print(branch1_128_scaled.shape, bert_tweets_scaled.shape)

    merged_tensor_data = {
        "dates": dates_tensor,  # Retain original dates
        "features": branch1_128_scaled,  # Retain original feature vectors
        "tweet_embeddings": bert_tweets_scaled,  # Add tweet embeddings
        "target": target_tensor
    }

    
    output_file_path = os.path.join(output_tensor_path, f"{stock_name}.pt")
    torch.save(merged_tensor_data, output_file_path)
    
    print(f"✅ Merged tensor saved for {stock_name}: {output_file_path}")
    

print("🎯 Processing complete! All merged tensors saved successfully.")
