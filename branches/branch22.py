import os
import pandas as pd
import numpy as np

tweet_embedding_dir = "stock_data/new_sentiment_embeddings" 

start_date = "2014-01-01"
end_date = "2016-03-31" 
full_date_range = pd.date_range(start=start_date, end=end_date, freq="D")

for file in os.listdir(tweet_embedding_dir):
    if file.endswith(".csv"):
        file_path = os.path.join(tweet_embedding_dir, file)
        print(f"Processing: {file}")

        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values(by='Date', inplace=True)
 
        df_full = pd.DataFrame({'Date': full_date_range})  
        df = df_full.merge(df, on='Date', how='left') 

        # Perform weighted interpolation for missing embeddings
        for idx in df[df.isnull().any(axis=1)].index:
            prev_idx = df[:idx].dropna().index[-1] if not df[:idx].dropna().empty else None
            next_idx = df[idx:].dropna().index[0] if not df[idx:].dropna().empty else None
            
            if prev_idx is not None and next_idx is not None:
                d_prev = (df.loc[idx, 'Date'] - df.loc[prev_idx, 'Date']).days
                d_next = (df.loc[next_idx, 'Date'] - df.loc[idx, 'Date']).days
                
                prev_embedding = df.loc[prev_idx, df.columns[1:]].values  # Skip date column
                next_embedding = df.loc[next_idx, df.columns[1:]].values
                
                interpolated_embedding = (d_next * prev_embedding + d_prev * next_embedding) / (d_prev + d_next)
                df.loc[idx, df.columns[1:]] = interpolated_embedding
            elif prev_idx is not None:
                df.loc[idx, df.columns[1:]] = df.loc[prev_idx, df.columns[1:]]  # Forward fill
            elif next_idx is not None:
                df.loc[idx, df.columns[1:]] = df.loc[next_idx, df.columns[1:]]  # Backward fill

        df.to_csv(file_path, index=False)
        print(f"✅ Interpolation complete for {file}. No missing dates!")
