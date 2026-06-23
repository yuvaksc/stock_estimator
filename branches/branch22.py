"""Branch 2 - Step 2: align sentiment to a gap-free daily calendar.

Reindexes each stock's daily embeddings onto a continuous day-by-day range that
spans the historical price dates, then FORWARD-fills missing days (weekends,
holidays, silent days) and zero-fills any leading gap. Only past information is
carried forward -- interpolation or backfill would leak a future tweet into an
earlier day. This is what lets branch31/predict align "the prior calendar day"
without ever hitting a missing row.

In/out  stock_data/new_sentiment_embeddings/sentiment_embeddings_<stock>.csv
        (rewritten in place, now one row per calendar day)
"""
import os
import pandas as pd

tweet_embedding_dir = "stock_data/new_sentiment_embeddings"
# Only the date column is read here (for the calendar span), so any of the
# per-stock CSVs works; use the raw source the rest of the pipeline reads.
historical_dir = "stock_data/raw_pca_historical_csv"

for file in os.listdir(tweet_embedding_dir):
    if file.endswith(".csv"):
        file_path = os.path.join(tweet_embedding_dir, file)
        print(f"Processing: {file}")

        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values(by='Date', inplace=True)
        stock_name = file.replace("sentiment_embeddings_", "").replace(".csv", "")
        historical_path = os.path.join(historical_dir, f"{stock_name}.csv")
        if os.path.exists(historical_path):
            historical_dates = pd.read_csv(historical_path, usecols=["Date"])
            historical_dates["Date"] = pd.to_datetime(historical_dates["Date"])
            start_date = historical_dates["Date"].min() - pd.Timedelta(days=1)
            end_date = historical_dates["Date"].max()
        else:
            start_date = df["Date"].min()
            end_date = df["Date"].max()
        full_date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        df_full = pd.DataFrame({'Date': full_date_range})  
        df = df_full.merge(df, on='Date', how='left') 

        # Only carry information forward. Interpolation/backfill would use a
        # future tweet to construct an earlier day's feature.
        embedding_cols = df.columns[1:]
        df[embedding_cols] = df[embedding_cols].ffill().fillna(0.0)

        df.to_csv(file_path, index=False)
        print(f"Forward-fill complete for {file}. No missing dates!")
