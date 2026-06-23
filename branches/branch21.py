"""Branch 2 - Step 1: daily tweet-sentiment embeddings via BERT.

Embeds every cleaned tweet with a frozen ``bert-base-uncased`` (the 768-d [CLS]
vector) and averages the vectors within each calendar day. BERT is frozen and
the aggregation is a plain within-day mean, so nothing is fit across days -- a
future tweet can never influence an earlier day's feature (leakage-safe).

Input   stock_data/processed_tweets/<stock>.csv                  (Date, Tweet)
Output  stock_data/new_sentiment_embeddings/sentiment_embeddings_<stock>.csv
        (Date, 0..767)
"""
import os
import pandas as pd
import numpy as np
import torch
from transformers import BertModel, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tweets_dir = "stock_data/processed_tweets"
output_dir = "stock_data/new_sentiment_embeddings"
os.makedirs(output_dir, exist_ok=True)

# Load BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name).to(device)
bert_model.eval()

def get_tweet_embedding(tweet):
    inputs = tokenizer(tweet, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

def aggregate_embeddings(embeddings_list):
    # BERT is frozen, and a within-day mean requires no corpus-wide fitting.
    # This prevents future/test tweets from affecting training-day features.
    return np.mean(np.stack(embeddings_list), axis=0)

for file in os.listdir(tweets_dir):
    if file.endswith(".csv"):
        stock = file.replace(".csv", "")
        print(f"Processing {stock}...")
        
        df = pd.read_csv(os.path.join(tweets_dir, file))
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        df['embedding'] = df['Tweet'].apply(get_tweet_embedding)
        
        daily_sentiment = df.groupby("Date").apply(
            lambda x: aggregate_embeddings(x["embedding"].tolist())
        ).reset_index(name='embedding')
        
        sentiment_df = pd.DataFrame(daily_sentiment['embedding'].tolist())
        sentiment_df.insert(0, "Date", daily_sentiment["Date"])
        sentiment_df.to_csv(os.path.join(output_dir, f"sentiment_embeddings_{stock}.csv"), index=False)

print("Done! All files processed with leakage-safe daily mean aggregation.")
