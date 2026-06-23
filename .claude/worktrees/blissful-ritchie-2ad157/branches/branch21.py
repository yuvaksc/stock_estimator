import os
import pandas as pd
import numpy as np
import torch
from transformers import BertModel, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tweets_dir = "stock_data/processed_tweets"
output_dir = "stock_data/new_sentiment_embeddings"
os.makedirs(output_dir, exist_ok=True)

# Load BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name).to(device)
bert_model.eval()

#  TF-IDF vectorizer 
all_tweets = []
for file in os.listdir(tweets_dir):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(tweets_dir, file))
        all_tweets.extend(df["Tweet"].tolist())

tfidf = TfidfVectorizer(max_features=1000, stop_words="english")
tfidf.fit(all_tweets)  # Learn vocabulary from all tweets

def get_tweet_embedding(tweet):
    inputs = tokenizer(tweet, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

def aggregate_tfidf(tweets, embeddings_list):
    # Compute TF-IDF scores for these tweets => term frequency-inverse document frequency = how relevant a word is
    tfidf_scores = tfidf.transform(tweets)  
    tweet_weights = np.array(tfidf_scores.mean(axis=1)).flatten()  # Average score per tweet
    
    
    weights = torch.softmax(torch.tensor(tweet_weights, dtype=torch.float32), dim=0).to(device)
    
    # Weighted sum of embeddings
    embeddings_tensor = torch.tensor(np.stack(embeddings_list), dtype=torch.float32).to(device)
    pooled = torch.sum(weights.view(-1, 1) * embeddings_tensor, dim=0)
    return pooled.cpu().numpy()

for file in os.listdir(tweets_dir):
    if file.endswith(".csv"):
        stock = file.replace(".csv", "")
        print(f"Processing {stock}...")
        
        df = pd.read_csv(os.path.join(tweets_dir, file))
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        df['embedding'] = df['Tweet'].apply(get_tweet_embedding)
        
        daily_sentiment = df.groupby("Date").apply(
            lambda x: aggregate_tfidf(x["Tweet"].tolist(), x["embedding"].tolist())
        ).reset_index(name='embedding')
        
        sentiment_df = pd.DataFrame(daily_sentiment['embedding'].tolist())
        sentiment_df.insert(0, "Date", daily_sentiment["Date"])
        sentiment_df.to_csv(os.path.join(output_dir, f"sentiment_embeddings_{stock}.csv"), index=False)

print("Done! All files processed with TF-IDF aggregation.")