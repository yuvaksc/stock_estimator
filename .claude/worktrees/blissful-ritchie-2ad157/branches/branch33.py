import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

merged_data_path = "stock_data/historical_csv"  
branch1_tensor_dir = "test_tensors"           
os.makedirs(branch1_tensor_dir, exist_ok=True)

class SimplifiedHybridStockModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=1, num_heads=4, dropout_rate=0.25, output_feature_dim=128, use_residual=False):
        super(SimplifiedHybridStockModel, self).__init__()
        self.use_residual = use_residual
        
        self.fusion = nn.Linear(input_dim, input_dim)
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                              batch_first=True, bidirectional=True, dropout=dropout_rate)
        
        self.residual = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
        
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_feature_dim)
        self.out = nn.Linear(output_feature_dim, 1)
    
    def forward(self, x):
        x = F.relu(self.fusion(x))
        lstm_out, _ = self.bilstm(x)

        residual_out = self.residual(lstm_out)
        out = residual_out
        out = self.norm(out)

        attn_output, _ = self.attention(out, out, out)
        feature_vector = torch.mean(attn_output, dim=1)
        feature_vector = self.dropout(feature_vector) 
        
        output_vector = self.fc(feature_vector)
        return output_vector


def load_stock_data_with_date(stock):
    df = pd.read_csv(os.path.join(merged_data_path, f"{stock}.csv"))
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)
    return df

def create_sequences_with_date(data, target_col="High"):
    sequences = []
    dates = []
    targets = []
    for i in range(len(data) - SEQ_LEN):
        seq = data.iloc[i:i+SEQ_LEN].drop(columns=['Date', 'High']).values
        target_date = data.iloc[i+SEQ_LEN]['Date']  # Date corresponding to the prediction
        target = data.iloc[i+SEQ_LEN][target_col]
        sequences.append(seq)
        dates.append(target_date)
        targets.append(target)
    return np.array(sequences), np.array(dates), np.array(targets)

SEQ_LEN = 30
input_dim = 10      
output_feature_dim = 128   
model = SimplifiedHybridStockModel(input_dim=input_dim, output_feature_dim=output_feature_dim).to(device)
CHECKPOINT_PATH = "best_model1.pth"
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"✅ Loaded Branch 1 model from {CHECKPOINT_PATH}")
model.eval()


stocks = [file.replace(".csv", "") for file in os.listdir(merged_data_path) if file.endswith(".csv")]
final_targets = []
final_features = []
final_dates = []
for stock in tqdm(stocks, desc="Saving Branch 1 Tensors"):
    df = load_stock_data_with_date(stock)
    feature_cols = ['Movement_PerC', 'Open', 'Low', 'Close', 'Volume', 
                'latent_1', 'latent_2', 'latent_3', 'latent_4', 'latent_5']

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])

    train_data = df_scaled.iloc[-90:]
    X_seq, dates, target = create_sequences_with_date(train_data, target_col="High")

    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        feature_vectors = model(X_tensor).cpu()  
    
    
    stock_features = {
        "features": feature_vectors,  
        "dates": dates,
        "target": target
    }
    
    save_path = os.path.join(branch1_tensor_dir, f"{stock}.pt")
    torch.save(stock_features, save_path)
    print(f"Saved Branch 1 tensors for {stock} to {save_path}")













tensor_data_path = "test_tensors"  
tweet_embeddings_path = "stock_data/new_sentiment_embeddings" 

output_tensor_path = "test_merged_tensors"
os.makedirs(output_tensor_path, exist_ok=True)

stock_files = [f for f in os.listdir(tensor_data_path) if f.endswith(".pt")]

for stock_file in stock_files:
    stock_name = stock_file.replace(".pt", "")
    stock_name = stock_name.replace("branch1_features_", "")
    tensor_file_path = os.path.join(tensor_data_path, stock_file)

    tensor_data = torch.load(tensor_file_path)  
    feature_vectors = tensor_data["features"] 
    dates_tensor = tensor_data["dates"]  
    target_tensor = tensor_data["target"]
 
    tensor_df = pd.DataFrame({"Date": dates_tensor, "Feature_Vector": list(feature_vectors)})
    tweet_file_path = os.path.join(tweet_embeddings_path, f"sentiment_embeddings_{stock_name}.csv")
    
    if not os.path.exists(tweet_file_path):
        print(f"⚠️ Missing tweet embeddings for {stock_name}, skipping...")
        continue
    
    tweet_df = pd.read_csv(tweet_file_path)
    
    tweet_df["Date"] = pd.to_datetime(tweet_df["Date"])
    
    filtered_train_tweet_df = tweet_df[tweet_df["Date"].isin(tensor_df["Date"])]

    tweet_embeddings = [
        torch.tensor(row[1:].astype(float).values, dtype=torch.float32)
        for _, row in filtered_train_tweet_df.iterrows()
    ]
    tweet_embeddings = torch.stack(tweet_embeddings)
    final_features = np.hstack([feature_vectors, tweet_embeddings])

    final_scaler = StandardScaler()
    final_features_scaled = final_scaler.fit_transform(final_features)

    branch1_128_scaled = final_features_scaled[:, :feature_vectors.shape[1]]
    bert_tweets_scaled = final_features_scaled[:, feature_vectors.shape[1]:]

    print(branch1_128_scaled.shape, bert_tweets_scaled.shape)

    merged_tensor_data = {
        "dates": dates_tensor,  
        "features": branch1_128_scaled,  
        "tweet_embeddings": bert_tweets_scaled, 
        "target": target_tensor
    }

    
    output_file_path = os.path.join(output_tensor_path, f"{stock_name}.pt")
    torch.save(merged_tensor_data, output_file_path)
    
    print(f"✅ Merged tensor saved for {stock_name}: {output_file_path}")



class MultimodalRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.stock_proj = nn.Linear(128, 256)
        self.tweet_proj = nn.Linear(768, 256)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=4, dim_feedforward=512, dropout=0.2, batch_first=True
        )
        self.stock_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.tweet_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.fusion_layer = nn.Linear(2 * 256, 256) # For concatenating Transformer outputs
        self.layer_norm = nn.LayerNorm(256)
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        self.dropout_proj = nn.Dropout(0.3)

    def forward(self, x_stock, x_tweet):
        x1 = self.dropout_proj(self.stock_proj(x_stock)).unsqueeze(1) # (batch, 1, 256)
        x1_transformed = self.stock_transformer(x1).squeeze(1)     # (batch, 256)

        x2 = self.dropout_proj(self.tweet_proj(x_tweet)).unsqueeze(1) # (batch, 1, 256)
        x2_transformed = self.tweet_transformer(x2).squeeze(1)     # (batch, 256)

        x = torch.cat([x1_transformed, x2_transformed], dim=-1) # (batch, 512)
        x = self.fusion_layer(x)
        x = self.layer_norm(x)
        return self.regressor(x).squeeze()
   


def denormalize(norm_vals, min_price, max_price):
    return norm_vals * (max_price - min_price) + min_price



def plot_predictions(preds, true_vals, min_price, max_price, title):
    preds_denorm = denormalize(np.array(preds), min_price, max_price)
    true_denorm = denormalize(true_vals.numpy(), min_price, max_price)
    
    print(regression_accuracy(true_denorm, preds_denorm))

    plt.plot(true_denorm, label='Actual')
    plt.plot(preds_denorm, label='Predicted')
    plt.title(title)
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()



def test_model(features, tweets, target):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalRegressor().to(device)
    

    x_stock = torch.tensor(features, dtype=torch.float32).to(device)
    x_tweet = torch.tensor(tweets, dtype=torch.float32).to(device)
    y_true = torch.tensor(target, dtype=torch.float32).to(device)

    path = 'Results/AAPL_model.pt'
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state'])

    # Final Accuracy
    model.eval()
    with torch.no_grad():
        pred = model(x_stock, x_tweet).cpu().numpy()

    return model, pred





def regression_accuracy(y_true, y_pred, tolerance=0.01):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    relative_error = np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8)
    accurate_predictions = (relative_error <= tolerance).sum()
    accuracy = accurate_predictions / len(y_true)
    return accuracy * 100


import numpy as np



# fig, axes = plt.subplots(2, 2, figsize=(15, 10))
# axes = axes.flatten()  # Flatten the 2x2 array of axes for easy indexing


path = 'test_merged_tensors'
count = 0
for i, fname in enumerate(os.listdir(path)):
    if not fname.endswith(".pt"):
        continue

    stock_name = fname[:-3]

    if not stock_name == 'AAPL' and not stock_name == 'JPM' and not stock_name == "ORCL":
        continue 
    raw = f"stock_data/price/raw/{stock_name}.csv"
    print(f"\n=== 📉 Training on stock: {stock_name} ===")
    data = torch.load(os.path.join(path, fname))

    features = data["features"]
    tweets = data["tweet_embeddings"]
    target = torch.tensor(data["target"])

    raw_df = pd.read_csv(raw)
    target_dates = set([str(d) for d in data["dates"]])
    raw_df["Date"] = pd.to_datetime(raw_df["Date"])
    filtered_df = raw_df[raw_df["Date"].isin(target_dates)].sort_values("Date")

    raw_prices = filtered_df["High"].values
    
    model, preds = test_model(features=features, tweets=tweets, target=target)
    min_price = raw_prices.min()
    max_price = raw_prices.max()
    plot_predictions(preds, target, min_price, max_price, f'Stock: {stock_name}')
    # count+=1
    
    # if count == 4:
    #     break


# plt.tight_layout()
# plt.show()