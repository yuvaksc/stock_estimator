import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

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


def train_model(features, tweets, target, fname, epochs=700, patience=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalRegressor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    criterion = nn.MSELoss()

    x_stock = torch.tensor(features, dtype=torch.float32).to(device)
    x_tweet = torch.tensor(tweets, dtype=torch.float32).to(device)
    y_true = torch.tensor(target, dtype=torch.float32).to(device)

    split_idx = int(0.8 * len(x_stock))
    train_idx = slice(0, split_idx)
    val_idx = slice(split_idx, len(x_stock))

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        pred_train = model(x_stock[train_idx], x_tweet[train_idx])
        loss = criterion(pred_train, y_true[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(x_stock[val_idx], x_tweet[val_idx])
            val_loss = criterion(pred_val, y_true[val_idx]).item()

        print(f"Epoch {epoch+1}: Train Loss={loss.item():.4f}, Val Loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            torch.save({"state": model.state_dict()}, f'Results/{fname}_model.pt')
        # else:
        #     patience_counter += 1
        #     if patience_counter >= patience:
        #         print(f"⏹️ Early stopping at epoch {epoch+1}")
        #         break

    model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        pred = model(x_stock, x_tweet).cpu().numpy()
        y_true_np = y_true.cpu().numpy()
        mse = mean_squared_error(y_true_np, pred)
        acc = 1 - (mse / np.var(y_true_np))
        print(f"\n✅ Final Accuracy (1 - normalized MSE): {acc:.4f}")

    return model, pred


def plot_predictions(preds, true_vals, min_price, max_price):
    preds_denorm = denormalize(np.array(preds), min_price, max_price)
    true_denorm = denormalize(true_vals.numpy(), min_price, max_price)

    plt.figure(figsize=(12, 6))
    plt.plot(true_denorm, label='Actual')
    plt.plot(preds_denorm, label='Predicted')
    plt.title('📊 Stock High Price Prediction')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()


path = 'new_merged_tensors'
raw = "stock_data/price/raw"
for i, fname in enumerate(os.listdir(path)):
    if not fname.endswith(".pt"):
        continue

    stock_name = fname[:-3]
    print(f"\n=== 📉 Training on stock: {stock_name} ===")
    data = torch.load(os.path.join(path, fname))

    features = data["features"]
    tweets = data["tweet_embeddings"]
    target = torch.tensor(data["target"])

    raw_df = pd.read_csv(os.path.join(raw, f"{stock_name}.csv"))
    target_dates = set([str(d) for d in data["dates"]])
    raw_df["Date"] = pd.to_datetime(raw_df["Date"])
    filtered_df = raw_df[raw_df["Date"].isin(target_dates)].sort_values("Date")

    raw_prices = filtered_df["High"].values

    model, preds = train_model(features=features, tweets=tweets, target=target, fname=stock_name)

    min_price = raw_prices.min()
    max_price = raw_prices.max()
    # plot_predictions(preds, target, min_price, max_price)
    # if i == 3:
    #  break  
