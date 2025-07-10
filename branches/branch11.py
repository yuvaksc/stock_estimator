import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler


device = "cuda"
print(f"Using device: {device}")

merged_data_path = "stock_data/pca_historical_csv"


# Define Sliding Window Length & Test Period
SEQ_LEN = 30
TEST_DAYS = 90  # Last 3 months for final testing
VAL_RATIO = 0.15  # Use 15% of training data for validation

def load_stock_data(stock):
    df = pd.read_csv(os.path.join(merged_data_path, f"{stock}.csv"))
    df.sort_values(by='Date', inplace=True)
    df = df.drop(columns=['Date'])
    return df

def create_sequences(data, target_col="High"):
    targ = data['High']
    data = data.drop(columns=['High'])
    sequences, targets = [], []
    for i in range(len(data) - SEQ_LEN):
        seq = data.iloc[i:i+SEQ_LEN].values
        target = targ.iloc[i+SEQ_LEN]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

stocks = [file.replace(".csv", "") for file in os.listdir(merged_data_path) if file.endswith(".csv")]

X_train_list, y_train_list = [], []
X_test_list, y_test_list = [], []

for stock in tqdm(stocks, desc="Loading Stock Data"):
    df = load_stock_data(stock)

    feature_cols = ['Movement_PerC', 'Open', 'Low', 'Close', 'Volume', 
                'latent_1', 'latent_2', 'latent_3', 'latent_4', 'latent_5']

    scaler = StandardScaler()

    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    train_data = df_scaled.iloc[:-TEST_DAYS]  # Exclude last 3 months
    test_data = df_scaled.iloc[-TEST_DAYS - SEQ_LEN:]  # Keep last 3 months + sequence buffer

    # Generate Sequences
    X_train, y_train = create_sequences(train_data)
    X_test, y_test = create_sequences(test_data)

    X_train_list.append(X_train)
    y_train_list.append(y_train)
    X_test_list.append(X_test)
    y_test_list.append(y_test)

X_train = np.concatenate(X_train_list)
y_train = np.concatenate(y_train_list)
X_test = np.concatenate(X_test_list)
y_test = np.concatenate(y_test_list)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# **Split Training Data into Training & Validation**
total_samples = X_train_tensor.shape[0]
val_size = int(total_samples * VAL_RATIO)
train_size = total_samples - val_size

train_dataset, val_dataset = random_split(TensorDataset(X_train_tensor, y_train_tensor), [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

print(f"✅ Training Data: {train_size} samples")
print(f"✅ Validation Data: {val_size} samples")
print(f"✅ Testing Data: {X_test_tensor.shape[0]} samples")

# Define Model

class SimplifiedHybridStockModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=1, num_heads=4, dropout_rate=0.25, output_feature_dim=128, use_residual=False):
        super(SimplifiedHybridStockModel, self).__init__()
        
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
        final = self.out(output_vector).squeeze(-1)
        return final


# Initialize Model
input_dim = X_train.shape[2]
model = SimplifiedHybridStockModel(input_dim=input_dim).to(device)
checkpoint = 'best_model.pth'
#model.load_state_dict(checkpoint['state_dict'])

# Optimizer & Loss Function
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
#optimizer.load_state_dict(checkpoint['optimizer'])
criterion = nn.MSELoss()

# Learning Rate Scheduler & Early Stopping
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True)

EARLY_STOPPING_PATIENCE = 15
best_val_loss = float('inf')
patience_counter = 0
import math

EPOCHS = 150
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
    
    for batch_x, batch_y in progress_bar:
        optimizer.zero_grad()
        feature_vector = model(batch_x)
        price_pred = feature_vector
        loss = criterion(price_pred, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += math.sqrt(loss.item())
        progress_bar.set_postfix(loss=f"{train_loss / len(train_loader):.4f}")
    
    # **Validation Step**
    model.eval()
    val_loss = 0
    val_r2 = 0
    correct_trend = 0
    total_samples = 0
    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_feature_vector = model(val_x)
            val_price_pred = val_feature_vector    
            val_loss += criterion(val_price_pred, val_y).item()
            val_r2 += r2_score(val_y.cpu().numpy(), val_price_pred.cpu().numpy())  # R² Score

            predicted_trend = torch.sign(val_price_pred[1:] - val_price_pred[:-1])
            actual_trend = torch.sign(val_y[1:] - val_y[:-1])
            correct_trend += (predicted_trend == actual_trend).sum().item()
            total_samples += len(actual_trend)
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    val_r2 /= len(val_loader)
    val_directional_acc = correct_trend / total_samples * 100  # Percentage


    scheduler.step(val_loss)

    print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"✅ Val R² Score: {val_r2:.4f}")
    print(f"✅ Val Directional Accuracy: {val_directional_acc:.2f}%")


    # **Early Stopping Logic**
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0  # Reset patience counter
        torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, "pca_best_model1.pth")
        print(f"✅ Best model saved with Val Loss: {val_loss:.4f}")
    else:
        patience_counter += 1

    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"⏹️ Early Stopping Triggered at Epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
        break

# Final Model Save
torch.save(model.state_dict(), "final_trained_stock_model.pth")
print("✅ Model Training Complete & Saved!")
