import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

merged_data_path = "stock_data/pca_historical_csv"  # Folder with merged CSVs (must include 'Date' column)
branch1_tensor_dir = "stock_data/pca_branch1_tensors"           # Folder to save Branch 1 tensor files
os.makedirs(branch1_tensor_dir, exist_ok=True)

SEQ_LEN = 30

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
        # out = lstm_out[:, :, :lstm_out.size(2)//2]
        # out = lstm_out[:, -1, :]

        residual_out = self.residual(lstm_out)
        out = residual_out
        out = self.norm(out)

        attn_output, _ = self.attention(out, out, out)
        feature_vector = torch.mean(attn_output, dim=1)
        feature_vector = self.dropout(feature_vector) 
        
        output_vector = self.fc(feature_vector)
        return output_vector




input_dim = 10        # Example: 5 historical + 5 latent features; 
output_feature_dim = 128    # Feature vector dimension

model = SimplifiedHybridStockModel(input_dim=input_dim, output_feature_dim=output_feature_dim).to(device)
CHECKPOINT_PATH = "pca_best_model1.pth"
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"✅ Loaded Branch 1 model from {CHECKPOINT_PATH}")
model.eval()

# Process each stock and save the Branch 1 feature vectors as tensor files
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

    train_data = df_scaled.iloc[:-90]
    X_seq, dates, target = create_sequences_with_date(train_data, target_col="High")

    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        feature_vectors = model(X_tensor).cpu()  # shape: (num_samples, output_feature_dim)
    
    
    stock_features = {
        "features": feature_vectors,  # Tensor of shape (num_samples, output_feature_dim)
        "dates": dates,
        "target": target
    }
    
    save_path = os.path.join(branch1_tensor_dir, f"{stock}.pt")
    torch.save(stock_features, save_path)
    print(f"Saved Branch 1 tensors for {stock} to {save_path}")
