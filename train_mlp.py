#!/usr/bin/env python3
"""
MLP (Neural Network) ML Scheduler Training
Trains a small MLP to predict warp priority score.
Architecture: 22 -> 16 -> 8 -> 1
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Feature names (22 features, excluding issued_inst_in_pipeline from input)
FEATURE_NAMES = [
    'age', 'warp_id_feat', 'cta_id', 'stream_id_hash',
    'inst_in_pipeline', 'inst_in_buffer', 'stores_outstanding',
    'completed_threads', 'active_thread_count', 'completion_ratio',
    'inst_miss', 'membar_waiting', 'n_atomic', 'stores_done', 'done_exit',
    'last_fetch_age', 'waiting_status',
    'scoreboard_stall',
    'pipeline_stall_ratio', 'age_squared', 'age_pipeline_interaction', 'ibuffer_empty'
]

class MLPScheduler(nn.Module):
    def __init__(self, input_size=22, hidden1=16, hidden2=8):
        super(MLPScheduler, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

def load_data(data_dir):
    pattern = os.path.join(data_dir, "*.csv")
    files = glob.glob(pattern)
    
    if not files:
        raise ValueError(f"No CSV files found in {data_dir}")
        
    print(f"Found {len(files)} CSV files in {data_dir}")
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            required = FEATURE_NAMES + ['recent_ipc']
            if not all(col in df.columns for col in required):
                continue
            dfs.append(df)
        except:
            pass
            
    if not dfs:
        raise ValueError("No valid data loaded")
        
    data = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(data)} samples")
    return data

def train_mlp(data, epochs=10, batch_size=4096, lr=0.001):
    X = data[FEATURE_NAMES].values
    y = data['recent_ipc'].values.reshape(-1, 1)
    
    # Remove NaN/Inf
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y).ravel()
    X = X[mask]
    y = y[mask]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test_scaled)
    y_test_t = torch.FloatTensor(y_test)
    
    # DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    model = MLPScheduler(input_size=len(FEATURE_NAMES))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\nTraining MLP (22 -> 16 -> 8 -> 1) for {epochs} epochs...")
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_t).numpy()
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
        
        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, R²={r2:.4f}, MSE={mse:.4f}")
    
    return model, scaler, r2

def save_mlp_weights(model, scaler, r2_score_val, output_file='mlp_scheduler_weights.txt'):
    """
    Save MLP weights in a format compatible with C++.
    We need to save:
    - Scaler parameters (mean, std) for input normalization
    - fc1 weights and bias
    - fc2 weights and bias
    - fc3 weights and bias
    """
    print(f"\nSaving MLP weights to {output_file}...")
    
    with open(output_file, 'w') as f:
        f.write("# MLP Scheduler Weights\n")
        f.write(f"# Architecture: 22 -> 16 -> 8 -> 1\n")
        f.write(f"# R² Score: {r2_score_val:.6f}\n\n")
        
        # Save scaler parameters
        f.write("# Scaler (mean and std for input normalization)\n")
        f.write("scaler_mean ")
        f.write(" ".join(f"{m:.6f}" for m in scaler.mean_))
        f.write("\n")
        f.write("scaler_std ")
        f.write(" ".join(f"{s:.6f}" for s in scaler.scale_))
        f.write("\n\n")
        
        # Save Layer 1 (fc1): 22 -> 16
        fc1_weight = model.fc1.weight.data.numpy()  # Shape: (16, 22)
        fc1_bias = model.fc1.bias.data.numpy()      # Shape: (16,)
        f.write("# Layer 1 (fc1): 22 -> 16\n")
        f.write(f"fc1_weight {fc1_weight.shape[0]} {fc1_weight.shape[1]}\n")
        for row in fc1_weight:
            f.write(" ".join(f"{w:.6f}" for w in row))
            f.write("\n")
        f.write("fc1_bias ")
        f.write(" ".join(f"{b:.6f}" for b in fc1_bias))
        f.write("\n\n")
        
        # Save Layer 2 (fc2): 16 -> 8
        fc2_weight = model.fc2.weight.data.numpy()  # Shape: (8, 16)
        fc2_bias = model.fc2.bias.data.numpy()      # Shape: (8,)
        f.write("# Layer 2 (fc2): 16 -> 8\n")
        f.write(f"fc2_weight {fc2_weight.shape[0]} {fc2_weight.shape[1]}\n")
        for row in fc2_weight:
            f.write(" ".join(f"{w:.6f}" for w in row))
            f.write("\n")
        f.write("fc2_bias ")
        f.write(" ".join(f"{b:.6f}" for b in fc2_bias))
        f.write("\n\n")
        
        # Save Layer 3 (fc3): 8 -> 1
        fc3_weight = model.fc3.weight.data.numpy()  # Shape: (1, 8)
        fc3_bias = model.fc3.bias.data.numpy()      # Shape: (1,)
        f.write("# Layer 3 (fc3): 8 -> 1\n")
        f.write(f"fc3_weight {fc3_weight.shape[0]} {fc3_weight.shape[1]}\n")
        for row in fc3_weight:
            f.write(" ".join(f"{w:.6f}" for w in row))
            f.write("\n")
        f.write("fc3_bias ")
        f.write(" ".join(f"{b:.6f}" for b in fc3_bias))
        f.write("\n")
    
    print(f"Saved MLP weights to {output_file}")
    print("Done!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='training_data')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output', default='mlp_scheduler_weights.txt')
    args = parser.parse_args()
    
    data = load_data(args.data_dir)
    model, scaler, r2 = train_mlp(data, args.epochs, args.batch_size, args.lr)
    
    save_mlp_weights(model, scaler, r2, args.output)

if __name__ == "__main__":
    main()
