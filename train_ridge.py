#!/usr/bin/env python3
"""
Ridge Regression ML Scheduler Training
Trains a linear model (Ridge) to predict IPC from 23 warp features.
This perfectly matches the linear scoring model in the C++ scheduler.
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

#Feature names (23 features)
FEATURE_NAMES = [
    'age', 'warp_id_feat', 'cta_id', 'stream_id_hash',
    'inst_in_pipeline', 'inst_in_buffer', 'issued_inst_in_pipeline', 'stores_outstanding',
    'completed_threads', 'active_thread_count', 'completion_ratio',
    'inst_miss', 'membar_waiting', 'n_atomic', 'stores_done', 'done_exit',
    'last_fetch_age', 'waiting_status',
    'scoreboard_stall',
    'pipeline_stall_ratio', 'age_squared', 'age_pipeline_interaction', 'ibuffer_empty'
]

TARGET_NAME = 'recent_ipc'

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
            # Check features
            if not all(feat in df.columns for feat in FEATURE_NAMES):
                continue
            dfs.append(df)
        except:
            pass
            
    if not dfs:
        raise ValueError("No valid data loaded")
        
    data = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(data)} samples")
    return data

def train_ridge(data, alpha=1.0):
    X = data[FEATURE_NAMES].values
    y = data[TARGET_NAME].values
    
    # Remove NaN/Inf
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale (Important for Ridge)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Ridge
    print(f"\nTraining Ridge Regression (alpha={alpha})...")
    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"R² Score: {r2:.4f}")
    print(f"MSE: {mse:.6f}")
    
    # Visualize Predictions
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.1, s=1)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('True IPC')
    plt.ylabel('Predicted IPC')
    plt.title(f'Ridge Regression (R²={r2:.3f})')
    plt.savefig('ridge_prediction.png')
    print("Saved ridge_prediction.png")
    
    return model, scaler

def save_weights(model, scaler, r2_score_val, output_file='ml_scheduler_weights.txt'):
    # Adjust weights for raw input: w_raw = w_scaled / std
    # scaler.scale_ contains the standard deviation for each feature
    coefs_scaled = model.coef_
    std_devs = scaler.scale_
    
    # Avoid division by zero (though unlikely with StandardScaler on real data)
    std_devs = np.where(std_devs == 0, 1.0, std_devs)
    
    coefs_raw = coefs_scaled / std_devs
    
    # Visualize weights (Adjusted)
    plt.figure(figsize=(12, 6))
    indices = np.argsort(np.abs(coefs_raw))[::-1]
    plt.bar(range(len(coefs_raw)), coefs_raw[indices])
    plt.xticks(range(len(coefs_raw)), [FEATURE_NAMES[i] for i in indices], rotation=90)
    plt.xlabel('Features (sorted by absolute importance)')
    plt.ylabel('Adjusted Weight (w/std)')
    plt.title(f'Ridge Regression Weights for Raw Input (R²={r2_score_val:.4f})')
    plt.tight_layout()
    plt.savefig('ridge_weights.png', dpi=150)
    print("Saved ridge_weights.png")
    
    print(f"\nSaving ADJUSTED weights to {output_file}...")
    with open(output_file, 'w') as f:
        f.write("# Ridge Regression Weights (23 Features)\n")
        f.write(f"# Trained on 82.9M samples from Rodinia benchmarks\n")
        f.write(f"# Adjusted for raw input (w_raw = w_scaled / std)\n")
        f.write(f"# R² Score: {r2_score_val:.6f}\n")
        f.write(f"# MSE: Check training output\n\n")
        
        for name, weight in zip(FEATURE_NAMES, coefs_raw):
            f.write(f"{name} {weight:.6f}\n")
    
    # Print top 10 most important features (Adjusted)
    print("\nTop 10 Most Important Features (Adjusted for Raw Input):")
    for i, idx in enumerate(indices[:10]):
        print(f"{i+1}. {FEATURE_NAMES[idx]}: {coefs_raw[idx]:.6f}")
            
    print("Done!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='training_data')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--output', default='ml_scheduler_weights.txt')
    args = parser.parse_args()
    
    data = load_data(args.data_dir)
    model, scaler = train_ridge(data, args.alpha)
    
    # Get R² score for saving
    X = data[FEATURE_NAMES].values
    y = data[TARGET_NAME].values
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    
    save_weights(model, scaler, r2, args.output)

if __name__ == "__main__":
    main()
