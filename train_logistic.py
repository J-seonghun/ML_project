#!/usr/bin/env python3
"""
Logistic Regression ML Scheduler Training
Trains a classifier to predict if a warp will issue instructions (be active).
Target: issued_inst_in_pipeline > 0
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# Feature names (23 features matching CSV header)
# We exclude 'issued_inst_in_pipeline' from features because it's derived from the target
FEATURE_NAMES = [
    'age', 'warp_id_feat', 'cta_id', 'stream_id_hash',
    'inst_in_pipeline', 'inst_in_buffer', 'stores_outstanding',
    'completed_threads', 'active_thread_count', 'completion_ratio',
    'inst_miss', 'membar_waiting', 'n_atomic', 'stores_done', 'done_exit',
    'last_fetch_age', 'waiting_status',
    'scoreboard_stall',
    'pipeline_stall_ratio', 'age_squared', 'age_pipeline_interaction', 'ibuffer_empty'
]

# 'issued_inst_in_pipeline' is used to create the target, so we don't use it as input
# We also predict 'recent_ipc' > threshold as an alternative, but let's stick to activity first.

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
            # Check if required columns exist
            # We need FEATURE_NAMES + 'issued_inst_in_pipeline'
            required = FEATURE_NAMES + ['issued_inst_in_pipeline']
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

def train_logistic(data, C=1.0):
    X = data[FEATURE_NAMES].values
    
    # Target: 1 if warp issued instructions, 0 otherwise
    # We use 'issued_inst_in_pipeline' as proxy for "did useful work"
    # Or we can use 'recent_ipc' > 0.1
    
    # Let's use issued_inst_in_pipeline > 0.5 (since it's float, maybe averaged)
    # Actually, let's check what the values look like.
    # Assuming issued_inst_in_pipeline represents number of instructions.
    y = (data['issued_inst_in_pipeline'] > 0).astype(int).values
    
    print(f"Target distribution: Active={np.sum(y==1)}, Idle={np.sum(y==0)}")
    
    # Remove NaN/Inf
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    y = y[mask]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression
    print(f"\nTraining Logistic Regression (C={C})...")
    model = LogisticRegression(C=C, max_iter=1000, solver='lbfgs')
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model, scaler, acc

def save_weights(model, scaler, acc_score, output_file='ml_scheduler_weights.txt'):
    # Adjust weights for raw input: w_raw = w_scaled / std
    coefs_scaled = model.coef_[0] # Logistic regression coef_ is shape (1, n_features)
    std_devs = scaler.scale_
    
    std_devs = np.where(std_devs == 0, 1.0, std_devs)
    coefs_raw = coefs_scaled / std_devs
    
    # Visualize weights
    plt.figure(figsize=(12, 6))
    indices = np.argsort(np.abs(coefs_raw))[::-1]
    plt.bar(range(len(coefs_raw)), coefs_raw[indices])
    plt.xticks(range(len(coefs_raw)), [FEATURE_NAMES[i] for i in indices], rotation=90)
    plt.xlabel('Features (sorted by absolute importance)')
    plt.ylabel('Adjusted Weight (w/std)')
    plt.title(f'Logistic Regression Weights (Acc={acc_score:.4f})')
    plt.tight_layout()
    plt.savefig('logistic_weights.png', dpi=150)
    print("Saved logistic_weights.png")
    
    print(f"\nSaving ADJUSTED weights to {output_file}...")
    with open(output_file, 'w') as f:
        f.write("# Logistic Regression Weights (22 Features)\n")
        f.write(f"# Trained on Rodinia benchmarks\n")
        f.write(f"# Target: issued_inst_in_pipeline > 0\n")
        f.write(f"# Accuracy: {acc_score:.4f}\n\n")
        
        for name, weight in zip(FEATURE_NAMES, coefs_raw):
            f.write(f"{name} {weight:.6f}\n")
            
        # Logistic regression doesn't use issued_inst_in_pipeline as input, 
        # but we need to write 0.0 for it if the C++ expects it in the map?
        # The C++ loads into a map, so missing keys are fine (default 0.0).
        # But let's explicitly write it as 0.0 to be safe/clear.
        f.write("issued_inst_in_pipeline 0.000000\n")
    
    print("\nTop 10 Most Important Features (Adjusted):")
    for i, idx in enumerate(indices[:10]):
        print(f"{i+1}. {FEATURE_NAMES[idx]}: {coefs_raw[idx]:.6f}")
            
    print("Done!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='training_data')
    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--output', default='ml_scheduler_weights.txt')
    args = parser.parse_args()
    
    data = load_data(args.data_dir)
    model, scaler, acc = train_logistic(data, args.C)
    
    save_weights(model, scaler, acc, args.output)

if __name__ == "__main__":
    main()
