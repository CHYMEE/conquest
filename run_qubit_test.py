"""
Step 1: Test Qubit Configurations (2, 3, 4, 8)
Run this script to find the best qubit configuration for QPSO.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from CONQUEST_Journal_Extension import compare_qubit_configurations, SmartSubsampler

# Load dataset
print("Loading dataset...")
df = pd.read_csv('0.ACI-IoT-2023.csv')

# Identify target column
target_col = 'Label' if 'Label' in df.columns else 'label'
print(f"Target column: {target_col}")
print(f"Dataset shape: {df.shape}")

# Prepare features
y = df[target_col].values
X = df.drop(columns=[target_col], errors='ignore')

# Drop non-feature columns
drop_cols = ['Flow ID', 'Timestamp', 'Connection Type']
X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors='ignore')

# Encode categorical columns
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

X = np.nan_to_num(X.values, nan=0, posinf=0, neginf=0)

print(f"Features shape: {X.shape}")
print(f"Classes: {np.unique(y)}")

# Smart subsample for faster testing
print("\nSubsampling for efficiency...")
ss = SmartSubsampler(target_size=30000)
X_sub, y_sub = ss.subsample(X, y)

# Compare qubit configurations
print("\nComparing qubit configurations...")
results = compare_qubit_configurations(X_sub, y_sub, [2, 3, 4, 8])

# Summary
print("\n" + "="*60)
print("QUBIT COMPARISON RESULTS")
print("="*60)
for q, r in results.items():
    print(f"  {q} qubits: Fitness={r['final_fitness']:.4f}, Time={r['computation_time']:.2f}s")

best_q = max(results.keys(), key=lambda q: results[q]['final_fitness'])
print(f"\nRECOMMENDED: {best_q} qubits")
print("="*60)
