"""
Step 3: Horizontal Federated Learning with Quantum-Optimized Model
- Uses 3-qubit QPSO for feature selection (from Step 1)
- Uses AdamW optimizer (from Step 2)
- Compares FedAvg, SCAFFOLD-like, and FLAME strategies
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input,
                                      LayerNormalization, MultiHeadAttention)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import resample
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

print("="*70)
print("STEP 3: HORIZONTAL FEDERATED LEARNING")
print("With 3-Qubit QPSO Feature Selection + AdamW Optimizer")
print("="*70)

# ============================================================
# 1. QPSO Feature Selector (3 Qubits - Best from Step 1)
# ============================================================
class QuantumFeatureSelector:
    def __init__(self, num_qubits=3, num_particles=15, iterations=25, num_features=15):
        self.num_qubits = num_qubits
        self.num_particles = num_particles
        self.iterations = iterations
        self.num_features = num_features
        self.best_features = None
        
    def fit(self, X, y):
        n_feat = X.shape[1]
        states = np.abs(np.random.randn(self.num_particles, n_feat))
        states = states / states.sum(axis=1, keepdims=True)
        
        g_best_state = states[0].copy()
        g_best_fitness = 0
        
        for _ in range(self.iterations):
            for i in range(self.num_particles):
                # Quantum rotation
                theta = np.pi / (2 * self.num_qubits)
                states[i] += np.random.rand() * np.cos(theta) * (g_best_state - states[i])
                states[i] = np.abs(states[i])
                states[i] /= states[i].sum()
                
                # Measure and evaluate
                probs = states[i] ** 2
                probs /= probs.sum()
                features = np.random.choice(n_feat, min(self.num_features, n_feat), replace=False, p=probs)
                
                # Fisher criterion fitness
                X_sub = X[:, features]
                fitness = self._fisher_fitness(X_sub, y)
                
                if fitness > g_best_fitness:
                    g_best_fitness = fitness
                    g_best_state = states[i].copy()
        
        probs = g_best_state ** 2
        probs /= probs.sum()
        self.best_features = np.random.choice(n_feat, min(self.num_features, n_feat), replace=False, p=probs)
        print(f"[QPSO-3Q] Selected {len(self.best_features)} features, fitness: {g_best_fitness:.2f}")
        return self
    
    def _fisher_fitness(self, X, y):
        classes = np.unique(y)
        overall_mean = np.mean(X, axis=0)
        between, within = 0, 0
        for c in classes:
            mask = y == c
            if mask.sum() == 0: continue
            c_mean = np.mean(X[mask], axis=0)
            between += mask.sum() * np.sum((c_mean - overall_mean)**2)
            within += np.sum(np.var(X[mask], axis=0))
        return between / (within + 1e-10)
    
    def transform(self, X):
        return X[:, self.best_features]

# ============================================================
# 2. Semantic LSTM Model Builder
# ============================================================
def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = LSTM(128, return_sequences=True, dropout=0.3)(inputs)
    x = LayerNormalization()(x)
    attention = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = x + attention
    x = LayerNormalization()(x)
    x = LSTM(64, return_sequences=False, dropout=0.3)(x)
    x = LayerNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)

# ============================================================
# 3. Horizontal Federated Learning Class
# ============================================================
class HorizontalFederatedLearning:
    """
    Horizontal FL for IoV/IoD with multiple aggregation strategies.
    
    Justification for Horizontal FL:
    1. Data Privacy: Vehicle/drone data stays on-device
    2. Bandwidth: Only model weights transmitted
    3. Compliance: GDPR, data sovereignty
    4. Scalability: Thousands of edge devices
    """
    
    def __init__(self, num_clients=5, strategy='fedavg'):
        self.num_clients = num_clients
        self.strategy = strategy
        self.global_model = None
        self.metrics = []
        
    def create_clients(self, X, y, heterogeneity='iid'):
        n = len(y)
        idx = np.arange(n)
        
        if heterogeneity == 'iid':
            np.random.shuffle(idx)
            splits = np.array_split(idx, self.num_clients)
        else:  # non-iid
            sorted_idx = np.argsort(np.argmax(y, axis=1))
            splits = np.array_split(sorted_idx, self.num_clients)
        
        clients = [(X[s], y[s]) for s in splits]
        
        print(f"\n[FL] Created {self.num_clients} clients ({heterogeneity})")
        for i, (xc, yc) in enumerate(clients):
            classes = np.unique(np.argmax(yc, axis=1))
            print(f"  Client {i}: {len(yc)} samples, {len(classes)} classes")
        
        return clients
    
    def fedavg(self, weights_list, sizes):
        total = sum(sizes)
        avg = []
        for layer_idx in range(len(weights_list[0])):
            layer = np.zeros_like(weights_list[0][layer_idx])
            for i, w in enumerate(weights_list):
                layer += (sizes[i] / total) * w[layer_idx]
            avg.append(layer)
        return avg
    
    def flame(self, weights_list, sizes):
        """Byzantine-robust aggregation using cosine similarity."""
        flat = [np.concatenate([w.flatten() for w in ws]) for ws in weights_list]
        n = len(weights_list)
        
        # Trust scores
        sims = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    sims[i,j] = 1 - cosine(flat[i], flat[j])
        
        trust = np.mean(sims, axis=1)
        threshold = np.percentile(trust, 25)
        trusted = trust >= threshold
        
        print(f"  [FLAME] Trusted clients: {trusted.sum()}/{n}")
        
        trusted_w = [w for w, t in zip(weights_list, trusted) if t]
        trusted_s = [s for s, t in zip(sizes, trusted) if t]
        
        if len(trusted_w) == 0:
            return self.fedavg(weights_list, sizes)
        return self.fedavg(trusted_w, trusted_s)
    
    def scaffold_like(self, weights_list, sizes, prev_weights):
        """Variance-reduced aggregation (simplified SCAFFOLD)."""
        avg = self.fedavg(weights_list, sizes)
        
        # Apply momentum from previous round
        if prev_weights is not None:
            momentum = 0.9
            for i in range(len(avg)):
                avg[i] = momentum * prev_weights[i] + (1 - momentum) * avg[i]
        
        return avg
    
    def train(self, X_train, y_train, X_test, y_test, input_shape, num_classes,
              rounds=10, local_epochs=3, heterogeneity='iid'):
        
        print(f"\n{'='*60}")
        print(f"Federated Learning: {self.strategy.upper()}")
        print(f"Clients: {self.num_clients}, Rounds: {rounds}")
        print('='*60)
        
        clients = self.create_clients(X_train, y_train, heterogeneity)
        
        # Initialize global model with AdamW (best optimizer)
        self.global_model = build_model(input_shape, num_classes)
        self.global_model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        prev_weights = None
        
        for r in range(rounds):
            print(f"\n--- Round {r+1}/{rounds} ---")
            
            client_weights = []
            client_sizes = []
            client_accs = []
            
            for i, (X_c, y_c) in enumerate(clients):
                # Clone global model
                local = keras.models.clone_model(self.global_model)
                local.set_weights(self.global_model.get_weights())
                local.compile(
                    optimizer=keras.optimizers.AdamW(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Local training
                history = local.fit(X_c, y_c, epochs=local_epochs, batch_size=64, verbose=0)
                
                client_weights.append(local.get_weights())
                client_sizes.append(len(y_c))
                client_accs.append(history.history['accuracy'][-1])
            
            print(f"  Client accuracies: {[f'{a:.3f}' for a in client_accs]}")
            
            # Aggregate
            if self.strategy == 'fedavg':
                new_weights = self.fedavg(client_weights, client_sizes)
            elif self.strategy == 'flame':
                new_weights = self.flame(client_weights, client_sizes)
            elif self.strategy == 'scaffold':
                new_weights = self.scaffold_like(client_weights, client_sizes, prev_weights)
            else:
                new_weights = self.fedavg(client_weights, client_sizes)
            
            prev_weights = self.global_model.get_weights()
            self.global_model.set_weights(new_weights)
            
            # Evaluate
            loss, acc = self.global_model.evaluate(X_test, y_test, verbose=0)
            
            # Full metrics
            y_pred = np.argmax(self.global_model.predict(X_test, verbose=0), axis=1)
            y_true = np.argmax(y_test, axis=1)
            prec = precision_score(y_true, y_pred, average='weighted')
            rec = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            
            self.metrics.append({
                'round': r + 1,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
                'loss': loss
            })
            
            print(f"  Global: Acc={acc:.4f}, F1={f1:.4f}, Loss={loss:.4f}")
        
        return self.metrics

# ============================================================
# 4. Load and Prepare Data
# ============================================================
print("\n[1/5] Loading dataset...")
df = pd.read_csv('0.ACI-IoT-2023.csv')
target_col = 'Label' if 'Label' in df.columns else 'label'

y_orig = df[target_col].values
X = df.drop(columns=[target_col], errors='ignore')
drop_cols = ['Flow ID', 'Timestamp', 'Connection Type']
X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors='ignore')

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

X = np.nan_to_num(X.values, nan=0, posinf=0, neginf=0)
print(f"Dataset: {X.shape}")

# ============================================================
# 5. Subsample and Balance
# ============================================================
print("\n[2/5] Balancing dataset...")
classes = np.unique(y_orig)
samples_per_class = 2500
balanced_X, balanced_y = [], []

for c in classes:
    mask = y_orig == c
    Xc, yc = X[mask], y_orig[mask]
    if len(Xc) < samples_per_class:
        Xr, yr = resample(Xc, yc, replace=True, n_samples=samples_per_class, random_state=42)
    else:
        idx = np.random.choice(len(Xc), samples_per_class, replace=False)
        Xr, yr = Xc[idx], yc[idx]
    balanced_X.append(Xr)
    balanced_y.append(yr)

X_bal = np.vstack(balanced_X)
y_bal = np.hstack(balanced_y)
idx = np.random.permutation(len(y_bal))
X_bal, y_bal = X_bal[idx], y_bal[idx]
print(f"Balanced: {X_bal.shape[0]} samples")

# ============================================================
# 6. QPSO Feature Selection (3 Qubits)
# ============================================================
print("\n[3/5] Quantum feature selection (3 qubits)...")
qpso = QuantumFeatureSelector(num_qubits=3, num_features=20)
qpso.fit(X_bal, y_bal)
X_selected = qpso.transform(X_bal)
print(f"Features reduced: {X_bal.shape[1]} -> {X_selected.shape[1]}")

# ============================================================
# 7. Prepare Sequences
# ============================================================
print("\n[4/5] Preparing sequences...")
le = LabelEncoder()
y_enc = le.fit_transform(y_bal)
num_classes = len(np.unique(y_enc))
y_cat = to_categorical(y_enc, num_classes)

n_steps = 10
n_feat = X_selected.shape[1]

def make_sequences(X, y, steps):
    Xs, ys = [], []
    for i in range(len(X) - steps):
        Xs.append(X[i:i+steps])
        ys.append(y[i+steps-1])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = make_sequences(X_selected, y_cat, n_steps)

scaler = StandardScaler()
X_flat = X_seq.reshape(-1, n_feat)
X_scaled = scaler.fit_transform(X_flat).reshape(X_seq.shape)

X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_seq, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

input_shape = (n_steps, n_feat)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ============================================================
# 8. Run Federated Learning Experiments
# ============================================================
print("\n[5/5] Running Federated Learning experiments...")

configs = [
    ('FedAvg-IID', 'fedavg', 'iid'),
    ('FedAvg-NonIID', 'fedavg', 'non_iid'),
    ('SCAFFOLD-NonIID', 'scaffold', 'non_iid'),
    ('FLAME-NonIID', 'flame', 'non_iid'),
]

all_results = {}
start_total = time.time()

for name, strategy, hetero in configs:
    print(f"\n{'#'*60}")
    print(f"# {name}")
    print('#'*60)
    
    fl = HorizontalFederatedLearning(num_clients=5, strategy=strategy)
    start = time.time()
    metrics = fl.train(X_train, y_train, X_test, y_test, input_shape, num_classes,
                       rounds=8, local_epochs=3, heterogeneity=hetero)
    elapsed = time.time() - start
    
    all_results[name] = {
        'metrics': metrics,
        'final_accuracy': metrics[-1]['accuracy'],
        'final_f1': metrics[-1]['f1_score'],
        'time': elapsed
    }

total_time = time.time() - start_total

# ============================================================
# 9. Results Summary
# ============================================================
print("\n" + "="*70)
print("FEDERATED LEARNING RESULTS SUMMARY")
print("="*70)

print(f"\n{'Strategy':<20} {'Final Acc':>12} {'Final F1':>12} {'Time(s)':>10}")
print("-"*56)
for name, r in all_results.items():
    print(f"{name:<20} {r['final_accuracy']:>12.4f} {r['final_f1']:>12.4f} {r['time']:>10.1f}")

best = max(all_results.keys(), key=lambda x: all_results[x]['final_accuracy'])
print(f"\nBEST STRATEGY: {best}")
print(f"Total experiment time: {total_time:.1f}s")

# ============================================================
# 10. Generate Publication Plots
# ============================================================
print("\nGenerating plots...")

plt.rcParams.update({
    'font.size': 12, 'font.weight': 'bold',
    'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
    'figure.dpi': 300
})

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Horizontal Federated Learning for IoV/IoD Intrusion Detection\n(3-Qubit QPSO + AdamW Optimizer)', 
             fontsize=16, fontweight='bold', y=1.02)

colors = {'FedAvg-IID': '#3498DB', 'FedAvg-NonIID': '#E74C3C', 
          'SCAFFOLD-NonIID': '#2ECC71', 'FLAME-NonIID': '#9B59B6'}

# Plot 1: Accuracy convergence
ax1 = axes[0, 0]
for name, r in all_results.items():
    rounds = [m['round'] for m in r['metrics']]
    accs = [m['accuracy'] for m in r['metrics']]
    ax1.plot(rounds, accs, marker='o', linewidth=2.5, markersize=8, 
             color=colors[name], label=name)
ax1.set_xlabel('Federated Round', fontsize=13, fontweight='bold')
ax1.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
ax1.set_title('(a) Accuracy Convergence by Strategy', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0.99, color='red', linestyle='--', alpha=0.5, label='99% Target')

# Plot 2: F1-Score convergence
ax2 = axes[0, 1]
for name, r in all_results.items():
    rounds = [m['round'] for m in r['metrics']]
    f1s = [m['f1_score'] for m in r['metrics']]
    ax2.plot(rounds, f1s, marker='s', linewidth=2.5, markersize=8,
             color=colors[name], label=name)
ax2.set_xlabel('Federated Round', fontsize=13, fontweight='bold')
ax2.set_ylabel('F1-Score', fontsize=13, fontweight='bold')
ax2.set_title('(b) F1-Score Convergence by Strategy', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Final metrics comparison
ax3 = axes[1, 0]
names = list(all_results.keys())
x = np.arange(len(names))
width = 0.35
accs = [all_results[n]['final_accuracy'] for n in names]
f1s = [all_results[n]['final_f1'] for n in names]

bars1 = ax3.bar(x - width/2, accs, width, label='Accuracy', color='#3498DB', edgecolor='black')
bars2 = ax3.bar(x + width/2, f1s, width, label='F1-Score', color='#E74C3C', edgecolor='black')

ax3.set_ylabel('Score', fontsize=13, fontweight='bold')
ax3.set_title('(c) Final Performance Comparison', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(names, rotation=20, ha='right')
ax3.legend(loc='lower right')
ax3.set_ylim([0.8, 1.0])

for bar, val in zip(bars1, accs):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.3f}', ha='center', fontweight='bold', fontsize=9)
for bar, val in zip(bars2, f1s):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.3f}', ha='center', fontweight='bold', fontsize=9)

# Plot 4: Strategy justification table
ax4 = axes[1, 1]
ax4.axis('off')

table_data = [
    ['Strategy', 'Best For', 'Data Type', 'Robustness'],
    ['FedAvg', 'Baseline, simple deployment', 'IID', 'Low'],
    ['SCAFFOLD', 'Client drift reduction', 'Non-IID', 'Medium'],
    ['FLAME', 'Adversarial environments', 'Non-IID', 'High'],
]

table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                  colWidths=[0.2, 0.35, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.5)

for j in range(4):
    table[(0, j)].set_facecolor('#2C3E50')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

ax4.set_title('(d) Strategy Selection Guide for IoV/IoD', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('federated_learning_journal.png', dpi=300, bbox_inches='tight')
plt.savefig('federated_learning_journal.pdf', dpi=300, bbox_inches='tight')
print("Saved: federated_learning_journal.png/pdf")

# ============================================================
# 11. Print Justification
# ============================================================
justification = f"""
================================================================================
HORIZONTAL FEDERATED LEARNING JUSTIFICATION FOR IoV/IoD
================================================================================

1. WHY HORIZONTAL FEDERATED LEARNING?

   For Internet of Vehicles (IoV):
   - Vehicles generate sensitive driving/location data
   - Data cannot leave the vehicle due to privacy regulations
   - V2X communication has limited bandwidth
   - Thousands of vehicles need to contribute to model training
   
   For Internet of Drones (IoD):
   - Drones collect sensitive surveillance/mission data
   - Military/commercial drones have strict data policies
   - Swarm learning requires distributed training
   - Edge deployment on resource-constrained devices

2. EXPERIMENTAL CONFIGURATION:
   - Feature Selection: 3-Qubit QPSO (fastest, optimal fitness)
   - Optimizer: AdamW (highest accuracy from Step 2)
   - Clients: 5 (simulating 5 edge devices/vehicles/drones)
   - Rounds: 8 federated rounds
   - Local Epochs: 3 per round

3. RESULTS:
   Best Strategy: {best}
   - Accuracy: {all_results[best]['final_accuracy']:.4f}
   - F1-Score: {all_results[best]['final_f1']:.4f}

4. STRATEGY RECOMMENDATIONS:

   FedAvg (IID Data):
   - Use when: Data is uniformly distributed across clients
   - IoV Example: Similar traffic patterns across all vehicles
   - IoD Example: Homogeneous drone swarm in same environment
   
   SCAFFOLD (Non-IID Data):
   - Use when: Clients have different data distributions
   - IoV Example: Urban vs rural vehicles see different attacks
   - IoD Example: Surveillance vs delivery drones
   - Benefit: Reduces client drift, faster convergence
   
   FLAME (Adversarial):
   - Use when: Some clients may be compromised
   - IoV Example: Hacked vehicles sending malicious updates
   - IoD Example: Captured drones in hostile territory
   - Benefit: Byzantine-robust, filters malicious updates

5. PRIVACY GUARANTEES:
   - Raw data NEVER leaves the client device
   - Only model weights are transmitted
   - Aggregation happens at central server
   - Compliant with GDPR, CCPA, and military data policies

================================================================================
"""

print(justification)

with open('federated_justification.txt', 'w') as f:
    f.write(justification)
print("Saved: federated_justification.txt")

# Save results
import json
with open('federated_results.json', 'w') as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'metrics'} 
               for k, v in all_results.items()}, f, indent=2)
print("Saved: federated_results.json")
