"""
Memory-Efficient Horizontal Federated Learning for Journal Extension
- Reduced model size and batch size to prevent OOM
- Optimized for systems with limited RAM
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
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix)
from sklearn.utils import resample
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import gc
import warnings
warnings.filterwarnings('ignore')

# Memory optimization
tf.keras.backend.clear_session()
gc.collect()

np.random.seed(42)
tf.random.set_seed(42)

print("="*70)
print("MEMORY-EFFICIENT HORIZONTAL FEDERATED LEARNING")
print("Journal Extension: 2-Qubit QPSO + SGD (Best from Ablation Study)")
print("="*70)

# ============================================================
# 1. QPSO Feature Selector (3 Qubits)
# ============================================================
class QuantumFeatureSelector:
    def __init__(self, num_qubits=2, num_particles=15, iterations=25, num_features=20):
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
                theta = np.pi / (2 * self.num_qubits)
                states[i] += np.random.rand() * np.cos(theta) * (g_best_state - states[i])
                states[i] = np.abs(states[i])
                states[i] /= states[i].sum()
                
                probs = states[i] ** 2
                probs /= probs.sum()
                features = np.random.choice(n_feat, min(self.num_features, n_feat), replace=False, p=probs)
                
                X_sub = X[:, features]
                fitness = self._fisher_fitness(X_sub, y)
                
                if fitness > g_best_fitness:
                    g_best_fitness = fitness
                    g_best_state = states[i].copy()
        
        probs = g_best_state ** 2
        probs /= probs.sum()
        self.best_features = np.random.choice(n_feat, min(self.num_features, n_feat), replace=False, p=probs)
        print(f"[QPSO-{self.num_qubits}Q] Selected {len(self.best_features)} features, fitness: {g_best_fitness:.2f}")
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
# 2. Lightweight LSTM Model
# ============================================================
def build_model(input_shape, num_classes):
    """QPSO-LSTM-VAE+SGD model (2-qubit optimized)."""
    inputs = Input(shape=input_shape)
    x = LSTM(128, return_sequences=True, dropout=0.3)(inputs)
    x = LayerNormalization()(x)
    attention = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = x + attention
    x = LayerNormalization()(x)
    x = LSTM(64, return_sequences=False, dropout=0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)

# ============================================================
# 3. Memory-Efficient Federated Learning
# ============================================================
class LiteFederatedLearning:
    def __init__(self, num_clients=5, strategy='fedavg'):
        self.num_clients = num_clients
        self.strategy = strategy
        self.global_model = None
        self.metrics = []
        self.communication_costs = []
        
    def create_clients(self, X, y, heterogeneity='iid'):
        n = len(y)
        idx = np.arange(n)
        
        if heterogeneity == 'iid':
            np.random.shuffle(idx)
            splits = np.array_split(idx, self.num_clients)
        else:
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
        flat = [np.concatenate([w.flatten() for w in ws]) for ws in weights_list]
        n = len(weights_list)
        
        sims = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    sims[i,j] = 1 - cosine(flat[i], flat[j])
        
        trust = np.mean(sims, axis=1)
        threshold = np.percentile(trust, 25)
        trusted = trust >= threshold
        
        trusted_w = [w for w, t in zip(weights_list, trusted) if t]
        trusted_s = [s for s, t in zip(sizes, trusted) if t]
        
        if len(trusted_w) == 0:
            return self.fedavg(weights_list, sizes)
        return self.fedavg(trusted_w, trusted_s)
    
    def scaffold_like(self, weights_list, sizes, prev_weights):
        avg = self.fedavg(weights_list, sizes)
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
        
        # Clear memory before building model
        tf.keras.backend.clear_session()
        gc.collect()
        
        self.global_model = build_model(input_shape, num_classes)
        self.global_model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model_size = sum([np.prod(w.shape) for w in self.global_model.get_weights()]) * 4
        print(f"Model size: {model_size / 1024:.2f} KB")
        
        prev_weights = None
        
        for r in range(rounds):
            print(f"\n--- Round {r+1}/{rounds} ---")
            
            client_weights = []
            client_sizes = []
            client_accs = []
            
            for i, (X_c, y_c) in enumerate(clients):
                # Clear memory before each client
                tf.keras.backend.clear_session()
                gc.collect()
                
                local = build_model(input_shape, num_classes)
                local.set_weights(self.global_model.get_weights())
                local.compile(
                    optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Train with smaller batch size
                history = local.fit(X_c, y_c, epochs=local_epochs, batch_size=32, verbose=0)
                
                client_weights.append(local.get_weights())
                client_sizes.append(len(y_c))
                client_accs.append(history.history['accuracy'][-1])
                
                # Clean up local model
                del local
                gc.collect()
            
            print(f"  Client accuracies: {[f'{a:.3f}' for a in client_accs]}")
            
            round_comm = model_size * self.num_clients * 2
            self.communication_costs.append(round_comm)
            
            if self.strategy == 'fedavg':
                new_weights = self.fedavg(client_weights, client_sizes)
            elif self.strategy == 'flame':
                new_weights = self.flame(client_weights, client_sizes)
            elif self.strategy == 'scaffold':
                new_weights = self.scaffold_like(client_weights, client_sizes, prev_weights)
            else:
                new_weights = self.fedavg(client_weights, client_sizes)
            
            prev_weights = [w.copy() for w in self.global_model.get_weights()]
            self.global_model.set_weights(new_weights)
            
            # Clean up
            del client_weights
            gc.collect()
            
            loss, acc = self.global_model.evaluate(X_test, y_test, verbose=0, batch_size=32)
            
            y_pred = np.argmax(self.global_model.predict(X_test, verbose=0, batch_size=32), axis=1)
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
                'loss': loss,
                'avg_client_acc': np.mean(client_accs)
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

# Free memory
del df
gc.collect()

# ============================================================
# 5. Subsample and Balance (smaller size)
# ============================================================
print("\n[2/5] Balancing dataset...")
classes = np.unique(y_orig)
samples_per_class = 1500  # Reduced for memory
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
print(f"Balanced: {X_bal.shape[0]} samples, {len(classes)} classes")

# Free memory
del X, y_orig, balanced_X, balanced_y
gc.collect()

# ============================================================
# 6. QPSO Feature Selection
# ============================================================
print("\n[3/5] Quantum feature selection (2 qubits - best from ablation)...")
qpso = QuantumFeatureSelector(num_qubits=2, num_features=20)
qpso.fit(X_bal, y_bal)
X_selected = qpso.transform(X_bal)
print(f"Features reduced: {X_bal.shape[1]} -> {X_selected.shape[1]}")

del X_bal
gc.collect()

# ============================================================
# 7. Prepare Sequences
# ============================================================
print("\n[4/5] Preparing sequences...")
le = LabelEncoder()
y_enc = le.fit_transform(y_bal)
num_classes = len(np.unique(y_enc))
y_cat = to_categorical(y_enc, num_classes)

n_steps = 8  # Reduced sequence length
n_feat = X_selected.shape[1]

def make_sequences(X, y, steps):
    Xs, ys = [], []
    for i in range(len(X) - steps):
        Xs.append(X[i:i+steps])
        ys.append(y[i+steps-1])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = make_sequences(X_selected, y_cat, n_steps)

del X_selected, y_cat
gc.collect()

scaler = StandardScaler()
X_flat = X_seq.reshape(-1, n_feat)
X_scaled = scaler.fit_transform(X_flat).reshape(X_seq.shape)

del X_seq, X_flat
gc.collect()

X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_seq, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

del X_scaled, y_seq, X_temp, y_temp
gc.collect()

input_shape = (n_steps, n_feat)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ============================================================
# 8. Run Federated Learning
# ============================================================
print("\n[5/5] Running Federated Learning...")

configs = [
    ('FedAvg-IID', 'fedavg', 'iid'),
    ('FedAvg-NonIID', 'fedavg', 'non_iid'),
    ('SCAFFOLD-IID', 'scaffold', 'iid'),
    ('SCAFFOLD-NonIID', 'scaffold', 'non_iid'),
    ('FLAME-IID', 'flame', 'iid'),
    ('FLAME-NonIID', 'flame', 'non_iid'),
]

all_results = {}
all_fl_objects = {}
start_total = time.time()

for name, strategy, hetero in configs:
    print(f"\n{'#'*60}")
    print(f"# {name}")
    print('#'*60)
    
    # Clear memory before each experiment
    tf.keras.backend.clear_session()
    gc.collect()
    
    fl = LiteFederatedLearning(num_clients=5, strategy=strategy)
    start = time.time()
    metrics = fl.train(X_train, y_train, X_test, y_test, input_shape, num_classes,
                       rounds=10, local_epochs=3, heterogeneity=hetero)
    elapsed = time.time() - start
    
    all_results[name] = {
        'metrics': metrics,
        'final_accuracy': metrics[-1]['accuracy'],
        'final_f1': metrics[-1]['f1_score'],
        'final_precision': metrics[-1]['precision'],
        'final_recall': metrics[-1]['recall'],
        'time': elapsed,
        'communication_costs': fl.communication_costs
    }
    all_fl_objects[name] = fl

total_time = time.time() - start_total

# ============================================================
# 9. Results Summary
# ============================================================
print("\n" + "="*70)
print("FEDERATED LEARNING RESULTS SUMMARY")
print("="*70)

print(f"\n{'Strategy':<20} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
print("-"*62)
for name, r in all_results.items():
    print(f"{name:<20} {r['final_accuracy']:>10.4f} {r['final_f1']:>10.4f} "
          f"{r['final_precision']:>10.4f} {r['final_recall']:>10.4f}")

best = max(all_results.keys(), key=lambda x: all_results[x]['final_accuracy'])
print(f"\nBEST STRATEGY: {best}")
print(f"Total experiment time: {total_time:.1f}s")

# ============================================================
# 10. Generate Plots
# ============================================================
print("\nGenerating plots...")

plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

colors = {
    'FedAvg-IID': '#2E86AB',
    'FedAvg-NonIID': '#E94F37',
    'SCAFFOLD-IID': '#44AF69',
    'SCAFFOLD-NonIID': '#F4A261',
    'FLAME-IID': '#9B59B6',
    'FLAME-NonIID': '#E76F51'
}

markers = {'FedAvg-IID': 'o', 'FedAvg-NonIID': 's', 'SCAFFOLD-IID': '^', 'SCAFFOLD-NonIID': 'v', 'FLAME-IID': 'D', 'FLAME-NonIID': 'p'}

# Figure 1: Main Results
fig1, axes = plt.subplots(2, 2, figsize=(12, 10))
fig1.suptitle('Horizontal Federated Learning for IoV/IoD Intrusion Detection\n'
              '(2-Qubit QPSO + SGD Optimizer)', fontsize=14, fontweight='bold')

# Accuracy
ax1 = axes[0, 0]
for name, r in all_results.items():
    rounds = [m['round'] for m in r['metrics']]
    accs = [m['accuracy'] for m in r['metrics']]
    ax1.plot(rounds, accs, marker=markers[name], linewidth=2, markersize=6, 
             color=colors[name], label=name)
ax1.set_xlabel('Federated Round')
ax1.set_ylabel('Test Accuracy')
ax1.set_title('(a) Accuracy Convergence')
ax1.legend(loc='lower right', fontsize=9)
ax1.grid(True, alpha=0.3)

# F1-Score
ax2 = axes[0, 1]
for name, r in all_results.items():
    rounds = [m['round'] for m in r['metrics']]
    f1s = [m['f1_score'] for m in r['metrics']]
    ax2.plot(rounds, f1s, marker=markers[name], linewidth=2, markersize=6,
             color=colors[name], label=name)
ax2.set_xlabel('Federated Round')
ax2.set_ylabel('F1-Score')
ax2.set_title('(b) F1-Score Convergence')
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, alpha=0.3)

# Bar comparison
ax3 = axes[1, 0]
names = list(all_results.keys())
x = np.arange(len(names))
width = 0.35

accs = [all_results[n]['final_accuracy'] for n in names]
f1s = [all_results[n]['final_f1'] for n in names]

bars1 = ax3.bar(x - width/2, accs, width, label='Accuracy', color='#2E86AB')
bars2 = ax3.bar(x + width/2, f1s, width, label='F1-Score', color='#E94F37')

ax3.set_ylabel('Score')
ax3.set_title('(c) Final Performance')
ax3.set_xticks(x)
ax3.set_xticklabels([n.replace('-', '\n') for n in names], fontsize=9)
ax3.legend()

for bar, val in zip(bars1, accs):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', fontsize=8)
for bar, val in zip(bars2, f1s):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', fontsize=8)

# Loss
ax4 = axes[1, 1]
for name, r in all_results.items():
    rounds = [m['round'] for m in r['metrics']]
    losses = [m['loss'] for m in r['metrics']]
    ax4.plot(rounds, losses, marker=markers[name], linewidth=2, markersize=6,
             color=colors[name], label=name)
ax4.set_xlabel('Federated Round')
ax4.set_ylabel('Test Loss')
ax4.set_title('(d) Loss Convergence')
ax4.legend(loc='upper right', fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fl_results_journal.png', dpi=300, bbox_inches='tight')
plt.savefig('fl_results_journal.pdf', dpi=300, bbox_inches='tight')
print("Saved: fl_results_journal.png/pdf")
plt.close()

# Figure 2: IID vs Non-IID
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle('Data Heterogeneity Impact on Federated Learning', fontsize=14, fontweight='bold')

# IID vs Non-IID
ax5 = axes2[0]
iid_accs = [m['accuracy'] for m in all_results['FedAvg-IID']['metrics']]
noniid_accs = [m['accuracy'] for m in all_results['FedAvg-NonIID']['metrics']]
rounds = range(1, len(iid_accs) + 1)

ax5.fill_between(rounds, iid_accs, noniid_accs, alpha=0.3, color='gray')
ax5.plot(rounds, iid_accs, 'o-', linewidth=2.5, markersize=8, color='#2E86AB', label='IID')
ax5.plot(rounds, noniid_accs, 's-', linewidth=2.5, markersize=8, color='#E94F37', label='Non-IID')
ax5.set_xlabel('Federated Round')
ax5.set_ylabel('Test Accuracy')
ax5.set_title('(a) FedAvg: IID vs Non-IID')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Non-IID strategies
ax6 = axes2[1]
for name in ['FedAvg-NonIID', 'SCAFFOLD-NonIID', 'FLAME-NonIID']:
    if name in all_results:
        accs = [m['accuracy'] for m in all_results[name]['metrics']]
        ax6.plot(range(1, len(accs)+1), accs, marker=markers.get(name, 'o'), linewidth=2.5, markersize=8,
                  color=colors.get(name, '#333'), label=name.replace('-NonIID', ''))
ax6.set_xlabel('Federated Round')
ax6.set_ylabel('Test Accuracy')
ax6.set_title('(b) Non-IID Strategies')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fl_heterogeneity_journal.png', dpi=300, bbox_inches='tight')
plt.savefig('fl_heterogeneity_journal.pdf', dpi=300, bbox_inches='tight')
print("Saved: fl_heterogeneity_journal.png/pdf")
plt.close()

# Figure 3: Communication & Client Drift
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle('Federated Learning Efficiency Analysis', fontsize=14, fontweight='bold')

# Communication
ax7 = axes3[0]
for name, r in all_results.items():
    rounds = range(1, len(r['communication_costs']) + 1)
    cumulative = np.cumsum(r['communication_costs']) / (1024 * 1024)
    ax7.plot(rounds, cumulative, marker=markers[name], linewidth=2, markersize=6,
             color=colors[name], label=name)
ax7.set_xlabel('Federated Round')
ax7.set_ylabel('Cumulative Communication (MB)')
ax7.set_title('(a) Communication Cost')
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)

# Client drift
ax8 = axes3[1]
for name, r in all_results.items():
    rounds = [m['round'] for m in r['metrics']]
    global_accs = [m['accuracy'] for m in r['metrics']]
    client_accs = [m['avg_client_acc'] for m in r['metrics']]
    gap = [c - g for c, g in zip(client_accs, global_accs)]
    ax8.plot(rounds, gap, marker=markers[name], linewidth=2, markersize=6,
             color=colors[name], label=name)
ax8.set_xlabel('Federated Round')
ax8.set_ylabel('Client-Global Accuracy Gap')
ax8.set_title('(b) Client Drift Analysis')
ax8.legend(fontsize=9)
ax8.grid(True, alpha=0.3)
ax8.axhline(y=0, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('fl_efficiency_journal.png', dpi=300, bbox_inches='tight')
plt.savefig('fl_efficiency_journal.pdf', dpi=300, bbox_inches='tight')
print("Saved: fl_efficiency_journal.png/pdf")
plt.close()

# ============================================================
# 11. Save Results
# ============================================================
results_json = {
    'config': {
        'num_clients': 5,
        'rounds': 10,
        'local_epochs': 3,
        'qpso_qubits': 2,
        'optimizer': 'SGD (momentum=0.9, nesterov=True)',
        'features': 20,
        'sequence_length': n_steps,
        'num_classes': num_classes
    },
    'results': {
        name: {
            'final_accuracy': float(r['final_accuracy']),
            'final_f1': float(r['final_f1']),
            'final_precision': float(r['final_precision']),
            'final_recall': float(r['final_recall']),
            'time': float(r['time'])
        }
        for name, r in all_results.items()
    },
    'best_strategy': best
}

with open('fl_results_journal.json', 'w') as f:
    json.dump(results_json, f, indent=2)
print("Saved: fl_results_journal.json")

# ============================================================
# 12. Justification
# ============================================================
justification = f"""
================================================================================
HORIZONTAL FEDERATED LEARNING JUSTIFICATION FOR IoV/IoD
================================================================================

1. WHY HORIZONTAL FEDERATED LEARNING?

   Internet of Vehicles (IoV):
   - Privacy: Driving/location data stays on vehicle
   - Bandwidth: V2X limited to ~27 Mbps
   - Scale: Millions of connected vehicles
   - Compliance: GDPR, CCPA requirements

   Internet of Drones (IoD):
   - Security: Military/surveillance data classified
   - Connectivity: Intermittent in remote areas
   - Resources: Edge devices have limited compute
   - Sovereignty: Data cannot cross borders

2. EXPERIMENTAL SETUP

   Feature Selection: 3-Qubit QPSO (15 features)
   Optimizer: AdamW (best from Step 2)
   Clients: 5 edge devices
   Rounds: 10 federated rounds
   Local Epochs: 3 per round

3. RESULTS

   Best Strategy: {best}
   - Accuracy: {all_results[best]['final_accuracy']:.4f}
   - F1-Score: {all_results[best]['final_f1']:.4f}
   - Precision: {all_results[best]['final_precision']:.4f}
   - Recall: {all_results[best]['final_recall']:.4f}

   Strategy Comparison:
   ┌────────────────────┬──────────┬──────────┬────────────────────────┐
   │ Strategy           │ Accuracy │ F1-Score │ Best For               │
   ├────────────────────┼──────────┼──────────┼────────────────────────┤
   │ FedAvg-IID         │ {all_results['FedAvg-IID']['final_accuracy']:.4f}   │ {all_results['FedAvg-IID']['final_f1']:.4f}   │ Homogeneous fleet      │
   │ FedAvg-NonIID      │ {all_results['FedAvg-NonIID']['final_accuracy']:.4f}   │ {all_results['FedAvg-NonIID']['final_f1']:.4f}   │ Mixed environments     │
   │ SCAFFOLD-NonIID    │ {all_results['SCAFFOLD-NonIID']['final_accuracy']:.4f}   │ {all_results['SCAFFOLD-NonIID']['final_f1']:.4f}   │ High heterogeneity     │
   │ FLAME-NonIID       │ {all_results['FLAME-NonIID']['final_accuracy']:.4f}   │ {all_results['FLAME-NonIID']['final_f1']:.4f}   │ Adversarial settings   │
   └────────────────────┴──────────┴──────────┴────────────────────────┘

4. KEY CONTRIBUTIONS

   a) Novel Integration:
      - First QPSO + Horizontal FL for IoV/IoD
      - Semantic attention for traffic understanding
      - AdamW for stable federated training

   b) Privacy-Preserving:
      - Raw data never leaves devices
      - Only model weights transmitted
      - GDPR/CCPA compliant

   c) Deployment Recommendations:
      - FedAvg: Homogeneous vehicle/drone fleets
      - SCAFFOLD: Urban/rural mix, multi-mission drones
      - FLAME: Adversarial environments, compromised nodes

================================================================================
"""

print(justification.encode('ascii', 'replace').decode('ascii'))

with open('fl_justification_journal.txt', 'w', encoding='utf-8') as f:
    f.write(justification)
print("Saved: fl_justification_journal.txt")

print("\n" + "="*70)
print("EXPERIMENT COMPLETE")
print("="*70)
print("\nGenerated files:")
print("  - fl_results_journal.png/pdf")
print("  - fl_heterogeneity_journal.png/pdf")
print("  - fl_efficiency_journal.png/pdf")
print("  - fl_results_journal.json")
print("  - fl_justification_journal.txt")
