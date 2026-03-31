# -*- coding: utf-8 -*-
"""
CONQUEST Journal: Federated Learning Data Heterogeneity Study
QPSO-LSTM-VAE+SGD (2-Qubit) with Multiple FL Strategies
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, 
                                      LayerNormalization, MultiHeadAttention,
                                      Bidirectional, BatchNormalization)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                             recall_score, f1_score, roc_auc_score)
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import gc
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

print("="*70)
print("CONQUEST: Federated Learning Data Heterogeneity Study")
print("QPSO-LSTM-VAE+SGD (2-Qubit)")
print("="*70)

@dataclass
class FederatedConfig:
    num_clients: int = 5
    num_rounds: int = 15
    local_epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 0.001
    num_qubits: int = 2
    num_particles: int = 20
    qpso_iterations: int = 30
    num_features_to_select: int = 30
    sequence_length: int = 10
    lstm_units: int = 256
    attention_heads: int = 8
    dropout_rate: float = 0.4
    dirichlet_alpha: float = 0.5

class FederatedStrategy(Enum):
    FEDAVG = "FedAvg"
    FEDPROX = "FedProx"
    SCAFFOLD = "SCAFFOLD"
    FLAME = "FLAME"
    FEDNOVA = "FedNova"

class HeterogeneityType(Enum):
    IID = "IID"
    NON_IID_LABEL = "Non-IID-Label"
    NON_IID_QUANTITY = "Non-IID-Quantity"
    DIRICHLET = "Dirichlet"

class QuantumFeatureSelector:
    def __init__(self, num_qubits=2, num_particles=20, iterations=30, num_features_to_select=30):
        self.num_qubits = num_qubits
        self.num_particles = num_particles
        self.iterations = iterations
        self.num_features_to_select = num_features_to_select
        self.best_features = None
        self.fitness_history = []
        
    def _initialize_quantum_states(self, n_features):
        state_dim = 2 ** self.num_qubits
        states = np.ones((self.num_particles, n_features)) / np.sqrt(state_dim)
        states += np.random.randn(*states.shape) * (1.0 / (self.num_qubits ** 2 + 1))
        states = np.abs(states)
        return states / np.sum(states, axis=1, keepdims=True)
    
    def _quantum_rotation_gate(self, state, p_best, g_best, iteration):
        theta = (np.pi / (2 * self.num_qubits)) * (1 - iteration / self.iterations)
        new_state = state + np.random.rand() * np.cos(theta) * (p_best - state)
        new_state += np.random.rand() * np.sin(theta) * (g_best - state)
        mask = np.random.rand(*state.shape) < (1.0 / (2 ** self.num_qubits))
        new_state[mask] = np.random.rand(np.sum(mask))
        return np.abs(new_state) / (np.sum(np.abs(new_state)) + 1e-10)
    
    def _fitness_function(self, feature_indices, X, y):
        if len(feature_indices) == 0: return 0.0
        X_sub = X[:, feature_indices.astype(int)]
        overall_mean = np.mean(X_sub, axis=0)
        between_var, within_var = 0, 0
        for cls in np.unique(y):
            cls_data = X_sub[y == cls]
            if len(cls_data) == 0: continue
            cls_mean = np.mean(cls_data, axis=0)
            between_var += len(cls_data) * np.sum((cls_mean - overall_mean) ** 2)
            within_var += np.sum(np.var(cls_data, axis=0))
        return between_var / (within_var + 1e-10)
    
    def _measure_quantum_state(self, state):
        probs = state ** 2
        probs = probs / (np.sum(probs) + 1e-10)
        return np.random.choice(len(state), size=min(self.num_features_to_select, len(state)), 
                               replace=False, p=probs)
    
    def fit(self, X, y):
        print(f"\n[QPSO-{self.num_qubits}Q] Feature selection...")
        quantum_states = self._initialize_quantum_states(X.shape[1])
        p_best_states = quantum_states.copy()
        p_best_fitness = np.zeros(self.num_particles)
        g_best_state, g_best_fitness = quantum_states[0].copy(), 0
        
        for i in range(self.num_particles):
            fitness = self._fitness_function(self._measure_quantum_state(quantum_states[i]), X, y)
            p_best_fitness[i] = fitness
            if fitness > g_best_fitness:
                g_best_fitness, g_best_state = fitness, quantum_states[i].copy()
        
        for it in range(self.iterations):
            for i in range(self.num_particles):
                quantum_states[i] = self._quantum_rotation_gate(quantum_states[i], p_best_states[i], g_best_state, it)
                fitness = self._fitness_function(self._measure_quantum_state(quantum_states[i]), X, y)
                if fitness > p_best_fitness[i]:
                    p_best_fitness[i], p_best_states[i] = fitness, quantum_states[i].copy()
                if fitness > g_best_fitness:
                    g_best_fitness, g_best_state = fitness, quantum_states[i].copy()
            self.fitness_history.append(g_best_fitness)
        
        self.best_features = self._measure_quantum_state(g_best_state)
        print(f"[QPSO-{self.num_qubits}Q] Selected {len(self.best_features)} features")
        return self
    
    def transform(self, X): return X[:, self.best_features]
    def fit_transform(self, X, y): return self.fit(X, y).transform(X)

def build_model(input_shape, num_classes, config):
    """Build QPSO-LSTM-VAE+SGD model (clone-compatible version)."""
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(config.lstm_units, return_sequences=True, dropout=config.dropout_rate))(inputs)
    x = BatchNormalization()(x)
    attn = MultiHeadAttention(num_heads=config.attention_heads, key_dim=config.lstm_units//config.attention_heads)(x, x)
    x = LayerNormalization()(x + attn)
    x = Bidirectional(LSTM(config.lstm_units//2, return_sequences=False, dropout=config.dropout_rate))(x)
    x = BatchNormalization()(x)
    # Simplified latent space (no Lambda for clone compatibility)
    x = Dense(64, activation='relu', name='latent')(x)
    x = Dropout(config.dropout_rate)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(config.dropout_rate)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=SGD(learning_rate=config.learning_rate, momentum=0.9, nesterov=True),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

class DataPartitioner:
    def __init__(self, num_clients, config):
        self.num_clients = num_clients
        self.config = config
        
    def partition(self, X, y, heterogeneity):
        n = len(y)
        idx = np.arange(n)
        y_labels = np.argmax(y, axis=1) if len(y.shape) > 1 else y
        
        if heterogeneity == HeterogeneityType.IID:
            np.random.shuffle(idx)
            splits = np.array_split(idx, self.num_clients)
        elif heterogeneity == HeterogeneityType.NON_IID_LABEL:
            splits = np.array_split(np.argsort(y_labels), self.num_clients)
        elif heterogeneity == HeterogeneityType.NON_IID_QUANTITY:
            np.random.shuffle(idx)
            props = np.random.dirichlet(np.ones(self.num_clients) * 0.5)
            split_pts = (np.cumsum(props) * n).astype(int)[:-1]
            splits = np.split(idx, split_pts)
        else:  # Dirichlet
            num_classes = len(np.unique(y_labels))
            label_dist = np.random.dirichlet([self.config.dirichlet_alpha] * self.num_clients, num_classes)
            client_idx = [[] for _ in range(self.num_clients)]
            for c in range(num_classes):
                c_idx = np.where(y_labels == c)[0]
                np.random.shuffle(c_idx)
                split_pts = (np.cumsum(label_dist[c] / label_dist[c].sum()) * len(c_idx)).astype(int)[:-1]
                for i, s in enumerate(np.split(c_idx, split_pts)):
                    client_idx[i].extend(s.tolist())
            splits = [np.array(ci) if ci else np.random.choice(n, 100) for ci in client_idx]
        
        return [(X[s], y[s]) for s in splits if len(s) > 0][:self.num_clients]

class FederatedAggregator:
    def __init__(self, strategy, config):
        self.strategy = strategy
        self.config = config
        self.server_control = None
        
    def aggregate(self, client_weights, client_sizes, client_deltas=None, local_steps=None):
        if self.strategy == FederatedStrategy.FLAME:
            return self._flame(client_weights, client_sizes)
        elif self.strategy == FederatedStrategy.SCAFFOLD:
            return self._scaffold(client_weights, client_deltas, client_sizes)
        elif self.strategy == FederatedStrategy.FEDNOVA:
            return self._fednova(client_weights, client_sizes, local_steps or [self.config.local_epochs]*len(client_weights))
        return self._fedavg(client_weights, client_sizes)
    
    def _fedavg(self, weights_list, sizes):
        total = sum(sizes)
        return [sum(s/total * w[i] for w, s in zip(weights_list, sizes)) for i in range(len(weights_list[0]))]
    
    def _scaffold(self, weights_list, deltas, sizes):
        avg = self._fedavg(weights_list, sizes)
        if self.server_control is None:
            self.server_control = [np.zeros_like(w) for w in weights_list[0]]
        if deltas:
            total = sum(sizes)
            self.server_control = [sc + sum(s/total * d[i] for d, s in zip(deltas, sizes)) 
                                   for i, sc in enumerate(self.server_control)]
        return avg
    
    def _flame(self, weights_list, sizes):
        flat = [np.concatenate([w.flatten() for w in ws]) for ws in weights_list]
        n = len(weights_list)
        sims = np.array([[1 - cosine(flat[i], flat[j]) if i != j else 0 for j in range(n)] for i in range(n)])
        trust = np.mean(sims, axis=1)
        mask = trust >= np.percentile(trust, 25)
        print(f"    [FLAME] Trusted: {mask.sum()}/{n}")
        trusted_w = [w for w, m in zip(weights_list, mask) if m]
        trusted_s = [s for s, m in zip(sizes, mask) if m]
        return self._fedavg(trusted_w, trusted_s) if trusted_w else self._fedavg(weights_list, sizes)
    
    def _fednova(self, weights_list, sizes, steps):
        total = sum(sizes)
        tau_eff = sum(s * t for s, t in zip(sizes, steps)) / total
        return [sum((s/total) * (tau_eff/t) * w[i] for w, s, t in zip(weights_list, sizes, steps)) 
                for i in range(len(weights_list[0]))]

class FederatedTrainer:
    def __init__(self, config):
        self.config = config
        self.global_model = None
        self.round_metrics = []
        self.client_drift_history = []
        
    def train(self, X_train, y_train, X_test, y_test, input_shape, num_classes, class_names, strategy, heterogeneity):
        print(f"\n{'='*60}\nFL: {strategy.value} | Heterogeneity: {heterogeneity.value}\n{'='*60}")
        
        self.global_model = build_model(input_shape, num_classes, self.config)
        partitioner = DataPartitioner(self.config.num_clients, self.config)
        client_data = partitioner.partition(X_train, y_train, heterogeneity)
        aggregator = FederatedAggregator(strategy, self.config)
        
        print(f"Client sizes: {[len(c[1]) for c in client_data]}")
        self.round_metrics, self.client_drift_history = [], []
        start = time.time()
        
        for r in range(self.config.num_rounds):
            print(f"\n--- Round {r+1}/{self.config.num_rounds} ---")
            client_weights, client_deltas, client_sizes, client_accs = [], [], [], []
            
            for X_c, y_c in client_data:
                local = clone_model(self.global_model)
                local.set_weights(self.global_model.get_weights())
                local.compile(optimizer=SGD(self.config.learning_rate, momentum=0.9, nesterov=True),
                             loss='categorical_crossentropy', metrics=['accuracy'])
                init_w = local.get_weights()
                h = local.fit(X_c, y_c, epochs=self.config.local_epochs, batch_size=self.config.batch_size, verbose=0)
                final_w = local.get_weights()
                client_weights.append(final_w)
                client_deltas.append([f - i for f, i in zip(final_w, init_w)])
                client_sizes.append(len(y_c))
                client_accs.append(h.history['accuracy'][-1])
                del local
            
            drift = self._compute_drift(client_weights)
            self.client_drift_history.append(drift)
            print(f"  Client accs: {[f'{a:.3f}' for a in client_accs]}, Drift: {drift:.4f}")
            
            new_w = aggregator.aggregate(client_weights, client_sizes, client_deltas)
            self.global_model.set_weights(new_w)
            
            metrics = self._evaluate(X_test, y_test, class_names)
            metrics.update({'round': r+1, 'avg_client_acc': np.mean(client_accs), 'drift': drift})
            self.round_metrics.append(metrics)
            print(f"  Global: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
            keras.backend.clear_session()
        
        final = self._evaluate(X_test, y_test, class_names)
        return {'strategy': strategy.value, 'heterogeneity': heterogeneity.value, 'final_metrics': final,
                'round_metrics': self.round_metrics, 'drift_history': self.client_drift_history,
                'time': time.time() - start}
    
    def _compute_drift(self, weights_list):
        flat = [np.concatenate([w.flatten() for w in ws]) for ws in weights_list]
        drifts = [np.linalg.norm(flat[i] - flat[j]) for i in range(len(flat)) for j in range(i+1, len(flat))]
        return np.mean(drifts) if drifts else 0.0
    
    def _evaluate(self, X, y, class_names):
        y_prob = self.global_model.predict(X, verbose=0)
        y_pred, y_true = np.argmax(y_prob, axis=1), np.argmax(y, axis=1)
        cm = confusion_matrix(y_true, y_pred)
        far, md, auc = {}, {}, {}
        for i, cn in enumerate(class_names):
            fp = np.sum(cm[:, i]) - cm[i, i]
            tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
            fn = np.sum(cm[i, :]) - cm[i, i]
            far[cn] = fp / (fp + tn) if (fp + tn) > 0 else 0
            md[cn] = {'count': int(fn), 'rate': fn / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0}
            try: auc[cn] = roc_auc_score((y_true == i).astype(int), y_prob[:, i])
            except: auc[cn] = 0.0
        return {'accuracy': accuracy_score(y_true, y_pred), 'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0), 'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
                'confusion_matrix': cm.tolist(), 'far_per_class': far, 'missed_detections': md, 'auc_roc': auc}

def plot_results(results, class_names):
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Accuracy heatmap
    strategies = sorted(set(r['strategy'] for r in results.values()))
    heteros = sorted(set(r['heterogeneity'] for r in results.values()))
    matrix = np.zeros((len(strategies), len(heteros)))
    for name, r in results.items():
        matrix[strategies.index(r['strategy']), heteros.index(r['heterogeneity'])] = r['final_metrics']['accuracy'] * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=50, vmax=100)
    ax.set_xticks(range(len(heteros))); ax.set_xticklabels(heteros, rotation=45, ha='right')
    ax.set_yticks(range(len(strategies))); ax.set_yticklabels(strategies)
    for i in range(len(strategies)):
        for j in range(len(heteros)):
            ax.text(j, i, f'{matrix[i,j]:.1f}%', ha='center', va='center', fontsize=11)
    ax.set_title('Accuracy: Strategy vs Heterogeneity', fontsize=14, fontweight='bold')
    plt.colorbar(im, label='Accuracy (%)')
    plt.tight_layout(); plt.savefig('fl_accuracy_heatmap.png', dpi=300); plt.show()
    
    # 2. Convergence
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for name, r in results.items():
        rounds = [m['round'] for m in r['round_metrics']]
        axes[0,0].plot(rounds, [m['accuracy']*100 for m in r['round_metrics']], 'o-', ms=4, label=name)
        axes[0,1].plot(rounds, [m['f1_score']*100 for m in r['round_metrics']], 's-', ms=4, label=name)
        if 'drift_history' in r:
            axes[1,0].plot(range(1, len(r['drift_history'])+1), r['drift_history'], '^-', ms=4, label=name)
    
    axes[0,0].set_xlabel('Round'); axes[0,0].set_ylabel('Accuracy (%)'); axes[0,0].set_title('Accuracy Convergence'); axes[0,0].legend(fontsize=7)
    axes[0,1].set_xlabel('Round'); axes[0,1].set_ylabel('F1-Score (%)'); axes[0,1].set_title('F1 Convergence'); axes[0,1].legend(fontsize=7)
    axes[1,0].set_xlabel('Round'); axes[1,0].set_ylabel('Drift (L2)'); axes[1,0].set_title('Client Drift'); axes[1,0].legend(fontsize=7)
    
    # Final metrics bar
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    x = np.arange(len(metrics))
    width = 0.8 / len(results)
    for i, (name, r) in enumerate(results.items()):
        axes[1,1].bar(x + i*width, [r['final_metrics'][m]*100 for m in metrics], width, label=name[:12])
    axes[1,1].set_xticks(x + width*(len(results)-1)/2); axes[1,1].set_xticklabels(['Acc', 'Prec', 'Rec', 'F1'])
    axes[1,1].set_ylabel('Score (%)'); axes[1,1].set_title('Final Metrics'); axes[1,1].legend(fontsize=6)
    plt.tight_layout(); plt.savefig('fl_convergence.png', dpi=300); plt.show()
    
    # 3. Best result per-class metrics
    best = max(results.values(), key=lambda x: x['final_metrics']['accuracy'])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    classes = list(best['final_metrics']['far_per_class'].keys())
    
    axes[0].barh(classes, [best['final_metrics']['far_per_class'][c]*100 for c in classes], color='salmon')
    axes[0].set_xlabel('FAR (%)'); axes[0].set_title(f"FAR per Class ({best['strategy']})")
    
    axes[1].barh(classes, [best['final_metrics']['missed_detections'][c]['rate']*100 for c in classes], color='orange')
    axes[1].set_xlabel('Missed Detection Rate (%)'); axes[1].set_title('Missed Detections')
    
    axes[2].barh(classes, [best['final_metrics']['auc_roc'][c] for c in classes], color='green')
    axes[2].set_xlabel('AUC-ROC'); axes[2].set_title('AUC-ROC per Class'); axes[2].axvline(0.9, color='red', ls='--')
    plt.tight_layout(); plt.savefig('fl_per_class_metrics.png', dpi=300); plt.show()
    
    print(f"\nBest: {best['strategy']}-{best['heterogeneity']} with {best['final_metrics']['accuracy']*100:.2f}% accuracy")

def main():
    config = FederatedConfig()
    data_path = './0.ACI-IoT-2023.csv'
    
    print("\n[1/4] Loading data...")
    df = pd.read_csv(data_path)
    label_col = 'Label' if 'Label' in df.columns else df.columns[-1]
    X = df.drop(columns=[label_col]).apply(lambda c: pd.factorize(c)[0] if c.dtype == 'object' else c).fillna(0).replace([np.inf, -np.inf], 0)
    le = LabelEncoder()
    y = le.fit_transform(df[label_col])
    class_names = list(le.classes_)
    
    print(f"\n[2/4] Preprocessing...")
    X_scaled = StandardScaler().fit_transform(X)
    
    print("\n[3/4] QPSO Feature Selection (2-qubit)...")
    X_sel = QuantumFeatureSelector(num_qubits=2, num_features_to_select=30).fit_transform(X_scaled, y)
    
    print("\n[4/4] Creating sequences...")
    n_seq = len(X_sel) // config.sequence_length
    X_seq = X_sel[:n_seq * config.sequence_length].reshape(n_seq, config.sequence_length, -1)
    y_seq = to_categorical(y[:n_seq * config.sequence_length:config.sequence_length], len(class_names))
    
    X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    strategies = [FederatedStrategy.FEDAVG, FederatedStrategy.FEDPROX, FederatedStrategy.SCAFFOLD, 
                  FederatedStrategy.FLAME, FederatedStrategy.FEDNOVA]
    heterogeneities = [HeterogeneityType.IID, HeterogeneityType.NON_IID_LABEL, 
                       HeterogeneityType.NON_IID_QUANTITY, HeterogeneityType.DIRICHLET]
    
    results = {}
    for strat in strategies:
        for hetero in heterogeneities:
            name = f"{strat.value}-{hetero.value}"
            print(f"\n{'#'*60}\n# {name}\n{'#'*60}")
            try:
                results[name] = FederatedTrainer(config).train(X_train, y_train, X_test, y_test, 
                                                               X_train.shape[1:], len(class_names), class_names, strat, hetero)
                print(f">>> {name}: {results[name]['final_metrics']['accuracy']*100:.2f}%")
            except Exception as e:
                print(f"ERROR: {e}")
            gc.collect(); keras.backend.clear_session()
    
    # Save results
    def to_json(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)): return float(obj)
        if isinstance(obj, dict): return {k: to_json(v) for k, v in obj.items()}
        if isinstance(obj, list): return [to_json(i) for i in obj]
        return obj
    
    with open('federated_heterogeneity_results.json', 'w') as f:
        json.dump(to_json(results), f, indent=2)
    
    # Summary table
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"{'Strategy':<12} {'Heterogeneity':<18} {'Accuracy':<10} {'F1':<10} {'Time':<10}")
    print("-"*60)
    for name, r in sorted(results.items(), key=lambda x: -x[1]['final_metrics']['accuracy']):
        print(f"{r['strategy']:<12} {r['heterogeneity']:<18} {r['final_metrics']['accuracy']*100:>7.2f}%  {r['final_metrics']['f1_score']*100:>7.2f}%  {r['time']:>7.1f}s")
    
    plot_results(results, class_names)
    
    # LaTeX table
    with open('federated_heterogeneity_table.tex', 'w') as f:
        f.write("\\begin{table}[htbp]\n\\centering\n\\caption{Federated Learning Heterogeneity Study Results}\n")
        f.write("\\begin{tabular}{llcccc}\n\\hline\n")
        f.write("Strategy & Heterogeneity & Accuracy & Precision & Recall & F1-Score \\\\\n\\hline\n")
        for name, r in sorted(results.items(), key=lambda x: -x[1]['final_metrics']['accuracy']):
            m = r['final_metrics']
            f.write(f"{r['strategy']} & {r['heterogeneity']} & {m['accuracy']*100:.2f}\\% & {m['precision']*100:.2f}\\% & {m['recall']*100:.2f}\\% & {m['f1_score']*100:.2f}\\% \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}")
    
    print("\nResults saved to: federated_heterogeneity_results.json, federated_heterogeneity_table.tex")
    print("Plots saved: fl_accuracy_heatmap.png, fl_convergence.png, fl_per_class_metrics.png")

if __name__ == "__main__":
    main()
