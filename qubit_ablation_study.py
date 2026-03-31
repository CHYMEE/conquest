# -*- coding: utf-8 -*-
"""
================================================================================
CONQUEST Journal Extension: Qubit Ablation Study
================================================================================

Ablation study of QPSO-LSTM-VAE+SGD model across different quantum qubit sizes.
Tests qubit configurations: 1, 2, 3, 4, 8

Metrics captured:
- Number of qubits
- Accuracy, Precision, F1-Score, Recall
- Training/Inference Time
- False Alarm Rate (FAR) per class
- Mean Squared Error (MSE)
- Missed Detections per class
- AUC-ROC per class
- Memory Complexity

Authors: Sandra et al.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, 
                                      LayerNormalization, MultiHeadAttention,
                                      RepeatVector, TimeDistributed)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_curve, auc, mean_squared_error,
                             roc_auc_score)
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import tracemalloc
import gc
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AblationConfig:
    """Configuration for ablation study"""
    qubit_sizes: List[int] = None
    num_particles: int = 20
    qpso_iterations: int = 30
    num_features_to_select: int = 20
    subsample_size: int = 50000
    sequence_length: int = 10
    lstm_units: int = 128
    attention_heads: int = 4
    dropout_rate: float = 0.3
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 0.001
    
    def __post_init__(self):
        if self.qubit_sizes is None:
            self.qubit_sizes = [1, 2, 3, 4, 8]


# ============================================================================
# SMART SUBSAMPLER
# ============================================================================

class SmartSubsampler:
    """Intelligent subsampling with class balancing"""
    
    def __init__(self, target_size: int = 50000):
        self.target_size = target_size
        
    def subsample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.DataFrame(X)
        df['label'] = y
        
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        samples_per_class = self.target_size // n_classes
        
        balanced_dfs = []
        
        for cls in unique_classes:
            cls_data = df[df['label'] == cls]
            n_samples = len(cls_data)
            
            if n_samples < samples_per_class:
                sampled = resample(cls_data, replace=True, 
                                  n_samples=samples_per_class, random_state=42)
            else:
                sampled = cls_data.sample(n=min(samples_per_class, n_samples), 
                                         random_state=42)
            balanced_dfs.append(sampled)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        X_balanced = balanced_df.drop('label', axis=1).values
        y_balanced = balanced_df['label'].values
        
        print(f"[Subsampler] Original: {len(y)}, Subsampled: {len(y_balanced)}")
        return X_balanced, y_balanced


# ============================================================================
# QUANTUM-INSPIRED FEATURE SELECTOR (QPSO)
# ============================================================================

class QuantumInspiredFeatureSelector:
    """
    Quantum-Inspired Particle Swarm Optimization (QPSO) for feature selection.
    Supports variable qubit configurations: 1, 2, 3, 4, 8 qubits.
    """
    
    def __init__(self, num_qubits: int = 4, num_particles: int = 20, 
                 iterations: int = 30, num_features_to_select: int = 20):
        self.num_qubits = num_qubits
        self.num_particles = num_particles
        self.iterations = iterations
        self.num_features_to_select = num_features_to_select
        self.quantum_states = None
        self.best_features = None
        self.fitness_history = []
        self.computation_time = 0
        self.memory_usage = 0
        
    def _initialize_quantum_states(self, n_features: int) -> np.ndarray:
        """Initialize quantum-inspired probability amplitudes."""
        state_dim = 2 ** self.num_qubits
        states = np.ones((self.num_particles, n_features)) / np.sqrt(state_dim)
        noise_scale = 1.0 / (self.num_qubits ** 2 + 1)
        states += np.random.randn(*states.shape) * noise_scale
        states = np.abs(states)
        states = states / np.sum(states, axis=1, keepdims=True)
        return states
    
    def _quantum_rotation_gate(self, state: np.ndarray, 
                                p_best: np.ndarray, 
                                g_best: np.ndarray,
                                iteration: int) -> np.ndarray:
        """Apply quantum-inspired rotation gate for state update."""
        theta_max = np.pi / (2 * max(self.num_qubits, 1))
        theta = theta_max * (1 - iteration / self.iterations)
        
        delta_p = p_best - state
        delta_g = g_best - state
        
        alpha = np.random.rand() * np.cos(theta)
        beta = np.random.rand() * np.sin(theta)
        
        new_state = state + alpha * delta_p + beta * delta_g
        
        collapse_prob = 1.0 / (2 ** self.num_qubits)
        mask = np.random.rand(*state.shape) < collapse_prob
        new_state[mask] = np.random.rand(np.sum(mask))
        
        new_state = np.abs(new_state)
        new_state = new_state / (np.sum(new_state) + 1e-10)
        
        return new_state
    
    def _fitness_function(self, feature_indices: np.ndarray, 
                          X: np.ndarray, y: np.ndarray) -> float:
        """Compute fitness based on class separability (Fisher criterion)."""
        if len(feature_indices) == 0:
            return 0.0
        
        feature_indices = feature_indices.astype(int)
        X_subset = X[:, feature_indices]
        
        unique_classes = np.unique(y)
        overall_mean = np.mean(X_subset, axis=0)
        between_var = 0
        within_var = 0
        
        for cls in unique_classes:
            cls_mask = (y == cls)
            cls_data = X_subset[cls_mask]
            if len(cls_data) == 0:
                continue
            cls_mean = np.mean(cls_data, axis=0)
            between_var += len(cls_data) * np.sum((cls_mean - overall_mean) ** 2)
            within_var += np.sum(np.var(cls_data, axis=0))
        
        fitness = between_var / (within_var + 1e-10)
        return fitness
    
    def _measure_quantum_state(self, state: np.ndarray) -> np.ndarray:
        """Collapse quantum state to classical feature selection."""
        probabilities = state ** 2
        probabilities = probabilities / (np.sum(probabilities) + 1e-10)
        
        n_features = len(state)
        n_select = min(self.num_features_to_select, n_features)
        
        selected = np.random.choice(
            n_features, 
            size=n_select,
            replace=False,
            p=probabilities
        )
        return selected
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantumInspiredFeatureSelector':
        """Run QPSO feature selection with specified qubit configuration."""
        # Start memory tracking
        tracemalloc.start()
        start_time = time.time()
        
        n_features = X.shape[1]
        print(f"\n[QPSO-{self.num_qubits}Q] Starting with {n_features} features")
        
        self.quantum_states = self._initialize_quantum_states(n_features)
        
        p_best_states = self.quantum_states.copy()
        p_best_fitness = np.zeros(self.num_particles)
        g_best_state = self.quantum_states[0].copy()
        g_best_fitness = 0
        
        for i in range(self.num_particles):
            features = self._measure_quantum_state(self.quantum_states[i])
            fitness = self._fitness_function(features, X, y)
            p_best_fitness[i] = fitness
            if fitness > g_best_fitness:
                g_best_fitness = fitness
                g_best_state = self.quantum_states[i].copy()
        
        for iteration in range(self.iterations):
            for i in range(self.num_particles):
                self.quantum_states[i] = self._quantum_rotation_gate(
                    self.quantum_states[i],
                    p_best_states[i],
                    g_best_state,
                    iteration
                )
                
                features = self._measure_quantum_state(self.quantum_states[i])
                fitness = self._fitness_function(features, X, y)
                
                if fitness > p_best_fitness[i]:
                    p_best_fitness[i] = fitness
                    p_best_states[i] = self.quantum_states[i].copy()
                
                if fitness > g_best_fitness:
                    g_best_fitness = fitness
                    g_best_state = self.quantum_states[i].copy()
            
            self.fitness_history.append(g_best_fitness)
        
        self.best_features = self._measure_quantum_state(g_best_state)
        
        # Record time and memory
        self.computation_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        self.memory_usage = peak / (1024 * 1024)  # Convert to MB
        tracemalloc.stop()
        
        print(f"[QPSO-{self.num_qubits}Q] Selected {len(self.best_features)} features")
        print(f"[QPSO-{self.num_qubits}Q] Time: {self.computation_time:.2f}s, Memory: {self.memory_usage:.2f}MB")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using selected features."""
        if self.best_features is None:
            raise ValueError("Must call fit() before transform()")
        return X[:, self.best_features]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)


# ============================================================================
# LSTM-VAE MODEL WITH ATTENTION
# ============================================================================

class LSTM_VAE_Model:
    """
    LSTM-VAE model with attention mechanism for intrusion detection.
    Uses SGD optimizer as established in the conference paper.
    """
    
    def __init__(self, input_shape: Tuple[int, int], num_classes: int,
                 lstm_units: int = 128, attention_heads: int = 4,
                 dropout_rate: float = 0.3, latent_dim: int = 32):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate
        self.latent_dim = latent_dim
        self.model = None
        self.encoder = None
        self.history = None
        self.training_time = 0
        self.inference_time = 0
        self.memory_usage = 0
        
    def build(self) -> Model:
        """Build LSTM-VAE model with attention."""
        inputs = Input(shape=self.input_shape)
        
        # Encoder LSTM
        x = LSTM(self.lstm_units, return_sequences=True, 
                 dropout=self.dropout_rate)(inputs)
        x = LayerNormalization()(x)
        
        # Multi-head self-attention
        attention_output = MultiHeadAttention(
            num_heads=self.attention_heads,
            key_dim=self.lstm_units // self.attention_heads
        )(x, x)
        x = x + attention_output
        x = LayerNormalization()(x)
        
        # Second LSTM layer
        x = LSTM(self.lstm_units // 2, return_sequences=False,
                 dropout=self.dropout_rate)(x)
        x = LayerNormalization()(x)
        
        # VAE latent space
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
        
        # Reparameterization trick
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        z = keras.layers.Lambda(sampling, name='z')([z_mean, z_log_var])
        
        # Decoder/Classifier
        x = Dense(64, activation='relu')(z)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(32, activation='relu')(x)
        
        # Output layer
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        self.encoder = Model(inputs=inputs, outputs=[z_mean, z_log_var, z])
        
        return self.model
    
    def compile_with_sgd(self, learning_rate: float = 0.001):
        """Compile model with SGD optimizer (as per conference paper)."""
        optimizer = keras.optimizers.SGD(
            learning_rate=learning_rate, 
            momentum=0.9,
            nesterov=True
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return self
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50, batch_size: int = 64) -> Dict:
        """Train the model with early stopping."""
        tracemalloc.start()
        start_time = time.time()
        
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=10, 
                         restore_best_weights=True, mode='max'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                             patience=5, min_lr=1e-6)
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.training_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        self.memory_usage = peak / (1024 * 1024)
        tracemalloc.stop()
        
        return self.history.history
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with timing."""
        start_time = time.time()
        y_pred_prob = self.model.predict(X, verbose=0)
        self.inference_time = time.time() - start_time
        y_pred = np.argmax(y_pred_prob, axis=1)
        return y_pred, y_pred_prob


# ============================================================================
# COMPREHENSIVE METRICS CALCULATOR
# ============================================================================

class MetricsCalculator:
    """Calculate all required metrics for ablation study."""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_pred_prob: np.ndarray, y_true_onehot: np.ndarray,
                               training_time: float, inference_time: float,
                               qpso_time: float, memory_mb: float) -> Dict:
        """Calculate comprehensive metrics."""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Per-class metrics
        far_per_class = {}  # False Alarm Rate
        missed_detections_per_class = {}
        auc_roc_per_class = {}
        
        for i, class_name in enumerate(self.class_names):
            # False Alarm Rate (FPR) = FP / (FP + TN)
            fp = np.sum(cm[:, i]) - cm[i, i]  # False positives for class i
            tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
            far = fp / (fp + tn + 1e-10)
            far_per_class[class_name] = float(far)
            
            # Missed Detections (FN) = FN / (FN + TP)
            fn = np.sum(cm[i, :]) - cm[i, i]  # False negatives for class i
            tp = cm[i, i]
            missed_rate = fn / (fn + tp + 1e-10)
            missed_detections_per_class[class_name] = {
                'count': int(fn),
                'rate': float(missed_rate)
            }
            
            # AUC-ROC per class (one-vs-rest)
            try:
                if y_true_onehot.shape[1] > i:
                    auc_score = roc_auc_score(
                        y_true_onehot[:, i], 
                        y_pred_prob[:, i]
                    )
                    auc_roc_per_class[class_name] = float(auc_score)
                else:
                    auc_roc_per_class[class_name] = 0.0
            except Exception:
                auc_roc_per_class[class_name] = 0.0
        
        # MSE (reconstruction error proxy using prediction confidence)
        mse = mean_squared_error(y_true_onehot, y_pred_prob)
        
        # Total time
        total_time = qpso_time + training_time + inference_time
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'mse': float(mse),
            'training_time_sec': float(training_time),
            'inference_time_sec': float(inference_time),
            'qpso_time_sec': float(qpso_time),
            'total_time_sec': float(total_time),
            'memory_mb': float(memory_mb),
            'far_per_class': far_per_class,
            'missed_detections_per_class': missed_detections_per_class,
            'auc_roc_per_class': auc_roc_per_class,
            'confusion_matrix': cm.tolist()
        }


# ============================================================================
# ABLATION STUDY RUNNER
# ============================================================================

class QubitAblationStudy:
    """Run ablation study across different qubit configurations."""
    
    def __init__(self, config: AblationConfig = None):
        self.config = config or AblationConfig()
        self.results = {}
        self.class_names = None
        
    def load_and_preprocess_data(self, csv_path: str) -> Tuple:
        """Load and preprocess the dataset."""
        print("\n" + "="*80)
        print("LOADING AND PREPROCESSING DATA")
        print("="*80)
        
        df = pd.read_csv(csv_path)
        
        # Identify target column
        target_col = 'Label' if 'Label' in df.columns else 'label'
        
        # Drop non-feature columns
        drop_cols = ['Connection Type', 'Flow ID', 'Timestamp']
        drop_cols = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=drop_cols, errors='ignore')
        
        y_original = df[target_col].values
        X = df.drop(columns=[target_col])
        
        # Encode non-numeric columns
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        X = X.values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        print(f"Original shape: {X.shape}")
        print(f"Classes: {np.unique(y_original)}")
        
        # Smart subsampling
        subsampler = SmartSubsampler(target_size=self.config.subsample_size)
        X_sub, y_sub = subsampler.subsample(X, y_original)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_sub)
        self.class_names = list(label_encoder.classes_)
        num_classes = len(self.class_names)
        
        print(f"Number of classes: {num_classes}")
        print(f"Class names: {self.class_names}")
        
        return X_sub, y_sub, y_encoded, num_classes, label_encoder
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray, 
                         time_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM."""
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X[i:(i + time_steps)])
            ys.append(y[i + time_steps - 1])
        return np.array(Xs), np.array(ys)
    
    def run_single_qubit_experiment(self, num_qubits: int, 
                                     X: np.ndarray, y: np.ndarray,
                                     y_encoded: np.ndarray, 
                                     num_classes: int) -> Dict:
        """Run experiment for a single qubit configuration."""
        print("\n" + "="*80)
        print(f"RUNNING EXPERIMENT: {num_qubits} QUBIT(S)")
        print("="*80)
        
        gc.collect()
        tf.keras.backend.clear_session()
        
        # 1. QPSO Feature Selection
        print(f"\n[Step 1] QPSO Feature Selection with {num_qubits} qubit(s)...")
        qpso = QuantumInspiredFeatureSelector(
            num_qubits=num_qubits,
            num_particles=self.config.num_particles,
            iterations=self.config.qpso_iterations,
            num_features_to_select=self.config.num_features_to_select
        )
        
        X_selected = qpso.fit_transform(X, y)
        qpso_time = qpso.computation_time
        qpso_memory = qpso.memory_usage
        
        # 2. Create sequences
        print(f"\n[Step 2] Creating sequences...")
        y_categorical = to_categorical(y_encoded, num_classes=num_classes)
        
        n_time_steps = self.config.sequence_length
        n_features = X_selected.shape[1]
        
        X_seq, y_seq = self.create_sequences(X_selected, y_categorical, n_time_steps)
        y_seq_labels = np.argmax(y_seq, axis=1)
        
        # Scale
        scaler = StandardScaler()
        X_seq_flat = X_seq.reshape(-1, n_features)
        X_seq_scaled = scaler.fit_transform(X_seq_flat).reshape(X_seq.shape)
        
        # Split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_seq_scaled, y_seq, test_size=0.3, random_state=42, 
            stratify=np.argmax(y_seq, axis=1)
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42,
            stratify=np.argmax(y_temp, axis=1)
        )
        
        input_shape = (n_time_steps, n_features)
        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # 3. Build and train model
        print(f"\n[Step 3] Building LSTM-VAE+SGD model...")
        model = LSTM_VAE_Model(
            input_shape=input_shape,
            num_classes=num_classes,
            lstm_units=self.config.lstm_units,
            attention_heads=self.config.attention_heads,
            dropout_rate=self.config.dropout_rate
        )
        model.build()
        model.compile_with_sgd(learning_rate=self.config.learning_rate)
        
        print(f"\n[Step 4] Training model...")
        model.train(
            X_train, y_train, X_val, y_val,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size
        )
        
        # 4. Evaluate
        print(f"\n[Step 5] Evaluating model...")
        y_pred, y_pred_prob = model.predict(X_test)
        y_true = np.argmax(y_test, axis=1)
        
        # 5. Calculate metrics
        metrics_calc = MetricsCalculator(self.class_names)
        metrics = metrics_calc.calculate_all_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_pred_prob=y_pred_prob,
            y_true_onehot=y_test,
            training_time=model.training_time,
            inference_time=model.inference_time,
            qpso_time=qpso_time,
            memory_mb=qpso_memory + model.memory_usage
        )
        
        # Add qubit-specific info
        metrics['num_qubits'] = num_qubits
        metrics['qpso_fitness'] = qpso.fitness_history[-1] if qpso.fitness_history else 0
        metrics['selected_features'] = len(qpso.best_features)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"RESULTS FOR {num_qubits} QUBIT(S)")
        print(f"{'='*60}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"MSE:       {metrics['mse']:.6f}")
        print(f"Total Time: {metrics['total_time_sec']:.2f}s")
        print(f"Memory:    {metrics['memory_mb']:.2f}MB")
        
        return metrics
    
    def run_full_ablation(self, csv_path: str) -> Dict:
        """Run complete ablation study across all qubit configurations."""
        print("\n" + "="*80)
        print("CONQUEST JOURNAL EXTENSION - QUBIT ABLATION STUDY")
        print("QPSO-LSTM-VAE+SGD Model")
        print("="*80)
        print(f"Qubit configurations to test: {self.config.qubit_sizes}")
        
        # Load data once
        X, y, y_encoded, num_classes, label_encoder = self.load_and_preprocess_data(csv_path)
        
        # Run experiments for each qubit configuration
        for num_qubits in self.config.qubit_sizes:
            try:
                metrics = self.run_single_qubit_experiment(
                    num_qubits, X, y, y_encoded, num_classes
                )
                self.results[num_qubits] = metrics
            except Exception as e:
                print(f"ERROR for {num_qubits} qubits: {e}")
                import traceback
                traceback.print_exc()
                self.results[num_qubits] = {'error': str(e)}
        
        # Generate summary
        self.generate_summary()
        
        return self.results
    
    def generate_summary(self):
        """Generate and print summary of ablation study."""
        print("\n" + "="*80)
        print("ABLATION STUDY SUMMARY")
        print("="*80)
        
        # Create summary table
        summary_data = []
        for num_qubits in sorted(self.results.keys()):
            r = self.results[num_qubits]
            if 'error' not in r:
                summary_data.append({
                    'Qubits': num_qubits,
                    'Accuracy': f"{r['accuracy']:.4f}",
                    'Precision': f"{r['precision']:.4f}",
                    'Recall': f"{r['recall']:.4f}",
                    'F1-Score': f"{r['f1_score']:.4f}",
                    'MSE': f"{r['mse']:.6f}",
                    'Time (s)': f"{r['total_time_sec']:.2f}",
                    'Memory (MB)': f"{r['memory_mb']:.2f}"
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            print("\n" + summary_df.to_string(index=False))
            
            # Find best configuration
            valid_results = {k: v for k, v in self.results.items() if 'error' not in v}
            if valid_results:
                best_acc_qubits = max(valid_results.keys(), 
                                      key=lambda q: valid_results[q]['accuracy'])
                best_f1_qubits = max(valid_results.keys(), 
                                     key=lambda q: valid_results[q]['f1_score'])
                
                print(f"\n{'='*60}")
                print("RECOMMENDATIONS")
                print(f"{'='*60}")
                print(f"Best Accuracy:  {best_acc_qubits} qubits ({valid_results[best_acc_qubits]['accuracy']:.4f})")
                print(f"Best F1-Score:  {best_f1_qubits} qubits ({valid_results[best_f1_qubits]['f1_score']:.4f})")
    
    def save_results(self, output_path: str = "qubit_ablation_results.json"):
        """Save results to JSON file."""
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj
        
        serializable_results = convert_to_serializable(self.results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    def generate_latex_table(self) -> str:
        """Generate LaTeX table for journal paper."""
        latex = r"""
\begin{table*}[htbp]
\centering
\caption{QPSO-LSTM-VAE+SGD Model Performance Across Different Qubit Configurations}
\label{tab:qubit_ablation}
\begin{tabular}{|c|c|c|c|c|c|c|c|}
\hline
\textbf{Qubits} & \textbf{Acc. (\%)} & \textbf{Prec. (\%)} & \textbf{Rec. (\%)} & \textbf{F1-Sc. (\%)} & \textbf{MSE} & \textbf{Time (s)} & \textbf{Memory (MB)} \\
\hline
"""
        for num_qubits in sorted(self.results.keys()):
            r = self.results[num_qubits]
            if 'error' not in r:
                latex += f"{num_qubits} & {r['accuracy']*100:.2f} & {r['precision']*100:.2f} & {r['recall']*100:.2f} & {r['f1_score']*100:.2f} & {r['mse']:.4f} & {r['total_time_sec']:.1f} & {r['memory_mb']:.1f} \\\\\n"
        
        latex += r"""\hline
\end{tabular}
\end{table*}
"""
        return latex


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_ablation_results(results: Dict, output_dir: str = "."):
    """Generate visualization plots for ablation study."""
    
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if not valid_results:
        print("No valid results to plot")
        return
    
    qubits = sorted(valid_results.keys())
    
    # Extract metrics
    accuracies = [valid_results[q]['accuracy'] for q in qubits]
    precisions = [valid_results[q]['precision'] for q in qubits]
    recalls = [valid_results[q]['recall'] for q in qubits]
    f1_scores = [valid_results[q]['f1_score'] for q in qubits]
    times = [valid_results[q]['total_time_sec'] for q in qubits]
    memories = [valid_results[q]['memory_mb'] for q in qubits]
    mses = [valid_results[q]['mse'] for q in qubits]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('QPSO-LSTM-VAE+SGD: Qubit Ablation Study', fontsize=14, fontweight='bold')
    
    # Plot 1: Classification Metrics
    ax1 = axes[0, 0]
    x = np.arange(len(qubits))
    width = 0.2
    ax1.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='#2ecc71')
    ax1.bar(x - 0.5*width, precisions, width, label='Precision', color='#3498db')
    ax1.bar(x + 0.5*width, recalls, width, label='Recall', color='#e74c3c')
    ax1.bar(x + 1.5*width, f1_scores, width, label='F1-Score', color='#9b59b6')
    ax1.set_xlabel('Number of Qubits')
    ax1.set_ylabel('Score')
    ax1.set_title('Classification Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(qubits)
    ax1.legend(loc='lower right')
    ax1.set_ylim([0.8, 1.0])
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Accuracy vs Qubits
    ax2 = axes[0, 1]
    ax2.plot(qubits, accuracies, 'o-', linewidth=2, markersize=10, color='#2ecc71')
    ax2.fill_between(qubits, accuracies, alpha=0.3, color='#2ecc71')
    ax2.set_xlabel('Number of Qubits')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Qubit Configuration')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Time Complexity
    ax3 = axes[0, 2]
    ax3.bar(qubits, times, color='#e67e22', edgecolor='black')
    ax3.set_xlabel('Number of Qubits')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Computational Time')
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Memory Complexity
    ax4 = axes[1, 0]
    ax4.bar(qubits, memories, color='#1abc9c', edgecolor='black')
    ax4.set_xlabel('Number of Qubits')
    ax4.set_ylabel('Memory (MB)')
    ax4.set_title('Memory Usage')
    ax4.grid(axis='y', alpha=0.3)
    
    # Plot 5: MSE
    ax5 = axes[1, 1]
    ax5.plot(qubits, mses, 's-', linewidth=2, markersize=10, color='#e74c3c')
    ax5.set_xlabel('Number of Qubits')
    ax5.set_ylabel('MSE')
    ax5.set_title('Mean Squared Error')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Trade-off Analysis (Accuracy vs Time)
    ax6 = axes[1, 2]
    scatter = ax6.scatter(times, accuracies, c=qubits, s=200, cmap='viridis', 
                          edgecolors='black', linewidth=2)
    for i, q in enumerate(qubits):
        ax6.annotate(f'{q}Q', (times[i], accuracies[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    ax6.set_xlabel('Time (seconds)')
    ax6.set_ylabel('Accuracy')
    ax6.set_title('Accuracy vs Time Trade-off')
    ax6.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label('Qubits')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'qubit_ablation_study.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'qubit_ablation_study.pdf'), bbox_inches='tight')
    plt.show()
    
    print(f"\nPlots saved to: {output_dir}")


def plot_per_class_metrics(results: Dict, output_dir: str = "."):
    """Plot per-class FAR and AUC-ROC."""
    
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if not valid_results:
        return
    
    # Get class names from first result
    first_result = list(valid_results.values())[0]
    class_names = list(first_result['far_per_class'].keys())
    qubits = sorted(valid_results.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Per-Class Metrics Across Qubit Configurations', fontsize=14, fontweight='bold')
    
    # Plot FAR per class
    ax1 = axes[0]
    x = np.arange(len(class_names))
    width = 0.15
    for i, q in enumerate(qubits):
        fars = [valid_results[q]['far_per_class'].get(c, 0) for c in class_names]
        ax1.bar(x + i*width, fars, width, label=f'{q}Q')
    ax1.set_xlabel('Attack Class')
    ax1.set_ylabel('False Alarm Rate')
    ax1.set_title('False Alarm Rate per Class')
    ax1.set_xticks(x + width * (len(qubits)-1) / 2)
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot AUC-ROC per class
    ax2 = axes[1]
    for i, q in enumerate(qubits):
        aucs = [valid_results[q]['auc_roc_per_class'].get(c, 0) for c in class_names]
        ax2.bar(x + i*width, aucs, width, label=f'{q}Q')
    ax2.set_xlabel('Attack Class')
    ax2.set_ylabel('AUC-ROC')
    ax2.set_title('AUC-ROC per Class')
    ax2.set_xticks(x + width * (len(qubits)-1) / 2)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.pdf'), bbox_inches='tight')
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    config = AblationConfig(
        qubit_sizes=[1, 2, 3, 4, 8],
        num_particles=20,
        qpso_iterations=30,
        num_features_to_select=20,
        subsample_size=50000,
        sequence_length=10,
        lstm_units=128,
        attention_heads=4,
        dropout_rate=0.3,
        epochs=50,
        batch_size=64,
        learning_rate=0.001
    )
    
    # Dataset path
    csv_path = "0.ACI-IoT-2023.csv"
    
    # Run ablation study
    study = QubitAblationStudy(config)
    results = study.run_full_ablation(csv_path)
    
    # Save results
    study.save_results("qubit_ablation_results.json")
    
    # Generate LaTeX table
    latex_table = study.generate_latex_table()
    print("\n" + "="*80)
    print("LATEX TABLE FOR JOURNAL")
    print("="*80)
    print(latex_table)
    
    # Save LaTeX table
    with open("qubit_ablation_table.tex", 'w') as f:
        f.write(latex_table)
    
    # Generate plots
    plot_ablation_results(results)
    plot_per_class_metrics(results)
    
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE")
    print("="*80)
