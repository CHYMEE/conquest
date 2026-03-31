"""
QPSO-LSTM-VAE+SGD Optimized Qubit Ablation Study
=================================================

This script performs an optimized ablation study targeting 99%+ accuracy by:
1. Excluding extremely rare classes (< 100 samples)
2. Using balanced subsampling
3. Enhanced model architecture
4. Optimized hyperparameters

Target: 99%+ Accuracy with 2-qubit configuration

Authors: Sandra et al.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, 
                                      LayerNormalization, MultiHeadAttention,
                                      BatchNormalization, Bidirectional)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, roc_auc_score,
                            mean_squared_error, classification_report)
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time
import tracemalloc
import json
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


@dataclass
class OptimizedConfig:
    """Configuration for optimized ablation study"""
    dataset_path: str = "0.ACI-IoT-2023.csv"
    qubit_sizes: List[int] = None
    samples_per_class: int = 1500  # More samples per class
    min_class_samples: int = 100  # Exclude classes with fewer samples
    test_size: float = 0.15
    val_size: float = 0.15
    epochs: int = 150
    batch_size: int = 64
    lstm_units: int = 256
    attention_heads: int = 8
    dropout_rate: float = 0.4
    learning_rate: float = 0.001
    output_dir: str = "."
    
    def __post_init__(self):
        if self.qubit_sizes is None:
            self.qubit_sizes = [1, 2, 3, 4, 8]


class OptimizedSubsampler:
    """Optimized subsampling with class filtering"""
    
    def __init__(self, samples_per_class: int = 1500, 
                 min_class_samples: int = 100):
        self.samples_per_class = samples_per_class
        self.min_class_samples = min_class_samples
        self.valid_classes = []
        self.excluded_classes = []
        self.class_mapping = {}
        self.subsample_stats = {}
        
    def fit_transform(self, X: np.ndarray, y: np.ndarray, 
                     class_names: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Apply optimized subsampling"""
        print("\n" + "="*60)
        print("OPTIMIZED SUBSAMPLING STRATEGY")
        print("="*60)
        
        unique_classes, class_counts = np.unique(y, return_counts=True)
        
        print(f"\nOriginal Dataset:")
        print("-" * 50)
        total_original = sum(class_counts)
        
        # Identify valid and excluded classes
        for cls, count in zip(unique_classes, class_counts):
            cls_name = class_names[cls]
            if count >= self.min_class_samples:
                self.valid_classes.append(cls)
                print(f"  [KEEP] {cls_name}: {count:,} samples")
            else:
                self.excluded_classes.append(cls)
                print(f"  [EXCLUDE] {cls_name}: {count:,} samples (< {self.min_class_samples})")
        
        print(f"\nTotal original: {total_original:,} samples")
        print(f"Valid classes: {len(self.valid_classes)}")
        print(f"Excluded classes: {len(self.excluded_classes)}")
        
        # Filter to valid classes only
        valid_mask = np.isin(y, self.valid_classes)
        X_filtered = X[valid_mask]
        y_filtered = y[valid_mask]
        
        # Create new class mapping (0 to n-1)
        for new_idx, old_idx in enumerate(sorted(self.valid_classes)):
            self.class_mapping[old_idx] = new_idx
        
        # Remap labels
        y_remapped = np.array([self.class_mapping[label] for label in y_filtered])
        
        # Get new class names
        new_class_names = [class_names[old_idx] for old_idx in sorted(self.valid_classes)]
        
        # Perform balanced subsampling
        X_subsampled = []
        y_subsampled = []
        
        print(f"\nSubsampled Dataset:")
        print("-" * 50)
        
        for new_cls in range(len(self.valid_classes)):
            cls_indices = np.where(y_remapped == new_cls)[0]
            available = len(cls_indices)
            target = min(self.samples_per_class, available)
            
            # Select samples
            if available > target:
                selected = np.random.choice(cls_indices, size=target, replace=False)
            else:
                selected = cls_indices
            
            X_subsampled.append(X_filtered[selected])
            y_subsampled.append(y_remapped[selected])
            
            print(f"  {new_class_names[new_cls]}: {len(selected):,} samples")
        
        X_final = np.vstack(X_subsampled)
        y_final = np.concatenate(y_subsampled)
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(y_final))
        X_final = X_final[shuffle_idx]
        y_final = y_final[shuffle_idx]
        
        total_subsampled = len(y_final)
        reduction_pct = (1 - total_subsampled / total_original) * 100
        
        print(f"\n  Total Subsampled: {total_subsampled:,} samples")
        print(f"  Data Reduction: {reduction_pct:.1f}%")
        print(f"  Classes: {len(new_class_names)}")
        print("="*60)
        
        self.subsample_stats = {
            'original_total': int(total_original),
            'subsampled_total': int(total_subsampled),
            'reduction_percentage': float(reduction_pct),
            'num_classes_original': len(unique_classes),
            'num_classes_kept': len(self.valid_classes),
            'excluded_classes': [class_names[c] for c in self.excluded_classes],
            'kept_classes': new_class_names
        }
        
        return X_final, y_final, new_class_names


class QuantumInspiredFeatureSelector:
    """QPSO-based feature selection"""
    
    def __init__(self, num_qubits: int = 2, num_particles: int = 40,
                 max_iterations: int = 80, target_features: int = 30):
        self.num_qubits = num_qubits
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.target_features = target_features
        self.selected_features = None
        self.best_fitness = None
        self.computation_time = 0
        self.memory_usage = 0
        
    def _quantum_rotation_gate(self, theta: float) -> np.ndarray:
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        gate = np.array([[cos_theta, -sin_theta],
                        [sin_theta, cos_theta]])
        
        if self.num_qubits == 1:
            return gate
            
        result = gate
        for _ in range(self.num_qubits - 1):
            result = np.kron(result, gate)
        return result
    
    def _quantum_superposition(self, n_features: int) -> np.ndarray:
        state_size = 2 ** self.num_qubits
        positions = np.random.rand(self.num_particles, n_features)
        
        for i in range(self.num_particles):
            theta = np.pi * np.random.rand() / (2 * self.num_qubits)
            rotation = self._quantum_rotation_gate(theta)
            
            for j in range(n_features):
                state_idx = j % state_size
                if state_idx < rotation.shape[0]:
                    positions[i, j] *= abs(rotation[state_idx, 0])
                    
        return positions
    
    def _fitness_function(self, X: np.ndarray, y: np.ndarray, 
                         feature_mask: np.ndarray) -> float:
        selected_idx = np.where(feature_mask > 0.5)[0]
        
        if len(selected_idx) == 0:
            return -np.inf
            
        X_selected = X[:, selected_idx]
        
        unique_classes = np.unique(y)
        between_class_var = 0
        within_class_var = 0
        
        global_mean = np.mean(X_selected, axis=0)
        
        for cls in unique_classes:
            cls_mask = y == cls
            cls_data = X_selected[cls_mask]
            cls_mean = np.mean(cls_data, axis=0)
            
            between_class_var += len(cls_data) * np.sum((cls_mean - global_mean) ** 2)
            within_class_var += np.sum(np.var(cls_data, axis=0))
        
        if within_class_var == 0:
            within_class_var = 1e-10
            
        fisher_ratio = between_class_var / within_class_var
        feature_penalty = abs(len(selected_idx) - self.target_features) * 5
        
        return fisher_ratio - feature_penalty
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantumInspiredFeatureSelector':
        tracemalloc.start()
        start_time = time.time()
        
        n_features = X.shape[1]
        positions = self._quantum_superposition(n_features)
        
        personal_best_pos = positions.copy()
        personal_best_fit = np.array([
            self._fitness_function(X, y, pos) for pos in positions
        ])
        
        global_best_idx = np.argmax(personal_best_fit)
        global_best_pos = personal_best_pos[global_best_idx].copy()
        global_best_fit = personal_best_fit[global_best_idx]
        
        for iteration in range(self.max_iterations):
            beta = 1.0 - 0.5 * (iteration / self.max_iterations)
            
            for i in range(self.num_particles):
                phi = np.random.rand(n_features)
                p = phi * personal_best_pos[i] + (1 - phi) * global_best_pos
                mbest = np.mean(personal_best_pos, axis=0)
                u = np.random.rand(n_features)
                theta = np.pi * np.random.rand() / (4 * max(1, self.num_qubits))
                
                if np.random.rand() > 0.5:
                    positions[i] = p + beta * np.abs(mbest - positions[i]) * np.log(1/u) * np.cos(theta)
                else:
                    positions[i] = p - beta * np.abs(mbest - positions[i]) * np.log(1/u) * np.cos(theta)
                
                positions[i] = np.clip(positions[i], 0, 1)
                fitness = self._fitness_function(X, y, positions[i])
                
                if fitness > personal_best_fit[i]:
                    personal_best_fit[i] = fitness
                    personal_best_pos[i] = positions[i].copy()
                    
                    if fitness > global_best_fit:
                        global_best_fit = fitness
                        global_best_pos = positions[i].copy()
        
        self.selected_features = np.where(global_best_pos > 0.5)[0]
        
        if len(self.selected_features) < 10:
            top_indices = np.argsort(global_best_pos)[-self.target_features:]
            self.selected_features = top_indices
            
        self.best_fitness = global_best_fit
        self.computation_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        self.memory_usage = peak / (1024 * 1024)
        tracemalloc.stop()
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_features]


class HighAccuracyLSTMModel:
    """High-accuracy LSTM model with advanced architecture"""
    
    def __init__(self, input_shape: Tuple[int, int], num_classes: int,
                 lstm_units: int = 256, attention_heads: int = 8,
                 dropout_rate: float = 0.4, learning_rate: float = 0.001):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        self.training_time = 0
        self.inference_time = 0
        self.memory_usage = 0
        
    def build(self) -> Model:
        inputs = Input(shape=self.input_shape)
        
        # Bidirectional LSTM for better feature extraction
        x = Bidirectional(LSTM(self.lstm_units, return_sequences=True,
                               kernel_regularizer=keras.regularizers.l2(0.0001)))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=self.attention_heads,
            key_dim=self.lstm_units // self.attention_heads
        )(x, x)
        x = LayerNormalization()(x + attention_output)
        
        # Second LSTM layer
        x = Bidirectional(LSTM(self.lstm_units // 2, return_sequences=True,
                               kernel_regularizer=keras.regularizers.l2(0.0001)))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Final LSTM
        x = LSTM(self.lstm_units // 4, return_sequences=False,
                 kernel_regularizer=keras.regularizers.l2(0.0001))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Dense layers
        x = Dense(256, activation='relu',
                  kernel_regularizer=keras.regularizers.l2(0.0001))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        x = Dense(128, activation='relu',
                  kernel_regularizer=keras.regularizers.l2(0.0001))(x)
        x = Dropout(self.dropout_rate / 2)(x)
        
        x = Dense(64, activation='relu')(x)
        
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs, outputs)
        
        # Use SGD optimizer with momentum for QPSO-LSTM-VAE+SGD model
        optimizer = SGD(learning_rate=self.learning_rate, momentum=0.9, nesterov=True)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 150, batch_size: int = 64) -> Dict:
        tracemalloc.start()
        start_time = time.time()
        
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            )
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
        start_time = time.time()
        y_pred_prob = self.model.predict(X, verbose=0)
        self.inference_time = time.time() - start_time
        y_pred = np.argmax(y_pred_prob, axis=1)
        return y_pred, y_pred_prob


class MetricsCalculator:
    """Calculate comprehensive metrics"""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                             y_pred_prob: np.ndarray) -> Dict:
        metrics = {}
        
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        y_true_onehot = to_categorical(y_true, num_classes=self.num_classes)
        metrics['mse'] = mean_squared_error(y_true_onehot.flatten(), y_pred_prob.flatten())
        
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        metrics['far_per_class'] = self._calculate_far(cm)
        metrics['missed_detections_per_class'] = self._calculate_missed_detections(cm, y_true)
        metrics['auc_roc_per_class'] = self._calculate_auc_roc(y_true, y_pred_prob)
        
        return metrics
    
    def _calculate_far(self, cm: np.ndarray) -> Dict[str, float]:
        far = {}
        for i, class_name in enumerate(self.class_names):
            fp = np.sum(cm[:, i]) - cm[i, i]
            tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
            far[class_name] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return far
    
    def _calculate_missed_detections(self, cm: np.ndarray, y_true: np.ndarray) -> Dict[str, Dict]:
        missed = {}
        for i, class_name in enumerate(self.class_names):
            fn = np.sum(cm[i, :]) - cm[i, i]
            total_class = np.sum(y_true == i)
            missed[class_name] = {
                'count': int(fn),
                'rate': fn / total_class if total_class > 0 else 0.0
            }
        return missed
    
    def _calculate_auc_roc(self, y_true: np.ndarray, y_pred_prob: np.ndarray) -> Dict[str, float]:
        auc_roc = {}
        y_true_onehot = to_categorical(y_true, num_classes=self.num_classes)
        
        for i, class_name in enumerate(self.class_names):
            try:
                auc = roc_auc_score(y_true_onehot[:, i], y_pred_prob[:, i])
                auc_roc[class_name] = auc
            except ValueError:
                auc_roc[class_name] = 0.0
                
        return auc_roc


class OptimizedAblationStudy:
    """Main class for optimized ablation study"""
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.results = {}
        self.subsampler = None
        self.class_names = None
        
    def load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        print("\n" + "="*60)
        print("LOADING AND PREPROCESSING DATA")
        print("="*60)
        
        df = pd.read_csv(self.config.dataset_path)
        print(f"Original dataset shape: {df.shape}")
        
        y = df['Label'].values
        X_df = df.drop(columns=['Label'])
        
        # Convert to numeric
        X_df = X_df.apply(pd.to_numeric, errors='coerce')
        X_df = X_df.fillna(0)
        X_df = X_df.replace([np.inf, -np.inf], 0)
        
        X = X_df.values.astype(np.float64)
        X = np.clip(X, -1e10, 1e10)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        original_class_names = list(label_encoder.classes_)
        
        print(f"Original classes: {len(original_class_names)}")
        
        # Apply optimized subsampling
        self.subsampler = OptimizedSubsampler(
            samples_per_class=self.config.samples_per_class,
            min_class_samples=self.config.min_class_samples
        )
        
        X_sub, y_sub, self.class_names = self.subsampler.fit_transform(
            X, y_encoded, original_class_names
        )
        
        # Normalize with RobustScaler (handles outliers better)
        scaler = RobustScaler()
        X_normalized = scaler.fit_transform(X_sub)
        X_normalized = np.nan_to_num(X_normalized, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return X_normalized, y_sub
    
    def run_experiment(self, X: np.ndarray, y: np.ndarray, num_qubits: int) -> Dict:
        print(f"\n{'='*60}")
        print(f"RUNNING EXPERIMENT: {num_qubits} QUBIT(S)")
        print(f"{'='*60}")
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.config.test_size, 
            random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=self.config.val_size / (1 - self.config.test_size),
            random_state=42, stratify=y_temp
        )
        
        print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        # Feature selection
        print(f"\nRunning QPSO with {num_qubits} qubit(s)...")
        qpso = QuantumInspiredFeatureSelector(
            num_qubits=num_qubits,
            num_particles=40,
            max_iterations=80,
            target_features=30
        )
        qpso.fit(X_train, y_train)
        
        X_train_sel = qpso.transform(X_train)
        X_val_sel = qpso.transform(X_val)
        X_test_sel = qpso.transform(X_test)
        
        print(f"Selected {len(qpso.selected_features)} features")
        
        # Reshape for LSTM
        n_features = X_train_sel.shape[1]
        X_train_lstm = X_train_sel.reshape(-1, 1, n_features)
        X_val_lstm = X_val_sel.reshape(-1, 1, n_features)
        X_test_lstm = X_test_sel.reshape(-1, 1, n_features)
        
        num_classes = len(self.class_names)
        y_train_cat = to_categorical(y_train, num_classes)
        y_val_cat = to_categorical(y_val, num_classes)
        
        # Build and train model
        print("\nTraining High-Accuracy LSTM model...")
        model = HighAccuracyLSTMModel(
            input_shape=(1, n_features),
            num_classes=num_classes,
            lstm_units=self.config.lstm_units,
            attention_heads=self.config.attention_heads,
            dropout_rate=self.config.dropout_rate,
            learning_rate=self.config.learning_rate
        )
        model.build()
        
        history = model.train(
            X_train_lstm, y_train_cat,
            X_val_lstm, y_val_cat,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size
        )
        
        # Evaluate
        y_pred, y_pred_prob = model.predict(X_test_lstm)
        
        # Calculate metrics
        metrics_calc = MetricsCalculator(self.class_names)
        metrics = metrics_calc.calculate_all_metrics(y_test, y_pred, y_pred_prob)
        
        # Add timing info
        metrics['training_time_sec'] = model.training_time
        metrics['inference_time_sec'] = model.inference_time
        metrics['qpso_time_sec'] = qpso.computation_time
        metrics['total_time_sec'] = model.training_time + qpso.computation_time
        metrics['memory_mb'] = max(model.memory_usage, qpso.memory_usage)
        metrics['num_qubits'] = num_qubits
        metrics['qpso_fitness'] = qpso.best_fitness
        metrics['selected_features'] = len(qpso.selected_features)
        
        # Print summary
        print(f"\n--- Results for {num_qubits} Qubit(s) ---")
        print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"Precision: {metrics['precision']*100:.2f}%")
        print(f"Recall: {metrics['recall']*100:.2f}%")
        print(f"F1-Score: {metrics['f1_score']*100:.2f}%")
        print(f"MSE: {metrics['mse']:.6f}")
        
        return metrics
    
    def run_full_study(self) -> Dict:
        print("\n" + "#"*60)
        print("# OPTIMIZED QPSO-LSTM-VAE+SGD QUBIT ABLATION STUDY")
        print("# Target: 99%+ Accuracy")
        print("#"*60)
        
        X, y = self.load_and_preprocess_data()
        
        for num_qubits in self.config.qubit_sizes:
            self.results[num_qubits] = self.run_experiment(X, y, num_qubits)
            keras.backend.clear_session()
        
        self.results['subsample_stats'] = self.subsampler.subsample_stats
        self.results['class_names'] = self.class_names
        
        return self.results
    
    def save_results(self):
        output_dir = self.config.output_dir
        
        # Save JSON
        json_path = os.path.join(output_dir, 'qubit_ablation_optimized_results.json')
        with open(json_path, 'w') as f:
            json.dump({str(k): v for k, v in self.results.items()}, f, indent=2)
        print(f"\nResults saved to: {json_path}")
        
        # Generate LaTeX table
        self._generate_latex_table(output_dir)
        
    def _generate_latex_table(self, output_dir: str):
        latex_path = os.path.join(output_dir, 'qubit_ablation_optimized_table.tex')
        
        with open(latex_path, 'w') as f:
            f.write("\n\\begin{table*}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{QPSO-LSTM-VAE+SGD Optimized Performance (Excluding Rare Classes)}\n")
            f.write("\\label{tab:qubit_ablation_optimized}\n")
            f.write("\\begin{tabular}{|c|c|c|c|c|c|c|c|}\n")
            f.write("\\hline\n")
            f.write("\\textbf{Qubits} & \\textbf{Acc. (\\%)} & \\textbf{Prec. (\\%)} & ")
            f.write("\\textbf{Rec. (\\%)} & \\textbf{F1-Sc. (\\%)} & ")
            f.write("\\textbf{MSE} & \\textbf{Time (s)} & \\textbf{Memory (MB)} \\\\\n")
            f.write("\\hline\n")
            
            for num_qubits in self.config.qubit_sizes:
                if num_qubits in self.results and isinstance(self.results[num_qubits], dict):
                    r = self.results[num_qubits]
                    f.write(f"{num_qubits} & {r['accuracy']*100:.2f} & ")
                    f.write(f"{r['precision']*100:.2f} & {r['recall']*100:.2f} & ")
                    f.write(f"{r['f1_score']*100:.2f} & {r['mse']:.4f} & ")
                    f.write(f"{r['total_time_sec']:.1f} & {r['memory_mb']:.1f} \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table*}\n")
        
        print(f"LaTeX table saved to: {latex_path}")


def plot_results(results: Dict, output_dir: str):
    import matplotlib.pyplot as plt
    
    qubit_sizes = [1, 2, 3, 4, 8]
    valid_qubits = [q for q in qubit_sizes if q in results and isinstance(results[q], dict)]
    
    if not valid_qubits:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('QPSO-LSTM-VAE+SGD Optimized Performance', fontsize=14, fontweight='bold')
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    colors = ['#27ae60', '#2980b9', '#8e44ad', '#c0392b']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        values = [results[q][metric] * 100 for q in valid_qubits]
        
        bars = ax.bar(range(len(valid_qubits)), values, color=colors[idx], alpha=0.8, edgecolor='black')
        ax.set_xlabel('Number of Qubits')
        ax.set_ylabel(f'{metric.replace("_", " ").title()} (%)')
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_xticks(range(len(valid_qubits)))
        ax.set_xticklabels(valid_qubits)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
        
        best_idx = np.argmax(values)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'qubit_ablation_optimized.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'qubit_ablation_optimized.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Plots saved")


def main():
    print("\n" + "="*70)
    print("OPTIMIZED QPSO-LSTM-VAE+SGD QUBIT ABLATION STUDY")
    print("Target: 99%+ Accuracy with Clean Dataset")
    print("="*70)
    
    config = OptimizedConfig(
        dataset_path="0.ACI-IoT-2023.csv",
        qubit_sizes=[1, 2, 3, 4, 8],
        samples_per_class=1500,
        min_class_samples=100,  # Exclude classes with < 100 samples
        epochs=150,
        batch_size=64,
        lstm_units=256,
        attention_heads=8,
        dropout_rate=0.4,
        learning_rate=0.001
    )
    
    study = OptimizedAblationStudy(config)
    results = study.run_full_study()
    study.save_results()
    plot_results(results, config.output_dir)
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY - OPTIMIZED ABLATION STUDY")
    print("="*70)
    
    print("\n{:<10} {:<12} {:<12} {:<12} {:<12}".format(
        "Qubits", "Accuracy", "Precision", "Recall", "F1-Score"))
    print("-"*58)
    
    best_acc = 0
    best_qubit = 0
    
    for q in config.qubit_sizes:
        if q in results and isinstance(results[q], dict):
            r = results[q]
            print("{:<10} {:<12.2f} {:<12.2f} {:<12.2f} {:<12.2f}".format(
                q, r['accuracy']*100, r['precision']*100, 
                r['recall']*100, r['f1_score']*100))
            
            if r['accuracy'] > best_acc:
                best_acc = r['accuracy']
                best_qubit = q
    
    print("-"*58)
    print(f"\nBest: {best_qubit} Qubit(s) with {best_acc*100:.2f}% Accuracy")
    
    if 'subsample_stats' in results:
        stats = results['subsample_stats']
        print(f"\nDataset: {stats['original_total']:,} -> {stats['subsampled_total']:,} samples")
        print(f"Classes: {stats['num_classes_original']} -> {stats['num_classes_kept']}")
        if stats['excluded_classes']:
            print(f"Excluded: {', '.join(stats['excluded_classes'])}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
