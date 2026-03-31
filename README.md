# CONQUEST: Context-Aware Quantum-Inspired Federated Learning for Mission-Critical IoV/IoD Cyber Threat Detection

A journal extension of a MILCOM conference paper that integrates **Quantum-Inspired Particle Swarm Optimization (QPSO)** feature selection, **Bidirectional LSTM-VAE with Multi-Head Attention**, and **Horizontal Federated Learning** for intrusion detection in Internet of Vehicles (IoV) and Internet of Drones (IoD) systems.

## Key Contributions

- **Variable-Qubit QPSO Feature Selection** — Systematic ablation across 1-8 qubits identifies 6 qubits as optimal, achieving 96.85% F1-score with minimal overhead (~19 seconds)
- **Privacy-Preserving Federated Learning** — FedAvg-based horizontal FL reaches 91.62% F1 without centralizing raw data, a ~2.9% trade-off vs. centralized training
- **Attention-Based LSTM-VAE Architecture** — Bidirectional LSTM with 8-head attention and VAE regularization captures temporal dependencies in network traffic
- **Comprehensive Heterogeneity Analysis** — Evaluation across IID, non-IID (label/quantity), and Dirichlet distributions for realistic deployment scenarios

## Architecture

```
Input (40 features × 10 timesteps)
  → BiLSTM(256) → BatchNorm → MultiHeadAttention(8 heads)
  → LayerNorm → BiLSTM(128) → BatchNorm
  → VAE Encoder (μ, σ → z ∈ R⁶⁴) → Dropout(0.4)
  → Dense(128) → Dropout(0.4) → Dense(64)
  → Softmax(2) → {Benign, Attack}
```

## Dataset

**ACI-IoT-2023** — An IoT cybersecurity dataset with 83 features and ~1.2M samples, used for binary classification (Benign vs. Attack). QPSO selects 40 optimal features from the original 83.

> The dataset file (`0.ACI-IoT-2023.csv`, ~625 MB) is excluded from this repo via `.gitignore`. Download it from the [ACI-IoT-2023 source](https://www.kaggle.com/datasets) and place it in the project root.

## Results Summary

| Phase | Configuration | Accuracy | F1-Score | AUC-ROC | Training Time |
|-------|--------------|----------|----------|---------|---------------|
| 1 | Non-Quantum Baseline (LSTM-VAE + SGD) | 94.64% | 94.64% | 0.9879 | 31.1 min |
| 2 | QPSO Ablation Best (6 qubits, 5-run mean) | 96.85% | 96.85% | 0.9947 | 35.0 min |
| 3 | QPSO Optimal (6 qubits, single run) | 94.49% | 94.49% | 0.9887 | 30.0 min |
| 4 | FL Aggregator Comparison (FedAvg) | 86.90% | 86.90% | 0.926 | 11.1 min |
| 5 | FL Heterogeneity Best (30 rounds) | 91.62% | 91.62% | 0.9708 | 100.1 min |

### Qubit Ablation (6 qubits is optimal)

| Qubits | F1-Score (mean ± std) | AUC-ROC |
|--------|-----------------------|---------|
| 1 | 92.85 ± 0.29% | 0.9749 |
| 3 | 95.65 ± 0.24% | 0.9908 |
| **6** | **96.85 ± 0.22%** | **0.9947** |
| 8 | 94.66 ± 0.82% | 0.9858 |

### Federated Learning Aggregator Comparison

| Aggregator | F1-Score | AUC | Notes |
|------------|----------|-----|-------|
| **FedAvg** | **86.9%** | **0.926** | Best performance under non-IID |
| Scaffold-like | 52.2% | 0.605 | Poor convergence |
| FLAME | 70.5% | 0.774 | Byzantine-robust but slower |

## Project Structure

```
conquest-milcom/
├── CONQUEST_Journal_Extension.py              # Core implementation (models, QPSO, FL)
├── CONQUEST_Hyperparameters_Manuscript.md      # Hyperparameter specifications
├── CONQUEST_Journal_Experimental_Results.md    # Detailed results documentation
│
├── # Phase Execution Scripts
├── run_phase1_lstm_vae_sgd_sweep.py           # Phase 1: Classical baseline sweep
├── run_qubit_ablation_improved_lstm_vae_sgd.py # Phase 2: Qubit ablation study
├── run_phase2_federated_heterogeneity_qpso_lstm_vae_sgd.py  # Phase 2: FL experiments
├── run_federated_lite.py                      # Lightweight FL runner
├── run_quick_tune_qpso_lstm_sgd.py            # Quick hyperparameter tuning
│
├── # Analysis & Visualization
├── make_classical_vs_fl_comparison.py         # Centralized vs FL comparison
├── make_phase2_federated_plots_tables.py      # FL result plots
├── make_plots_and_table_for_run.py            # General plotting utilities
├── plot_qubit_results.py                      # Qubit ablation visualizations
│
├── # Notebooks (EDA & Experiments)
├── 1.ACI-IOT-Analysis.ipynb                   # Dataset exploration
├── EDA-new.ipynb                              # Extended EDA
├── Conquest-lstm.ipynb                        # LSTM experiments
├── XAI-Context-lstm-adam.ipynb                # Explainability analysis
├── ...                                        # Additional experiment notebooks
│
├── outputs/                                   # Experiment artifacts
│   ├── phase1/                                # Baseline sweep results
│   ├── qubit_ablation_improved/               # Qubit ablation results
│   ├── phase2_federated_heterogeneity/        # FL heterogeneity results
│   ├── classical_vs_fl_comparison/            # Comparison outputs
│   └── quick_tune*/                           # Tuning results
│
└── *.pdf / *.png                              # Generated figures and reports
```

## Training Configuration

**Centralized Training:**
- Optimizer: SGD (LR=0.001, Nesterov momentum=0.9)
- Batch size: 32 | Epochs: 35 (early stopping, patience=6)
- LR schedule: ReduceOnPlateau (factor=0.5, patience=3, min=1e-6)
- Sequence length: 10 timesteps

**Federated Training:**
- Aggregator: FedAvg | Clients: 5 | Rounds: 15-30
- Local epochs: 2 | Batch size: 64 | Client fraction: 1.0

## Dependencies

- **TensorFlow / Keras** — LSTM-VAE, attention layers, training
- **Scikit-learn** — Preprocessing, metrics, cross-validation
- **NumPy / Pandas** — Data manipulation, quantum state operations
- **Matplotlib / Seaborn** — Visualization

> Note: QPSO and federated aggregation (FedAvg, Scaffold, FLAME) are implemented from scratch — no external quantum or FL frameworks required.

## Getting Started

```bash
# Clone the repository
git clone https://github.com/CHYMEE/conquest.git
cd conquest

# Install dependencies
pip install tensorflow scikit-learn numpy pandas matplotlib seaborn

# Place the ACI-IoT-2023 dataset in the project root
# Then run Phase 1 baseline:
python run_phase1_lstm_vae_sgd_sweep.py

# Run qubit ablation study:
python run_qubit_ablation_improved_lstm_vae_sgd.py

# Run federated learning experiments:
python run_phase2_federated_heterogeneity_qpso_lstm_vae_sgd.py

# Generate comparison plots:
python make_classical_vs_fl_comparison.py
```

## Mission-Critical Context

**IoV (Internet of Vehicles):** Raw driving/location data cannot leave vehicles due to privacy regulations (GDPR). Federated learning enables collaborative threat detection across millions of vehicles without centralizing sensitive data.

**IoD (Internet of Drones):** Military and surveillance data is classified. QPSO with lower qubit counts (3-6) reduces memory footprint for edge deployment on resource-constrained drone hardware. FLAME aggregation provides Byzantine robustness for adversarial settings.

## License

This project is part of academic research. Please cite appropriately if used in publications.
