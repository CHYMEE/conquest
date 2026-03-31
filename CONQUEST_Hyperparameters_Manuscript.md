# CONQUEST Journal Extension: Hyperparameter Configuration

## For Manuscript Lock-In

---

## 1. QPSO Parameters

| Parameter | Symbol | Value | Notes |
|-----------|--------|-------|-------|
| **Particle count** | $J$ | 20 | Fixed across all experiments |
| **Iterations** | $T$ | 30 | Fixed across all experiments |
| **β schedule** | $\beta^{(t)}$ | **Annealed (linear decay)** | $\beta = 1.0 - 0.5 \times (t / T)$ |

### β Schedule Details
The contraction-expansion coefficient $\beta^{(t)}$ follows a **linear annealing schedule**:

$$\beta^{(t)} = 1.0 - 0.5 \times \frac{t}{T}$$

- **Initial value:** $\beta^{(0)} = 1.0$ (exploration)
- **Final value:** $\beta^{(T)} = 0.5$ (exploitation)
- **Behavior:** Linearly decreases from 1.0 to 0.5 over $T=30$ iterations

### Quantum Rotation Gate
The rotation angle $\theta$ is also annealed:

$$\theta^{(t)} = \frac{\pi}{2Q} \times \left(1 - \frac{t}{T}\right)$$

Where $Q$ = number of qubits (optimal: 6).

---

## 2. LSTM-VAE Architecture

| Component | Parameter | Value |
|-----------|-----------|-------|
| **LSTM Layer 1** | Units | 256 (Bidirectional → 512 total) |
| **LSTM Layer 2** | Units | 128 (Bidirectional → 256 total) |
| **Number of LSTM layers** | - | **2** (stacked) |
| **Hidden dimension** | $d_h$ | **256** (per direction) |
| **Attention heads** | $H$ | 8 |
| **Attention key dimension** | $d_k$ | $d_h / H = 32$ |
| **Latent dimension** | $d_z$ | **64** |
| **Dropout rate** | $p$ | 0.4 |
| **Sequence length** | $L$ | 10 (time steps) |

### Architecture Summary
```
Input → BiLSTM(256) → BatchNorm → MultiHeadAttention(8 heads) → LayerNorm 
      → BiLSTM(128) → BatchNorm → Dense(64, latent) → Dropout(0.4) 
      → Dense(128) → Dropout(0.4) → Dense(64) → Softmax(2)
```

### VAE Latent Space
- **Latent dimension:** 64
- **Reparameterization:** Standard VAE sampling with $z = \mu + \sigma \cdot \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$

---

## 3. Threshold Selection

| Aspect | Method | Details |
|--------|--------|---------|
| **Classification threshold** | **Fixed at 0.5** | Standard softmax argmax |
| **Percentile-based?** | **No** | Not used |
| **Validation-based optimization?** | **No** | Threshold not optimized |
| **ROC-based operating point?** | **No** | Standard decision boundary |

### Classification Decision Rule
The model uses **standard argmax classification** on softmax outputs:

$$\hat{y} = \arg\max_c \, p(y=c | x)$$

- **No threshold tuning** is performed
- **No ROC-based operating point selection**
- Classification is deterministic based on highest probability class

---

## 4. Training Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | SGD with Nesterov momentum |
| **Learning rate** | 0.001 |
| **Momentum** | 0.9 |
| **Batch size** | 32 (centralized) / 64 (federated) |
| **Epochs** | 35 (with early stopping) |
| **Early stopping patience** | 6 epochs |
| **LR reduction patience** | 3 epochs |
| **LR reduction factor** | 0.5 |
| **Minimum LR** | 1e-6 |

---

## 5. Feature Selection Configuration

| Parameter | Value |
|-----------|-------|
| **Number of features selected** | 40 |
| **Original features** | 83 (after preprocessing) |
| **Selection method** | QPSO (quantum-inspired) |
| **Fitness function** | Fisher criterion (between-class / within-class variance) |

---

## 6. Federated Learning Configuration

| Parameter | Value |
|-----------|-------|
| **Aggregator** | FedAvg |
| **Number of clients** | 5 |
| **Communication rounds** | 15-30 |
| **Local epochs** | 2 |
| **Client fraction** | 1.0 (full participation) |
| **FedProx μ** | 0.001 (when using FedProx) |

---

## Summary Table for Manuscript

### Table: Hyperparameter Configuration

| Category | Parameter | Value |
|----------|-----------|-------|
| **QPSO** | Particles $J$ | 20 |
| | Iterations $T$ | 30 |
| | $\beta$ schedule | Linear decay: $1.0 \to 0.5$ |
| | Optimal qubits $Q$ | 6 |
| **LSTM-VAE** | LSTM layers | 2 (stacked BiLSTM) |
| | Hidden dimension | 256 |
| | Latent dimension | 64 |
| | Attention heads | 8 |
| **Threshold** | Method | Fixed argmax (0.5) |
| | ROC optimization | Not applied |
| **Training** | Optimizer | SGD (Nesterov, μ=0.9) |
| | Learning rate | 0.001 |
| | Batch size | 32 |

---

*Ready for manuscript lock-in.*
