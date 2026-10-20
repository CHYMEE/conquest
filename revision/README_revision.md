# CONQUEST revision pipeline (leakage-free)

All code for the IEEE IoT-J revision lives in `revision/`. No existing script was modified.
The original `QuantumInspiredFeatureSelector`, `LSTM_VAE_Model`, and `MetricsCalculator` are
**imported** from `../qubit_ablation_study.py`, not copied.

| file | purpose |
|---|---|
| `common.py` | paths, seeding, TF thread configuration (must run before TF import), atomic JSON/CSV, `RSSMonitor` (psutil peak-RSS sampler) |
| `context.py` | the 17 causal context features: vectorised batch implementation + streaming `StreamingContext` (bit-identical); feature table in the docstring |
| `data.py` | CSV -> sorted full-stream cache (`cache/full_v1/*.npy`), context features, balanced subsample, chronological/blocked split, train-only MinMax, windowing; `selftest` = streaming == batch on real flows |
| `model.py` | `RevisedLSTMVAE(LSTM_VAE_Model)`: identical layer stack, KL regulariser (gamma = 1e-3, `add_loss`), deterministic inference (z = z_mean when `training=False`) |
| `feature_selectors.py` | `select(name, X_train, y_train, k, seed, ...)` for `qpso` (original class), `pso` (same budget), `variance`, `all` (named `feature_selectors.py`, not `selectors.py`, because a file called `selectors.py` would shadow the stdlib module that `subprocess` imports) |
| `centralized.py` | `run` (one experiment -> JSON + saved weights), `sweep` (grid driver, `--workers N`, resumable), `deploy` (re-measure deployment metrics in isolation from saved weights) |
| `federated.py` | `run` / `sweep` for FL (K clients, iid / label_skew / quantity_skew / dirichlet, fedavg / scaffold (real SCAFFOLD, option II) / flame (real FLAME: HDBSCAN + clipping + noise); legacy fedavgm / cosine_trust, mask modes proxy / vote / central), per-round per-client F1, R90, round-level checkpoint/resume |
| `stats.py` | aggregation: mean +- std, Wilcoxon (paired by seed), Friedman across Q, QPSO-vs-PSO, 2x2 context x selector, mask stability (Jaccard), feature frequency, Q* selection (validation F1), LaTeX |
| `grids/*.json` | grid specs for blocks A–E (+ `AB.json`, `CD.json` combined queues, smoke grids) |

## Fixed experimental protocol

* **Data**: ACI-IoT-2023, binary (Benign vs Attack). Rows are stable-sorted by Timestamp; equal timestamps keep file order.
  The CSV mixes three timestamp formats. The ISO-looking ones are actually **year-day-month** (`2023-01-11` = 1 Nov 2023), and this was verified against the epoch value hidden in `Idle Max` for 100 % of rows.
* **Features**: 79 CICFlowMeter columns minus `Idle Mean/Std/Max/Min`, giving 75 base features (ports and protocol kept as numeric).
  `Flow ID`, `Src IP`, `Dst IP`, `Timestamp` and `Connection Type` are dropped. IPs are used only as grouping keys.
  The `Idle*` columns hold absolute epoch microseconds, i.e. a wall-clock proxy for the attack sessions, so they are dropped by default (`keep_time_proxies=True` restores them for an ablation).
  With context on there are 75 + 17 = **92** features.
* **Subsample**: 40k benign + 40k attack, uniform within class, then re-sorted. The data seed is **fixed at 42** for every run.
* **Split**: the chronological 70/15/15 test split would be 100 % benign, so the rule (any class < 25 %) switches to the **blocked** split.
  This uses 40 blocks of 2000 rows shuffled with seed 42 into 28/6/6, and drops the first 10 rows of every block as a purge gap.
  See `cache/split_report_*.json` for the per-split class and attack-type counts.
* **Scaler**: MinMax fitted on train rows only, then clipped to [0, 1].
* **Windows**: T = 10, built inside each contiguous segment; the label is that of the last flow.
* **Run seed** (`--seed`, 42..51): controls QPSO/PSO, which is re-executed every run with no caching, plus model init and shuffling. Both seeds are stored in each JSON.
* **Model**: LSTM 256, 2 heads, dropout 0.2, latent 64, KL gamma 1e-3, SGD lr 1e-3 with momentum 0.9 and Nesterov, batch 32, at most 35 epochs.
  Training uses EarlyStopping(val_loss, patience 6, restore best) and ReduceLROnPlateau(val_loss, 0.5, patience 3, min 1e-6). The model has **788,514** trainable parameters with k = 40 (840,738 with all 92).
* **Q\***: argmax of the mean **validation** F1 over the block-A configuration (`python stats.py qstar --dir outputs/central`). In C/D/E grids, `"Q": "auto:outputs/central"` resolves it when the sweep starts.
* **FL**: R = 30, E = 2, full participation, batch 64, same model. The train split is cut into 250-row chunks, and a client never shares a flow with another client or with its own eval split.

## Preparation and checks (once)

```bash
cd revision
python data.py prepare            # ~25 s: full-stream cache + context features + subsample + split report
python data.py selftest           # streaming == batch context features on 3 x 5000 real flows (bit-identical)
python data.py table              # context-feature table (markdown)
```

## Smoke tests (tiny data; each finishes in 1–3 minutes)

```bash
python centralized.py sweep --grid grids/smoke_central.json   --workers 3 --threads 2
python federated.py   sweep --grid grids/smoke_federated.json --workers 3 --threads 2
python stats.py centralized --dir outputs/smoke/central   --out outputs/smoke/stats_central
python stats.py federated   --dir outputs/smoke/federated --out outputs/smoke/stats_fl
```

## Planned grid: exact commands

Run the blocks in this order. Q\* is needed by C, D and E, so block A must finish first.
All sweeps are resumable: re-running the same command skips finished runs, and FL resumes from the last completed round.
Use `--dry-run` to list pending runs.

```bash
cd revision
# 1) A + B in one queue (165 runs; 15 B configs are identical to A runs and are run once)
python centralized.py sweep --grid grids/AB.json --workers 3 --threads 2
python stats.py qstar --dir outputs/central                      # prints Q* (validation F1)
# 2) C + D (Q* resolved automatically from outputs/central)
python centralized.py sweep --grid grids/CD.json --workers 3 --threads 2
# 3) E (45 FL runs)
python federated.py sweep --grid grids/E.json --workers 3 --threads 2
# 4) deployment numbers measured in isolation (nothing else running), e.g. the Q* / variance / PSO runs
python centralized.py deploy --runs "outputs/central/qpso_ctxon_Q<Q*>_P20_I30_k40_s4[2-4].json" "outputs/central/variance_*_s4[2-4].json"
# 5) statistics + LaTeX
python stats.py centralized --dir outputs/central   --out outputs/stats/central --ref-q 6
python stats.py federated   --dir outputs/federated --out outputs/stats/federated
```

| block | content | runs (new) |
|---|---|---|
| A [R1.2, R1.4] | QPSO, Q = 1..8, 20 particles / 30 iters, context on, seeds 42–51 | 80 |
| B [R1.3] | Q in {6..10} x particles {20, 40} x iters {30, 60}, seeds 42–46 | 100 (85 new) |
| C [R1.6] | context {on, off} x selector {variance, QPSO Q\*}, seeds 42–51 | 40 (30 new) |
| D [R2.3] | PSO vs QPSO Q\*, same budget (20 x 31 = 620 fitness evaluations), seeds 42–51 | 20 (10 new) |
| E [R1.8 + FL] | K=5 iid / label / quantity / Dir(0.1, 0.3, 0.5, 1.0) fedavg; aggregators on quantity skew; K in {5, 10, 20}; mask modes proxy / vote / central on quantity skew and Dir(0.1); 3 seeds | 45 |

E2 compares **fedavg, scaffold and flame** on quantity skew (K=5, 3 seeds, R=30, E=2). The legacy phase-2 variants remain available as `fedavgm` (server momentum 0.9, formerly `scaffold_like`) and `cosine_trust` (formerly `flame`), but they are not in grid E.

E1–E3 use `mask_mode = central` (train-only QPSO at Q\*, the leakage-free equivalent of the previous protocol) so that only the FL factor varies. E4 compares the mask modes. Change `mask_mode` in `grids/E.json` if you want a deployable mode everywhere.

**Proxy set** (mask mode `proxy`): ~5 % of the train chunks, evenly spaced in time (every 20th chunk).
It is held out at the server and never given to clients, in all mask modes.
With the 500-row block split this is 11 chunks / 2,750 rows, 53.9 % attack: Benign 1,269, ICMP Flood 476, Port Scan 411, Vulnerability Scan 244, Ping Sweep 215, DNS Flood 135.
The earlier chronological-tail proxy was dropped, because it was 100 % benign and so gave zero Fisher fitness.

**Aggregator details** (see the `federated.py` docstring):

* `scaffold` follows Karimireddy et al. 2020, option II.
  * The corrected gradient g - c_i + c is applied in a custom train step *before* the SGD-momentum optimizer step.
  * Client update: c_i+ = c_i - c + (x - y_i)/(K_steps * lr_cv).
  * Server (lr 1, full participation): x = mean_i y_i and c += mean_i delta c_i, using uniform means as in the paper.
  * The control-variate update uses lr_cv = lr/(1-0.9) by default (`--scaffold-cv-lr momentum`), because SGD momentum amplifies the effective step about 10x.
    With the plain-lr formula (`--scaffold-cv-lr plain`), SCAFFOLD diverged to NaN on IID by round 4 in the smoke test.
    With lr/(1-beta) it tracked FedAvg (final F1 0.880 vs 0.893 after 6 rounds, n_sub 8000).
* `flame` follows Nguyen et al. 2022.
  * HDBSCAN (min_cluster_size = K//2+1, min_samples = 1, precomputed cosine distances of the updates, allow_single_cluster) selects the majority cluster.
  * Updates are clipped to the median L2 norm, then averaged uniformly.
  * Gaussian noise with sigma = lambda * median_norm is added, with lambda = 0.001 (`--flame-lambda`).
  * If HDBSCAN labels every client as noise, all clients are admitted, and this is logged in `agg_info`.

## Measured costs (this machine: 6 cores, no GPU, TF 2.20)

| measurement | value |
|---|---|
| full data prep (CSV -> cache, incl. context features on 1.23 M flows) | 25 s (context features alone 3.2 s) |
| QPSO 20 particles x 30 iters (620 Fisher evaluations) on 55.7k x 92 train rows | 35 s alone (PSO with the same budget: 30 s) |
| training epoch, 55.5k windows, batch 32, 1 process x 2 threads | 64–69 s |
| training epoch, **3 processes x 2 threads** in parallel | 123–127 s each (= 42 s per epoch of machine time) |
| training epoch, 4 processes x 1 thread | 158 s each (= 40 s per epoch; leaves only ~1 GB RAM free, not recommended) |
| peak RSS per process, training / inference | ~820 MB / ~770 MB |
| latency at B=1 (median / p95), isolated | 2.7 / 3.1 ms (3.6 / 4.5 ms under 3-process load, so use `deploy` for the paper) |
| throughput at B=64, isolated | ~4,600 windows/s (0.22 ms/window); T_infer_te (11,886 windows) ~3.0 s |
| L_C (17 context features + scaling + masking per new flow, streaming) | 0.015 ms median, 0.018 ms p95 |
| FL round, K=5, full data, 1 process x 2 threads | 88 s (about 167 s expected under 3-process load) |

Per-run estimates assume 3 workers x 2 threads. Epochs per run are unknown until A starts (cap 35, patience 6); the central estimate assumes 25 epochs, with a range of 15–35.

| block | new runs | wall per run (under load) | machine wall time |
|---|---|---|---|
| A | 80 | ~54 min (33–75) | ~24 h (15–33) |
| B | 85 | ~55 min (larger QPSO budgets add ~1–4 min) | ~26 h (16–36) |
| C | 30 | ~53 min | ~9 h (5.5–12.5) |
| D | 10 | ~54 min | ~3 h (2–4) |
| **A–D total** | 205 | | **~62 h (38–86), i.e. ~370 core-hours** |
| E (FL) | 45 | ~84 min (30 rounds x ~167 s; scaffold and flame cost the same per round as fedavg in smoke tests) | ~21 h (after A–D; slower if run concurrently with them) |
| **Grand total** | | | **~3.5 days** of continuous use of the 6-core machine |

Parallelisation plan: always use `--workers 3 --threads 2`.
RAM, not cores, is the binding constraint, since only 2–4 GB were free during the tests and each process needs ~0.8 GB.
Run the A+B queue first, then C+D, then E, and finally `deploy` with a single process on an idle machine.
To save time, B could use 3 seeds instead of 5 (B drops to ~16 h).
