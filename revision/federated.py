"""Federated CONQUEST experiments with per-client metrics (min-F1, sigma(F1), R90).

Single consistent FL configuration for ALL experiments (override only for smoke tests):
  R = 30 rounds, E = 2 local epochs, full participation, batch 64, SGD lr 1e-3 (momentum 0.9,
  Nesterov; momentum buffers reset at the start of each local update, as when the original
  script rebuilt the local model), same model as centralised (256 / 2 heads / 0.2 / latent 64,
  KL 1e-3, deterministic inference).

Data / partitioning (all from the TRAIN split of the fixed leakage-free split, data seed 42):
  * The train split is cut into contiguous *chunks* of ``--chunk-rows`` (default 250) flows
    inside each train segment. Windows of T=10 are built inside chunks only, so no two clients
    ever share a flow and no window crosses a chunk boundary.
  * Proxy set: ~5 % of the train chunks, evenly spaced in time (every 20th chunk by time,
    offset 10), held out at the server and NOT given to any client.  (The chronological tail
    of train was dropped: in this split it was 100 % benign, so the Fisher fitness was 0.)  They are held out in every mask mode, so that the clients' data
    are identical across proxy / vote / central.
  * Partitions of the remaining chunks over K clients (seeded):
      iid            random assignment of chunks (round-robin over a permutation)
      label_skew     chunks sorted by attack fraction, K contiguous shards (as original)
      quantity_skew  client shares ~ Dir(0.5 * 1_K) (as original), >= 2 chunks per client
      dirichlet      per majority-label class, client shares ~ Dir(alpha), >= 2 chunks/client
  * Each client's chunks are sorted chronologically; the last chunks covering >= 15 % of its
    rows become its local held-out eval split (never trained on).
  * Global test = the central test split.

Mask modes (feature selection, k features, Q qubits, QPSO particles/iterations):
  proxy    QPSO on the server proxy slice only
  vote     each client runs QPSO on its own local train rows; the server takes the top-k
           features by vote count (ties -> lower feature index); only indices are shared
  central  QPSO on the full train split (oracle reference)
The MinMax scaler is the train-split min/max (equivalently the element-wise min/max of
the clients' + proxy min/max vectors, which only needs 2*d numbers per client).

Aggregators (grid E uses fedavg / scaffold / flame):
  fedavg        size-weighted average of client models (McMahan et al.)
  scaffold      SCAFFOLD option II (Karimireddy et al. 2020): server c, client c_i (zero init); local
                gradient corrected g - c_i + c BEFORE the SGD-momentum step (custom train step);
                c_i+ = c_i - c + (x - y_i) / (K_steps * lr_cv); server lr 1:
                x <- x + mean_i(y_i - x), c <- c + (1/N) sum_i (c_i+ - c_i)  (full participation,
                uniform means as in the paper).  lr_cv = lr / (1 - 0.9) by default (--scaffold-cv-lr
                momentum): with SGD momentum beta the steady-state step is lr/(1-beta)*g, so the
                option-II estimate (x - y_i)/(K lr) would overestimate the gradient ~10x; with plain lr
                SCAFFOLD diverged (NaN) on IID in smoke tests. --scaffold-cv-lr plain = paper formula.
  flame         FLAME (Nguyen et al. 2022): HDBSCAN majority cluster on cosine distances of updates,
                median-norm clipping, uniform mean, Gaussian noise sigma = 0.001 * median norm
  fedavgm       legacy phase-2 "scaffold_like" (server momentum 0.9)       - not in grid E
  cosine_trust  legacy phase-2 "flame" (cosine-trust percentile filter)    - not in grid E

Per-round log: global test acc / F1 / AUC, per-client F1 of the GLOBAL model on each client's
local eval split -> min_k F1, mean, sigma(F1) (population std, ddof = 0).
R90 = first round whose global test F1 >= 0.9 * F1 at the final round.
Runs are resumable at round granularity (checkpoint after every round).

Usage:
  python federated.py run --partition quantity_skew --K 5 --aggregator fedavg --mask-mode proxy --seed 42
  python federated.py sweep --grid grids/E.json --workers 3 --threads 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import common
from common import OUT_DIR, REV_DIR, RSSMonitor, ensure_dir, read_json, write_csv, write_json

PARTITIONS = ["iid", "label_skew", "quantity_skew", "dirichlet"]
AGGREGATORS = ["fedavg", "scaffold", "flame", "fedavgm", "cosine_trust"]  # last two: legacy, not in E
MASK_MODES = ["proxy", "vote", "central"]


# ======================================================================================
# aggregators (ported unchanged from run_phase2_federated_heterogeneity_qpso_lstm_vae_sgd.py)
# ======================================================================================

def _fedavg(client_weights: List[List[np.ndarray]], client_sizes: List[int]) -> List[np.ndarray]:
    total = float(sum(client_sizes))
    out: List[np.ndarray] = []
    for layer_idx in range(len(client_weights[0])):
        layer = np.zeros_like(client_weights[0][layer_idx])
        for w, s in zip(client_weights, client_sizes):
            layer += (float(s) / total) * w[layer_idx]
        out.append(layer)
    return out


def _update_norms(global_weights: List[np.ndarray], client_weights: List[List[np.ndarray]]) -> List[float]:
    """L2 norm of each client's model update (w_k - w_global), over all layers."""
    return [float(np.sqrt(sum(float(np.sum((w[li] - global_weights[li]).astype(np.float64) ** 2))
                              for li in range(len(global_weights))))) for w in client_weights]


def _dp_fedavg(global_weights, client_weights, clip: float, noise_mult: float, rng: np.random.Generator):
    """Client-level DP-FedAvg (McMahan et al., ICLR 2018), full participation.

    Each update is clipped to L2 norm ``clip``; updates are averaged UNIFORMLY (fixed denominator K,
    so the sensitivity of the mean to one client is clip/K); Gaussian noise with standard deviation
    noise_mult * clip / K is added to every coordinate of the mean update."""
    K = len(client_weights)
    norms = _update_norms(global_weights, client_weights)
    mean_upd = [np.zeros_like(g, dtype=np.float64) for g in global_weights]
    for w, n in zip(client_weights, norms):
        scale = min(1.0, clip / n) if n > 0 else 1.0
        for li, g in enumerate(global_weights):
            mean_upd[li] += scale * (w[li] - g).astype(np.float64) / K
    sigma = noise_mult * clip / K
    new_w = [(g + mu + rng.normal(0.0, sigma, size=g.shape)).astype(g.dtype)
             for g, mu in zip(global_weights, mean_upd)]
    return new_w, {"dp_sigma": sigma, "dp_clip": clip, "dp_clipped_frac": float(np.mean([n > clip for n in norms]))}


def rdp_epsilon(noise_mult: float, rounds: int, delta: float) -> Dict[str, float]:
    """(epsilon, delta) for `rounds` adaptive compositions of the Gaussian mechanism with noise
    multiplier z (no client subsampling): RDP(alpha) = rounds * alpha / (2 z^2) (Mironov, 2017),
    converted with eps = min_alpha RDP(alpha) + log(1/delta)/(alpha - 1)."""
    if noise_mult <= 0:
        return {"epsilon": float("inf"), "delta": delta, "alpha": float("nan")}
    alphas = np.concatenate([np.linspace(1.01, 10, 900), np.linspace(10, 512, 2000)])
    eps = rounds * alphas / (2 * noise_mult ** 2) + np.log(1 / delta) / (alphas - 1)
    i = int(np.argmin(eps))
    return {"epsilon": float(eps[i]), "delta": delta, "alpha": float(alphas[i])}


def _fedavgm(prev_global_weights, client_weights, client_sizes, momentum: float = 0.9):
    # Legacy "scaffold_like" of the phase-2 script, renamed: server-side momentum on the FedAvg
    # model (no control variates). Kept only for reference; NOT used in grid E.
    avg = _fedavg(client_weights, client_sizes)
    if prev_global_weights is None:
        return avg
    return [momentum * w_prev + (1.0 - momentum) * w_avg for w_prev, w_avg in zip(prev_global_weights, avg)]


def _cosine_trust(client_weights, client_sizes, drop_percentile: float = 25.0):
    # Legacy "flame" of the phase-2 script, renamed: cosine-trust filtering only (drops the bottom
    # 25th percentile of mean cosine similarity). Kept only for reference; NOT used in grid E.
    flats = [np.concatenate([w.flatten() for w in ws]) for ws in client_weights]
    n = len(flats)
    if n <= 2:
        return _fedavg(client_weights, client_sizes), list(range(n))
    flats = np.asarray(flats)
    flats = flats / (np.linalg.norm(flats, axis=1, keepdims=True) + 1e-12)
    sims = flats @ flats.T
    np.fill_diagonal(sims, 0.0)
    trust = np.mean(sims, axis=1)
    thr = np.percentile(trust, drop_percentile)
    mask = trust >= thr
    trusted_w = [w for w, m in zip(client_weights, mask) if m]
    trusted_s = [s for s, m in zip(client_sizes, mask) if m]
    if len(trusted_w) == 0:
        return _fedavg(client_weights, client_sizes), list(range(n))
    return _fedavg(trusted_w, trusted_s), [i for i, m in enumerate(mask) if m]


def _flame_real(global_weights: List[np.ndarray], client_weights: List[List[np.ndarray]],
                noise_lambda: float, rng: np.random.Generator) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """FLAME (Nguyen et al., USENIX Security 2022), applied to client updates d_i = w_i - w_g:

      1. HDBSCAN (min_cluster_size = K//2 + 1, min_samples = 1, metric = precomputed,
         allow_single_cluster) on pairwise cosine distances 1 - cos(d_i, d_j); admit the largest
         cluster (the majority).  If HDBSCAN labels every client as noise, all clients are
         admitted (fallback, logged in agg_info).
      2. Adaptive clipping: S = median_i ||d_i||_2 over ALL clients; admitted updates are scaled
         by min(1, S / ||d_i||).
      3. Aggregate: w_g + uniform mean of the clipped admitted updates (as in FLAME).
      4. Adaptive noise: add N(0, sigma^2) to every parameter, sigma = lambda * S, lambda = 0.001.
    """
    from sklearn.cluster import HDBSCAN

    K = len(client_weights)
    shapes = [w.shape for w in global_weights]
    gflat = np.concatenate([w.ravel() for w in global_weights]).astype(np.float64)
    D = np.stack([np.concatenate([w.ravel() for w in ws]).astype(np.float64) - gflat for ws in client_weights])
    norms = np.linalg.norm(D, axis=1)
    U = D / (norms[:, None] + 1e-12)
    dist = np.clip(1.0 - U @ U.T, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    labels = HDBSCAN(min_cluster_size=K // 2 + 1, min_samples=1, metric="precomputed",
                     allow_single_cluster=True).fit_predict(dist)
    valid = labels[labels >= 0]
    if len(valid):
        major = np.bincount(valid).argmax()
        admitted = [i for i in range(K) if labels[i] == major]
        fallback = False
    else:
        admitted, fallback = list(range(K)), True
    S = float(np.median(norms))
    clipped = [D[i] * min(1.0, S / (norms[i] + 1e-12)) for i in admitted]
    sigma = noise_lambda * S
    new = gflat + np.mean(clipped, axis=0) + rng.normal(0.0, sigma, size=gflat.shape)
    out, o = [], 0
    for sh, w in zip(shapes, global_weights):
        n = int(np.prod(sh))
        out.append(new[o:o + n].reshape(sh).astype(w.dtype))
        o += n
    info = {"labels": labels.tolist(), "admitted": admitted, "fallback_all": fallback, "median_norm": S,
            "noise_sigma": sigma, "norms": norms.tolist()}
    return out, info


# ======================================================================================
# chunking / partitioning
# ======================================================================================

Seg = Tuple[int, int]


def make_chunks(train_segments: List[Seg], chunk_rows: int, T: int) -> List[Seg]:
    chunks: List[Seg] = []
    for s, e in sorted(train_segments):
        b = s
        while b < e:
            c_end = min(b + chunk_rows, e)
            if e - c_end < max(T, chunk_rows // 2):  # merge short tail
                c_end = e
            if c_end - b >= T:
                chunks.append((b, c_end))
            b = c_end
    return chunks


def _ensure_min(assign: List[List[int]], min_chunks: int, rng: np.random.Generator) -> List[List[int]]:
    for k in range(len(assign)):
        while len(assign[k]) < min_chunks:
            donor = int(np.argmax([len(a) for a in assign]))
            if len(assign[donor]) <= min_chunks:
                break
            assign[k].append(assign[donor].pop(int(rng.integers(len(assign[donor])))))
    return assign


def partition_chunks(chunks: List[Seg], y: np.ndarray, K: int, partition: str, alpha: float,
                     seed: int, min_chunks: int = 2) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    n = len(chunks)
    if n < K * min_chunks:
        raise ValueError(f"{n} chunks < K*min_chunks={K * min_chunks}; lower --chunk-rows")
    att = np.array([y[s:e].mean() for s, e in chunks])
    if partition == "iid":
        perm = rng.permutation(n)
        assign = [list(perm[k::K]) for k in range(K)]
    elif partition == "label_skew":
        order = np.lexsort((rng.random(n), att))
        assign = [list(a) for a in np.array_split(order, K)]
    elif partition == "quantity_skew":
        perm = list(rng.permutation(n))
        assign = [[perm.pop() for _ in range(min_chunks)] for _ in range(K)]
        props = rng.dirichlet(np.ones(K) * 0.5)
        rest = np.array(perm)
        cuts = (np.cumsum(props) * len(rest)).astype(int)[:-1]
        for k, part in enumerate(np.split(rest, cuts)):
            assign[k].extend(part.tolist())
    elif partition == "dirichlet":
        maj = (att >= 0.5).astype(int)
        assign = [[] for _ in range(K)]
        for c in (0, 1):
            idx = np.flatnonzero(maj == c)
            rng.shuffle(idx)
            props = rng.dirichlet([alpha] * K)
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for k, part in enumerate(np.split(idx, cuts)):
                assign[k].extend(part.tolist())
        assign = _ensure_min(assign, min_chunks, rng)
    else:
        raise ValueError(partition)
    return [sorted(int(i) for i in a) for a in assign]


def carve_eval(chunk_ids: List[int], chunks: List[Seg], frac: float) -> Tuple[List[int], List[int]]:
    """Chronologically last chunks covering >= frac of the client's rows -> eval."""
    ids = sorted(chunk_ids, key=lambda i: chunks[i][0])
    sizes = np.array([chunks[i][1] - chunks[i][0] for i in ids])
    total = sizes.sum()
    ev: List[int] = []
    acc = 0
    for i, sz in zip(reversed(ids), reversed(sizes)):
        if acc >= frac * total:
            break
        ev.append(i)
        acc += sz
    if len(ev) == len(ids):  # keep at least one train chunk
        ev = ev[:-1]
    tr = [i for i in ids if i not in ev]
    return tr, sorted(ev, key=lambda i: chunks[i][0])


# ======================================================================================
# run
# ======================================================================================

def run_name(a: argparse.Namespace) -> str:
    part = a.partition + (f"{a.alpha:g}" if a.partition == "dirichlet" else "")
    sel = f"{a.selector}Q{a.Q}" if a.selector == "qpso" else a.selector
    n = f"fl_{part}_K{a.K}_{a.aggregator}_{a.mask_mode}_{sel}_ctx{a.context}_s{a.seed}"
    if getattr(a, "dp_clip", 0) > 0:
        n += f"_dpz{a.dp_noise:g}_C{a.dp_clip:g}"
    return n + (f"_{a.tag}" if a.tag else "")


def _rows(segs: List[Seg]) -> np.ndarray:
    return np.concatenate([np.arange(s, e) for s, e in segs]) if segs else np.zeros(0, np.int64)


def run_single(a: argparse.Namespace) -> Dict[str, Any]:
    common.configure_threads(a.threads, a.inter_threads)
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    from data import load_dataset, make_windows, split_report
    from feature_selectors import select
    from model import build_model, count_params

    name = run_name(a)
    out_json = os.path.join(a.out_dir, name + ".json")
    ckpt = os.path.join(a.out_dir, "ckpt", name + ".npz")
    if os.path.exists(out_json) and not a.force:
        print(f"[skip] {out_json} exists")
        return read_json(out_json)
    common.set_seed(a.seed)
    t_start = time.time()

    ds = load_dataset(context=a.context == "on", n_sub=a.n_sub, data_seed=a.data_seed,
                      time_steps=a.time_steps, block_size=a.block_size)
    T = a.time_steps
    chunks = make_chunks(ds.segments("train"), a.chunk_rows, T)
    n_train_rows = sum(e - s for s, e in chunks)
    # proxy = evenly spaced train chunks with >= proxy_frac of train rows (held out everywhere)
    by_time = sorted(range(len(chunks)), key=lambda i: chunks[i][0])
    proxy_ids, acc = [], 0
    step = max(1, int(round(1.0 / a.proxy_frac)))  # evenly spaced chunks in time ("spread")
    sys_pick = by_time[step // 2::step]
    cand = sys_pick + [i for i in by_time if i not in set(sys_pick)]
    for i in cand:
        if acc >= a.proxy_frac * n_train_rows:
            break
        proxy_ids.append(i)
        acc += chunks[i][1] - chunks[i][0]
    pool = [i for i in range(len(chunks)) if i not in set(proxy_ids)]
    pool_chunks = [chunks[i] for i in pool]
    assign = partition_chunks(pool_chunks, ds.y, a.K, a.partition, a.alpha, a.seed, a.min_chunks)
    clients = []
    for k, ids in enumerate(assign):
        tr_ids, ev_ids = carve_eval(ids, pool_chunks, a.client_eval_frac)
        tr_segs = [pool_chunks[i] for i in tr_ids]
        ev_segs = [pool_chunks[i] for i in ev_ids]
        clients.append({"id": k, "train_segs": tr_segs, "eval_segs": ev_segs})
    proxy_segs = [chunks[i] for i in sorted(proxy_ids)]

    # ---------------- feature mask ----------------
    sel_kw = dict(Q=a.Q, particles=a.particles, iterations=a.iterations)
    mask_info: Dict[str, Any] = {"mode": a.mask_mode}
    t_sel = time.perf_counter()
    if a.mask_mode == "central":
        tr_rows = ds.rows("train")
        r = select(a.selector, ds.X[tr_rows], ds.y[tr_rows], k=a.k, seed=a.seed, **sel_kw)
        idx = r.indices
        mask_info["selection"] = {k: v for k, v in r.to_dict().items() if k != "fitness_history"}
    elif a.mask_mode == "proxy":
        pr = _rows(proxy_segs)
        r = select(a.selector, ds.X[pr], ds.y[pr], k=a.k, seed=a.seed, **sel_kw)
        idx = r.indices
        mask_info["selection"] = {k: v for k, v in r.to_dict().items() if k != "fitness_history"}
        mask_info["proxy_attack_frac"] = float(ds.y[pr].mean())
        mask_info["proxy_attack_types"] = {ds.label_names[int(t)]: int(c) for t, c in
                                           zip(*np.unique(ds.atk[pr], return_counts=True))}
    else:  # vote
        votes = np.zeros(ds.X.shape[1], dtype=np.int64)
        per_client = []
        for c in clients:
            cr = _rows(c["train_segs"])
            r = select(a.selector, ds.X[cr], ds.y[cr], k=a.k, seed=a.seed + 1000 * (c["id"] + 1), **sel_kw)
            votes[r.indices] += 1
            per_client.append({"client": c["id"], "indices": r.indices, "wall_time_s": r.wall_time_s,
                               "attack_frac": float(ds.y[cr].mean())})
        order = np.lexsort((np.arange(len(votes)), -votes))
        idx = sorted(int(i) for i in order[: a.k])
        mask_info.update(per_client=per_client, votes=votes.tolist())
    mask_info["selection_time_total_s"] = time.perf_counter() - t_sel
    mask_info["indices"] = idx
    mask_info["names"] = [ds.feature_names[i] for i in idx]

    Xs = ds.X[:, idx]
    for c in clients:
        c["Xtr"], c["ytr"], _ = make_windows(Xs, ds.y, c["train_segs"], T)
        c["Xev"], c["yev"], _ = make_windows(Xs, ds.y, c["eval_segs"], T)
        c["Ytr"] = tf.keras.utils.to_categorical(c["ytr"], 2).astype(np.float32)
    Xte, yte, _ = make_windows(Xs, ds.y, ds.segments("test"), T)
    client_summary = [{"client": c["id"], "n_train_windows": int(len(c["ytr"])),
                       "n_eval_windows": int(len(c["yev"])),
                       "train_attack_frac": float(c["ytr"].mean()) if len(c["ytr"]) else None,
                       "eval_attack_frac": float(c["yev"].mean()) if len(c["yev"]) else None,
                       "n_train_chunks": len(c["train_segs"]), "n_eval_chunks": len(c["eval_segs"])}
                      for c in clients]
    print(f"[fl] {name}: {len(chunks)} chunks ({len(proxy_ids)} proxy), clients:",
          [(s["n_train_windows"], s["n_eval_windows"], round(s["train_attack_frac"] or 0, 2)) for s in client_summary],
          flush=True)

    # ---------------- models ----------------
    common.set_seed(a.seed)
    shape = (T, len(idx))
    g = build_model(shape, 2, learning_rate=a.lr, kl_weight=a.kl_weight).model
    local = build_model(shape, 2, learning_rate=a.lr, kl_weight=a.kl_weight).model
    local.set_weights(g.get_weights())

    def reset_optimizer(m):
        for v in m.optimizer.variables:
            p = getattr(v, "path", getattr(v, "name", ""))
            if "momentum" in p or "iteration" in p:
                v.assign(tf.zeros_like(v))
        m.optimizer.learning_rate.assign(a.lr)

    # ---- SCAFFOLD: control variates + corrected custom train step ----
    tvars = local.trainable_variables
    corr_vars = [tf.Variable(tf.zeros_like(v), trainable=False) for v in tvars]  # holds c - c_i
    cce = tf.keras.losses.CategoricalCrossentropy()
    opt = local.optimizer

    @tf.function
    def scaffold_step(xb, yb):
        with tf.GradientTape() as tape:
            pred = local(xb, training=True)
            loss = cce(yb, pred)
            if local.losses:
                loss = loss + tf.add_n(local.losses)  # KL regulariser (add_loss)
        grads = tape.gradient(loss, tvars)
        grads = [gr + cv for gr, cv in zip(grads, corr_vars)]  # g - c_i + c
        opt.apply_gradients(zip(grads, tvars))
        return loss

    is_sc = a.aggregator == "scaffold"
    sc_c = [np.zeros(v.shape, np.float32) for v in tvars] if is_sc else None
    sc_ci = [[np.zeros(v.shape, np.float32) for v in tvars] for _ in clients] if is_sc else None
    lr_cv = a.lr if a.scaffold_cv_lr == "plain" else a.lr / (1.0 - 0.9)

    def scaffold_local(c, rng_l):
        """Local SCAFFOLD update for client c starting from the current local weights (= x)."""
        x_tr = [v.numpy() for v in tvars]
        for cv, gc, cic in zip(corr_vars, sc_c, sc_ci[c["id"]]):
            cv.assign(gc - cic)
        n = len(c["ytr"])
        steps = 0
        for _ep in range(a.local_epochs):
            perm = rng_l.permutation(n)
            for b in range(0, n, a.batch_size):
                bi = perm[b:b + a.batch_size]
                scaffold_step(tf.constant(c["Xtr"][bi]), tf.constant(c["Ytr"][bi]))
                steps += 1
        y_tr = [v.numpy() for v in tvars]
        new_ci = [cic - gc + (xv - yv) / (steps * lr_cv)
                  for cic, gc, xv, yv in zip(sc_ci[c["id"]], sc_c, x_tr, y_tr)]
        delta_c = [nc - oc for nc, oc in zip(new_ci, sc_ci[c["id"]])]
        sc_ci[c["id"]] = [v.astype(np.float32) for v in new_ci]
        return delta_c, steps

    def evaluate(Xw, yw) -> Dict[str, float]:
        if len(yw) == 0:
            return {"accuracy": float("nan"), "f1_weighted": float("nan"), "f1_attack": float("nan"),
                    "auc_attack": float("nan")}
        p = g.predict(Xw, batch_size=512, verbose=0)
        pred = p.argmax(1)
        out = {"accuracy": float(accuracy_score(yw, pred)),
               "f1_weighted": float(f1_score(yw, pred, average="weighted", zero_division=0)),
               "f1_attack": float(f1_score(yw, pred, pos_label=1, average="binary", zero_division=0))}
        try:
            out["auc_attack"] = float(roc_auc_score(yw, p[:, 1]))
        except ValueError:
            out["auc_attack"] = float("nan")
        return out

    rows: List[Dict[str, Any]] = []
    start_round = 0
    if os.path.exists(ckpt) and not a.force:
        with np.load(ckpt, allow_pickle=False) as z:  # context manager: Windows file lock
            g.set_weights([z[f"w{i}"] for i in range(int(z["n"]))])
            rows = json.loads(str(z["rows"]))
            if is_sc and "nv" in z:
                nv = int(z["nv"])
                sc_c = [z[f"c{j}"] for j in range(nv)]
                sc_ci = [[z[f"ci{k}_{j}"] for j in range(nv)] for k in range(len(clients))]
        start_round = len(rows)
        print(f"[fl] resumed from round {start_round}", flush=True)

    with RSSMonitor() as rss:
        for r in range(start_round, a.rounds):
            t_r = time.perf_counter()
            gw = g.get_weights()
            cws, csz, ctimes, dcs = [], [], [], []
            rng_l = np.random.default_rng(a.seed * 1000 + r)
            for c in clients:
                local.set_weights(gw)
                reset_optimizer(local)
                tc = time.perf_counter()
                if is_sc:
                    dc, _ = scaffold_local(c, rng_l)
                    dcs.append(dc)
                else:
                    local.fit(c["Xtr"], c["Ytr"], epochs=a.local_epochs, batch_size=a.batch_size, shuffle=True,
                              verbose=0)
                ctimes.append(time.perf_counter() - tc)
                cws.append(local.get_weights())
                csz.append(int(len(c["ytr"])))
            kept = list(range(len(clients)))
            agg_info: Dict[str, Any] = {}
            upd_norms = _update_norms(gw, cws)
            if a.dp_clip > 0:
                new_w, agg_info = _dp_fedavg(gw, cws, a.dp_clip, a.dp_noise,
                                             np.random.default_rng(a.seed * 104729 + r))
            elif a.aggregator == "flame":
                new_w, agg_info = _flame_real(gw, cws, a.flame_lambda, np.random.default_rng(a.seed * 7919 + r))
                kept = agg_info["admitted"]
            elif is_sc:
                # server lr 1, full participation: x <- mean_i y_i ; c <- c + (1/N) sum_i dc_i
                new_w = [np.mean([w[li] for w in cws], axis=0) for li in range(len(gw))]
                sc_c = [gc + np.mean([dc[j] for dc in dcs], axis=0).astype(np.float32) for j, gc in enumerate(sc_c)]
                agg_info = {"c_norm": float(np.sqrt(sum(float(np.sum(v ** 2)) for v in sc_c)))}
            elif a.aggregator == "fedavgm":
                new_w = _fedavgm(gw, cws, csz)
            elif a.aggregator == "cosine_trust":
                new_w, kept = _cosine_trust(cws, csz)
            else:
                new_w = _fedavg(cws, csz)
            agg_info["update_norms"] = upd_norms
            g.set_weights(new_w)
            train_t = time.perf_counter() - t_r
            gm = evaluate(Xte, yte)
            cf1, cacc = [], []
            for c in clients:
                cm = evaluate(c["Xev"], c["yev"])
                cf1.append(cm["f1_weighted"])
                cacc.append(cm["accuracy"])
            cf1a = np.asarray(cf1, dtype=float)
            row = {"round": r + 1, "test_accuracy": gm["accuracy"], "test_f1": gm["f1_weighted"],
                   "test_f1_attack": gm["f1_attack"], "test_auc": gm["auc_attack"],
                   "client_f1_min": float(np.nanmin(cf1a)), "client_f1_mean": float(np.nanmean(cf1a)),
                   "client_f1_std_pop": float(np.nanstd(cf1a, ddof=0)),
                   "client_f1": [float(v) for v in cf1], "client_acc": [float(v) for v in cacc],
                   "agg_kept": kept, "agg_info": agg_info, "round_train_time_s": train_t, "client_fit_times_s": ctimes,
                   "round_total_time_s": time.perf_counter() - t_r}
            rows.append(row)
            print(f"[fl] round {r + 1}/{a.rounds} test_f1={gm['f1_weighted']:.4f} auc={gm['auc_attack']:.4f} "
                  f"minF1={row['client_f1_min']:.4f} sd={row['client_f1_std_pop']:.4f} ({row['round_total_time_s']:.0f}s)",
                  flush=True)
            ensure_dir(os.path.dirname(ckpt))
            ww = g.get_weights()
            tmp = ckpt + ".tmp.npz"
            extra = {}
            if is_sc:
                extra["nv"] = len(sc_c)
                extra.update({f"c{j}": v for j, v in enumerate(sc_c)})
                extra.update({f"ci{k}_{j}": v for k, ci in enumerate(sc_ci) for j, v in enumerate(ci)})
            np.savez(tmp, n=len(ww), rows=json.dumps(rows), **{f"w{i}": w for i, w in enumerate(ww)}, **extra)
            os.replace(tmp, ckpt)

    f_final = rows[-1]["test_f1"]
    r90 = next((row["round"] for row in rows if row["test_f1"] >= 0.9 * f_final), None)
    res = {
        "run_name": name,
        "config": vars(a),
        "seeds": {"run_seed": a.seed, "data_seed": a.data_seed},
        "data": {**split_report(ds), "n_chunks": len(chunks), "chunk_rows": a.chunk_rows,
                 "proxy": {"n_chunks": len(proxy_ids), "n_rows": int(sum(e - s for s, e in proxy_segs)),
                           "attack_frac": float(ds.y[_rows(proxy_segs)].mean()),
                           "types": {ds.label_names[int(t)]: int(c) for t, c in
                                     zip(*np.unique(ds.atk[_rows(proxy_segs)], return_counts=True))}},
                 "clients": client_summary, "n_test_windows": int(len(yte))},
        "mask": mask_info,
        "dp": ({"mechanism": "client-level DP-FedAvg, uniform mean, Gaussian noise, full participation",
                "clip": a.dp_clip, "noise_multiplier": a.dp_noise, "rounds": a.rounds,
                **rdp_epsilon(a.dp_noise, a.rounds, a.dp_delta)} if a.dp_clip > 0 else None),
        "model": {"params": count_params(g)},
        "final": {"test_f1": f_final, "test_accuracy": rows[-1]["test_accuracy"], "test_auc": rows[-1]["test_auc"],
                  "test_f1_attack": rows[-1]["test_f1_attack"], "client_f1_min": rows[-1]["client_f1_min"],
                  "client_f1_mean": rows[-1]["client_f1_mean"], "client_f1_std_pop": rows[-1]["client_f1_std_pop"],
                  "R90": r90, "best_test_f1": max(rw["test_f1"] for rw in rows)},
        "rounds": rows,
        "timing": {"wall_total_s": time.time() - t_start,
                   "mean_round_time_s": float(np.mean([rw["round_total_time_s"] for rw in rows])),
                   "peak_rss_mb": rss.peak_mb},
        "env": {"tf": tf.__version__, "threads_intra": a.threads, "python": sys.version.split()[0]},
    }
    write_json(out_json, res)
    write_csv(os.path.join(a.out_dir, name + "_rounds.csv"),
              [{k: v for k, v in rw.items() if not isinstance(v, list)} for rw in rows])
    print(f"[done] {name}: final F1={f_final:.4f} R90={r90} minF1={res['final']['client_f1_min']:.4f}")
    return res


# ======================================================================================
# sweep
# ======================================================================================

FL_ARG_KEYS = ["partition", "alpha", "K", "aggregator", "mask_mode", "selector", "Q", "particles", "iterations",
               "k", "seed", "data_seed", "rounds", "local_epochs", "batch_size", "lr", "kl_weight", "context",
               "n_sub", "time_steps", "block_size", "chunk_rows", "proxy_frac", "client_eval_frac", "scaffold_cv_lr", "flame_lambda",
               "min_chunks", "dp_clip", "dp_noise", "dp_delta", "tag"]


def sweep(a: argparse.Namespace) -> None:
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    from centralized import expand_grid

    jobs = expand_grid(read_json(a.grid))
    parser = build_parser()
    todo = []
    for j in jobs:
        cmd = [sys.executable, os.path.join(REV_DIR, "federated.py"), "run", "--threads", str(a.threads),
               "--inter-threads", str(a.inter_threads)]
        for k, v in j.items():
            if k == "out_dir":
                cmd += ["--out-dir", v if os.path.isabs(v) else os.path.join(REV_DIR, v)]
            elif k in FL_ARG_KEYS and v is not None:
                cmd += [f"--{k.replace('_', '-')}", str(v)]
        ns = parser.parse_args(cmd[2:])
        if os.path.exists(os.path.join(ns.out_dir, run_name(ns) + ".json")):
            continue
        if any(t[0] == run_name(ns) and t[1] == ns.out_dir for t in todo):
            continue  # same run listed in several grid blocks: run it once
        todo.append((run_name(ns), ns.out_dir, cmd))
    print(f"[sweep] {len(jobs)} jobs, {len(todo)} to run, {a.workers} workers x {a.threads} threads")
    if a.dry_run:
        for _, _, c in todo:
            print(" ", " ".join(c[1:]))
        return

    def _run(item):
        n, od, cmd = item
        ensure_dir(os.path.join(od, "logs"))
        t0 = time.time()
        with open(os.path.join(od, "logs", n + ".log"), "a", encoding="utf-8") as lf:
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
        print(f"[sweep] {'ok ' if rc == 0 else 'ERR'} {n} ({time.time() - t0:.0f}s)", flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        rcs = list(ex.map(_run, todo))
    print(f"[sweep] finished: {sum(r == 0 for r in rcs)} ok, {sum(r != 0 for r in rcs)} failed")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--partition", choices=PARTITIONS, default="quantity_skew")
    r.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha (partition=dirichlet)")
    r.add_argument("--K", type=int, default=5)
    r.add_argument("--aggregator", choices=AGGREGATORS, default="fedavg")
    r.add_argument("--mask-mode", choices=MASK_MODES, default="proxy")
    r.add_argument("--selector", choices=["qpso", "pso", "variance", "all"], default="qpso")
    r.add_argument("--Q", type=int, default=6)
    r.add_argument("--particles", type=int, default=20)
    r.add_argument("--iterations", type=int, default=30)
    r.add_argument("--k", type=int, default=40)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--data-seed", type=int, default=common.DATA_SEED_DEFAULT)
    r.add_argument("--rounds", type=int, default=30)
    r.add_argument("--local-epochs", type=int, default=2)
    r.add_argument("--batch-size", type=int, default=64)
    r.add_argument("--lr", type=float, default=1e-3)
    r.add_argument("--kl-weight", type=float, default=1e-3)
    r.add_argument("--context", choices=["on", "off"], default="on")
    r.add_argument("--n-sub", type=int, default=80_000)
    r.add_argument("--time-steps", type=int, default=10)
    r.add_argument("--block-size", type=int, default=None)
    r.add_argument("--chunk-rows", type=int, default=250)
    r.add_argument("--proxy-frac", type=float, default=0.05)
    r.add_argument("--scaffold-cv-lr", choices=["plain", "momentum"], default="momentum",
                   help="lr in the SCAFFOLD c_i update: momentum = lr/(1-0.9) (default; plain lr diverged "
                        "to NaN on IID in smoke tests because heavy-ball momentum makes the per-step "
                        "displacement ~lr/(1-beta)*g), plain = lr (paper, plain SGD)")
    r.add_argument("--flame-lambda", type=float, default=0.001, help="FLAME noise level lambda")
    r.add_argument("--dp-clip", type=float, default=0.0, help="client-level DP-FedAvg clip norm C (0 = off)")
    r.add_argument("--dp-noise", type=float, default=0.0, help="DP noise multiplier z (sigma = z*C/K)")
    r.add_argument("--dp-delta", type=float, default=1e-5)
    r.add_argument("--client-eval-frac", type=float, default=0.15)
    r.add_argument("--min-chunks", type=int, default=2)
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--inter-threads", type=int, default=1)
    r.add_argument("--out-dir", default=os.path.join(OUT_DIR, "federated"))
    r.add_argument("--tag", default="")
    r.add_argument("--force", action="store_true")
    s = sp.add_parser("sweep")
    s.add_argument("--grid", required=True)
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--threads", type=int, default=2)
    s.add_argument("--inter-threads", type=int, default=1)
    s.add_argument("--dry-run", action="store_true")
    return ap


def main() -> None:
    a = build_parser().parse_args()
    run_single(a) if a.cmd == "run" else sweep(a)


if __name__ == "__main__":
    main()
