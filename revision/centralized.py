"""Centralised CONQUEST experiments (leakage-free) + parallel sweep driver.

One run = (context on/off, selector, Q, particles, iterations, k, seed).
  * data subsample + split are FIXED by --data-seed (default 42); --seed controls QPSO/PSO
    (re-executed in every run - no caching across seeds), model init and batch shuffling.
  * selection fitted on TRAIN rows only; early stopping on val_loss (patience 6, restore best)
    + ReduceLROnPlateau(val_loss, 0.5, patience 3, min_lr 1e-6) as in the original ablation;
    TEST is used only once for the final metrics.
  * deployment measurements: per-window latency at B=1 (50 warm-up + 1000 timed calls,
    perf_counter), throughput at B=64, T_infer_te (model.predict over all test windows, B=64),
    peak RSS during training and during inference (separately), and L_C = incremental cost of
    computing the 17 context features + MinMax scaling + masking for one new flow (streaming
    updater, identical to the batch features; median/p95 ms per flow).

Usage:
  python centralized.py run --context on --selector qpso --Q 6 --seed 42 --out-dir outputs/A
  python centralized.py sweep --grid grids/A.json --workers 3 --threads 2
  python centralized.py deploy --runs "outputs/central/qpso_ctxon_Q6_*.json"   # isolated re-measure
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import numpy as np

import common
from common import OUT_DIR, REV_DIR, RSSMonitor, ensure_dir, read_json, write_json

CLASS_NAMES = ["Benign", "Attack"]  # label 1 = Attack


# ======================================================================================
# helpers
# ======================================================================================

def run_name(a: argparse.Namespace) -> str:
    sel = a.selector
    parts = [sel, f"ctx{a.context}"]
    if sel in ("qpso", "consensus"):
        parts.append(f"Q{a.Q}")
    if sel in ("qpso", "pso", "consensus"):
        parts += [f"P{a.particles}", f"I{a.iterations}"]
    if sel != "all":
        parts.append(f"k{a.k}")
    parts.append(f"s{a.seed}")
    if a.tag:
        parts.append(a.tag)
    return "_".join(parts)


def binary_metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    pred = prob.argmax(1)
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision_weighted": float(precision_score(y_true, pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "f1_attack": float(f1_score(y_true, pred, pos_label=1, average="binary", zero_division=0)),
        "precision_attack": float(precision_score(y_true, pred, pos_label=1, average="binary", zero_division=0)),
        "recall_attack": float(recall_score(y_true, pred, pos_label=1, average="binary", zero_division=0)),
    }
    try:
        out["auc_attack"] = float(roc_auc_score(y_true, prob[:, 1]))
    except ValueError:
        out["auc_attack"] = float("nan")
    return out


def per_attack_type_recall(atk_end: np.ndarray, pred: np.ndarray, label_names: List[str]) -> Dict[str, Any]:
    res = {}
    for a in np.unique(atk_end):
        m = atk_end == a
        name = label_names[int(a)]
        # for Benign this is specificity (1-FAR), for attacks it is the detection rate
        res[name] = {"n": int(m.sum()), "frac_pred_attack": float((pred[m] == 1).mean())}
    return res


def measure_context_cost(ds, mask: List[int], n_timed: int = 5000, warm_flows: int = 200) -> Dict[str, Any]:
    """L_C: per-flow cost of streaming context features + MinMax scaling + masking.

    Replays the FULL sorted stream from 60 s before the first test flow (state warm-up, not
    timed), then times ``n_timed`` consecutive flows (after ``warm_flows`` untimed flows).
    """
    from context import StreamingContext
    from data import load_full

    full = load_full()
    names = full["meta"]["base_feature_names"]
    ci = {c: names.index(c) for c in names}
    keep = ds.extra["base_cols_kept"]
    t_full = full["t"]
    first_test_full = int(ds.full_idx[ds.segments("test")[0][0]])
    t0 = int(t_full[first_test_full])
    start = int(np.searchsorted(t_full, t0 - 60, side="left"))
    end = min(len(t_full), first_test_full + warm_flows + n_timed)
    Xb = np.asarray(full["X_base"][start:end], dtype=np.float64)
    ts = np.asarray(t_full[start:end])
    src = np.asarray(full["src"][start:end])
    dst = np.asarray(full["dst"][start:end])
    dport = Xb[:, ci["Dst Port"]].astype(np.int64)
    proto = Xb[:, ci["Protocol"]].astype(np.int64)
    syn = Xb[:, ci["SYN Flag Count"]]
    pk = Xb[:, ci["Total Fwd Packet"]] + Xb[:, ci["Total Bwd packets"]]
    by = Xb[:, ci["Total Length of Fwd Packet"]] + Xb[:, ci["Total Length of Bwd Packet"]]
    Xk = Xb[:, keep]
    mn, sc = ds.scaler_min, ds.scaler_scale
    nonconst = np.asarray(ds.extra.get("nonconst_cols", np.arange(len(mn))), dtype=np.int64)
    mask_arr = np.asarray(mask, dtype=np.int64)
    sctx = StreamingContext()
    timed_from = (first_test_full - start) + warm_flows
    times = []
    for i in range(end - start):
        a = time.perf_counter()
        if ds.context:
            c = sctx.update(int(ts[i]), int(src[i]), int(dst[i]), int(dport[i]), int(proto[i]),
                            float(syn[i]), float(pk[i]), float(by[i]))
            x = np.concatenate([Xk[i], c])
        else:
            x = Xk[i]
        x = x[nonconst]
        xs = np.clip((x - mn) * sc, 0.0, 1.0)
        _ = xs[mask_arr]
        b = time.perf_counter()
        if i >= timed_from:
            times.append((b - a) * 1e3)
    tt = np.asarray(times)
    return {"what": "context(17)+minmax+mask per flow" if ds.context else "minmax+mask per flow (context off)",
            "n_timed": int(len(tt)), "n_untimed_warmup": int(timed_from),
            "median_ms": float(np.median(tt)) if len(tt) else None,
            "p95_ms": float(np.percentile(tt, 95)) if len(tt) else None,
            "mean_ms": float(np.mean(tt)) if len(tt) else None,
            "n_active_src_states": len(sctx.state)}


def measure_inference(model, X_test: np.ndarray, n_latency: int = 1000, warmup: int = 50,
                      batch: int = 64) -> Dict[str, Any]:
    import tensorflow as tf

    T, d = X_test.shape[1], X_test.shape[2]
    f1 = tf.function(lambda x: model(x, training=False),
                     input_signature=[tf.TensorSpec([1, T, d], tf.float32)])
    fB = tf.function(lambda x: model(x, training=False),
                     input_signature=[tf.TensorSpec([None, T, d], tf.float32)])
    n = len(X_test)
    res: Dict[str, Any] = {}
    with RSSMonitor() as rss:
        # (a) latency, batch size 1
        for i in range(warmup):
            f1(tf.constant(X_test[i % n: i % n + 1])).numpy()
        lat = []
        for i in range(n_latency):
            x = tf.constant(X_test[i % n: i % n + 1])
            a = time.perf_counter()
            f1(x).numpy()
            lat.append((time.perf_counter() - a) * 1e3)
        lat = np.asarray(lat)
        res["latency_b1"] = {"n_warmup": warmup, "n_timed": n_latency, "median_ms": float(np.median(lat)),
                             "p95_ms": float(np.percentile(lat, 95)), "mean_ms": float(lat.mean())}
        # (b) throughput, B = 64 (tf.function path)
        for i in range(3):
            fB(tf.constant(X_test[:batch])).numpy()
        a = time.perf_counter()
        for s in range(0, n, batch):
            fB(tf.constant(X_test[s:s + batch])).numpy()
        el = time.perf_counter() - a
        res["throughput_b64"] = {"samples_per_s": float(n / el), "ms_per_sample": float(el * 1e3 / n),
                                 "n_samples": int(n), "total_s": float(el)}
        # (c) T_infer_te: Keras predict over the whole test set at B = 64
        model.predict(X_test[:batch], batch_size=batch, verbose=0)
        a = time.perf_counter()
        model.predict(X_test, batch_size=batch, verbose=0)
        res["T_infer_te_s"] = float(time.perf_counter() - a)
    res["peak_rss_inference_mb"] = rss.peak_mb
    return res


# ======================================================================================
# single run
# ======================================================================================

def run_single(a: argparse.Namespace) -> Dict[str, Any]:
    common.configure_threads(a.threads, a.inter_threads)
    import tensorflow as tf

    from data import load_dataset, make_windows, split_report
    from feature_selectors import select
    from model import build_model, count_params, optimizer_summary
    from qubit_ablation_study import MetricsCalculator

    name = run_name(a)
    out_path = os.path.join(a.out_dir, name + ".json")
    if os.path.exists(out_path) and not a.force:
        print(f"[skip] {out_path} exists")
        return read_json(out_path)
    t_start = time.time()
    common.set_seed(a.seed)

    ds = load_dataset(context=a.context == "on", n_sub=a.n_sub, data_seed=a.data_seed,
                      time_steps=a.time_steps, block_size=a.block_size)
    tr = ds.rows("train")
    sel_params = dict(Q=a.Q, particles=a.particles, iterations=a.iterations)
    sres = select(a.selector, ds.X[tr], ds.y[tr], k=a.k, seed=a.seed, **sel_params)
    idx = sres.indices
    Xs = ds.X[:, idx]
    Xtr, ytr, _ = make_windows(Xs, ds.y, ds.segments("train"), a.time_steps)
    Xva, yva, _ = make_windows(Xs, ds.y, ds.segments("val"), a.time_steps)
    Xte, yte, end_te = make_windows(Xs, ds.y, ds.segments("test"), a.time_steps)
    if a.max_train_windows and len(Xtr) > a.max_train_windows:  # smoke tests only
        Xtr, ytr = Xtr[: a.max_train_windows], ytr[: a.max_train_windows]
    Ytr = tf.keras.utils.to_categorical(ytr, 2).astype(np.float32)
    Yva = tf.keras.utils.to_categorical(yva, 2).astype(np.float32)
    Yte = tf.keras.utils.to_categorical(yte, 2).astype(np.float32)

    common.set_seed(a.seed)  # model init independent of selector RNG consumption
    m = build_model((a.time_steps, len(idx)), 2, learning_rate=a.lr, kl_weight=a.kl_weight)
    model = m.model

    class EpochTimer(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.times = []

        def on_epoch_begin(self, epoch, logs=None):
            self._t = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            self.times.append(time.perf_counter() - self._t)

    timer = EpochTimer()
    cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=a.patience, restore_best_weights=True,
                                            mode="min"),
           tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(1, a.patience // 2),
                                                min_lr=1e-6),
           timer]
    with RSSMonitor() as rss_train:
        t0 = time.perf_counter()
        hist = model.fit(Xtr, Ytr, validation_data=(Xva, Yva), epochs=a.epochs, batch_size=a.batch_size,
                         shuffle=True, verbose=a.verbose, callbacks=cbs)
        train_time = time.perf_counter() - t0
    h = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    best_epoch = int(np.argmin(h["val_loss"])) + 1

    p_va = model.predict(Xva, batch_size=256, verbose=0)
    val_m = binary_metrics(yva, p_va)
    val_m["loss_best"] = float(np.min(h["val_loss"]))

    t_inf0 = time.perf_counter()
    p_te = model.predict(Xte, batch_size=256, verbose=0)
    t_inf = time.perf_counter() - t_inf0
    mc = MetricsCalculator(CLASS_NAMES).calculate_all_metrics(
        y_true=yte, y_pred=p_te.argmax(1), y_pred_prob=p_te, y_true_onehot=Yte, training_time=train_time,
        inference_time=t_inf, qpso_time=sres.wall_time_s, memory_mb=0.0)
    mc.pop("memory_mb", None)
    test_m = {**binary_metrics(yte, p_te), "mse": mc["mse"], "far_per_class": mc["far_per_class"],
              "missed_detections_per_class": mc["missed_detections_per_class"],
              "auc_roc_per_class": mc["auc_roc_per_class"], "confusion_matrix": mc["confusion_matrix"],
              "per_attack_type": per_attack_type_recall(ds.atk[end_te], p_te.argmax(1), ds.label_names),
              "metrics_calculator_raw": {k: mc[k] for k in ("accuracy", "precision", "recall", "f1_score")}}

    if not a.no_save_model:
        ensure_dir(os.path.join(a.out_dir, "models"))
        model.save_weights(os.path.join(a.out_dir, "models", name + ".weights.h5"))
    deploy = measure_inference(model, Xte, n_latency=a.latency_n, warmup=a.latency_warmup)
    deploy["peak_rss_training_mb"] = rss_train.peak_mb
    if not a.skip_lc:
        deploy["L_C_per_flow"] = measure_context_cost(ds, idx, n_timed=a.lc_n)
    deploy["note"] = ("latency/throughput via tf.function(model(x, training=False)); T_infer_te via "
                      "model.predict(batch_size=64)")

    res = {
        "run_name": name,
        "config": {k: v for k, v in vars(a).items() if k not in ("func",)},
        "seeds": {"run_seed": a.seed, "data_seed": a.data_seed},
        "data": {**split_report(ds), "n_features_total": len(ds.feature_names),
                 "n_windows": {"train": int(len(ytr)), "val": int(len(yva)), "test": int(len(yte))}},
        "selection": {"selector": a.selector, "k": len(idx), "indices": idx,
                      "names": [ds.feature_names[i] for i in idx],
                      "n_context_selected": int(sum(1 for i in idx if i >= ds.extra["n_base"])),
                      "dropped_constant": ds.extra.get("dropped_constant", []),
                      **{k: v for k, v in sres.to_dict().items() if k != "indices"}},
        "model": {"params": count_params(model), "optimizer": optimizer_summary(model),
                  "kl_weight": a.kl_weight, "lstm_units": 256, "attention_heads": 2, "dropout": 0.2,
                  "latent_dim": 64},
        "training": {"train_time_s": train_time, "epochs_run": len(h["loss"]), "best_epoch": best_epoch,
                     "epoch_times_s": timer.times, "mean_epoch_time_s": float(np.mean(timer.times)),
                     "history": h},
        "val_metrics": val_m,
        "test_metrics": test_m,
        "deployment": deploy,
        "timing": {"selection_time_s": sres.wall_time_s, "train_time_s": train_time,
                   "test_predict_b256_s": t_inf, "wall_total_s": time.time() - t_start},
        "env": {"tf": tf.__version__, "threads_intra": a.threads, "threads_inter": a.inter_threads,
                "python": sys.version.split()[0]},
    }
    write_json(out_path, res)
    print(f"[done] {name}: val_f1={val_m['f1_weighted']:.4f} test_f1={test_m['f1_weighted']:.4f} "
          f"epochs={len(h['loss'])} train={train_time:.0f}s -> {out_path}")
    return res


def redeploy(a: argparse.Namespace) -> None:
    """Re-measure deployment metrics in isolation (run with nothing else on the machine) from
    saved weights; results are stored under ``deployment_isolated`` in each run JSON."""
    common.configure_threads(a.threads, a.inter_threads)
    import glob as _glob

    from data import load_dataset, make_windows
    from model import build_model

    files = []
    for pat in a.runs:
        files += sorted(_glob.glob(pat))
    for f in files:
        r = read_json(f)
        if "deployment_isolated" in r and not a.force:
            continue
        c = r["config"]
        wpath = os.path.join(os.path.dirname(f), "models", r["run_name"] + ".weights.h5")
        if not os.path.exists(wpath):
            print(f"[deploy] no weights for {r['run_name']}")
            continue
        ds = load_dataset(context=c["context"] == "on", n_sub=c["n_sub"], data_seed=c["data_seed"],
                          time_steps=c["time_steps"], block_size=c.get("block_size"))
        idx = r["selection"]["indices"]
        Xte, _, _ = make_windows(ds.X[:, idx], ds.y, ds.segments("test"), c["time_steps"])
        m = build_model((c["time_steps"], len(idx)), 2, learning_rate=c["lr"], kl_weight=c["kl_weight"])
        m.model.load_weights(wpath)
        dep = measure_inference(m.model, Xte, n_latency=a.latency_n, warmup=a.latency_warmup)
        dep["L_C_per_flow"] = measure_context_cost(ds, idx, n_timed=a.lc_n)
        dep["threads_intra"] = a.threads
        r["deployment_isolated"] = dep
        write_json(f, r)
        print(f"[deploy] {r['run_name']}: B1 median {dep['latency_b1']['median_ms']:.2f} ms, "
              f"p95 {dep['latency_b1']['p95_ms']:.2f} ms, B64 {dep['throughput_b64']['samples_per_s']:.0f}/s, "
              f"L_C {dep['L_C_per_flow']['median_ms']:.4f} ms")


# ======================================================================================
# sweep driver
# ======================================================================================

ARG_KEYS = ["context", "selector", "Q", "particles", "iterations", "k", "seed", "data_seed", "epochs",
            "patience", "batch_size", "lr", "kl_weight", "n_sub", "time_steps", "block_size",
            "max_train_windows", "latency_n", "latency_warmup", "lc_n", "skip_lc", "tag"]


def _resolve_qstar(spec: Any) -> Any:
    if isinstance(spec, str) and spec.startswith("auto:"):
        from stats import select_qstar

        d = spec[5:]
        d = d if os.path.isabs(d) else os.path.join(REV_DIR, d)
        q = select_qstar(d)
        print(f"[sweep] Q* resolved from validation F1 in {d}: Q*={q}")
        return q
    return spec


def expand_grid(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    blocks = spec["blocks"] if "blocks" in spec else [spec]
    jobs = []
    for b in blocks:
        base = dict(b.get("base", {}))
        grid = {k: (v if isinstance(v, list) else [v]) for k, v in b.get("grid", {}).items()}
        grid = {k: [_resolve_qstar(x) for x in v] for k, v in grid.items()}
        base = {k: _resolve_qstar(v) for k, v in base.items()}
        keys = list(grid)
        for combo in itertools.product(*[grid[k] for k in keys]):
            job = {**base, **dict(zip(keys, combo))}
            job["out_dir"] = b.get("out_dir", spec.get("out_dir", os.path.join(OUT_DIR, "sweep")))
            jobs.append(job)
    # de-duplicate (e.g. Q is irrelevant for variance/pso)
    seen, uniq = set(), []
    for j in jobs:
        key = json.dumps(j, sort_keys=True)
        if key not in seen:
            seen.add(key)
            uniq.append(j)
    return uniq


def job_cmd(job: Dict[str, Any], threads: int, inter: int) -> List[str]:
    cmd = [sys.executable, os.path.join(REV_DIR, "centralized.py"), "run", "--threads", str(threads),
           "--inter-threads", str(inter)]
    for k, v in job.items():
        if k == "out_dir":
            od = v if os.path.isabs(v) else os.path.join(REV_DIR, v)
            cmd += ["--out-dir", od]
        elif k in ARG_KEYS:
            if isinstance(v, bool):
                if v:
                    cmd.append(f"--{k.replace('_', '-')}")
            elif v is not None:
                cmd += [f"--{k.replace('_', '-')}", str(v)]
    return cmd


def sweep(a: argparse.Namespace) -> None:
    spec = read_json(a.grid)
    jobs = expand_grid(spec)
    parser = build_parser()
    todo = []
    for j in jobs:
        cmd = job_cmd(j, a.threads, a.inter_threads)
        ns = parser.parse_args(cmd[2:])
        out = os.path.join(ns.out_dir, run_name(ns) + ".json")
        if os.path.exists(out):
            continue
        todo.append((run_name(ns), ns.out_dir, cmd))
    print(f"[sweep] {len(jobs)} jobs in grid, {len(jobs) - len(todo)} done, {len(todo)} to run "
          f"with {a.workers} workers x {a.threads} threads")
    if a.dry_run:
        for n, _, c in todo:
            print(" ", " ".join(c[1:]))
        return

    def _run(item):
        n, od, cmd = item
        ensure_dir(os.path.join(od, "logs"))
        t0 = time.time()
        with open(os.path.join(od, "logs", n + ".log"), "w", encoding="utf-8") as lf:
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
    r.add_argument("--context", choices=["on", "off"], default="on")
    r.add_argument("--selector", choices=["qpso", "pso", "variance", "all", "consensus"], default="qpso")
    r.add_argument("--Q", type=int, default=6)
    r.add_argument("--particles", type=int, default=20)
    r.add_argument("--iterations", type=int, default=30)
    r.add_argument("--k", type=int, default=40)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--data-seed", type=int, default=common.DATA_SEED_DEFAULT)
    r.add_argument("--epochs", type=int, default=35)
    r.add_argument("--patience", type=int, default=6)
    r.add_argument("--batch-size", type=int, default=32)
    r.add_argument("--lr", type=float, default=1e-3)
    r.add_argument("--kl-weight", type=float, default=1e-3)
    r.add_argument("--n-sub", type=int, default=80_000)
    r.add_argument("--time-steps", type=int, default=10)
    r.add_argument("--block-size", type=int, default=None)
    r.add_argument("--max-train-windows", type=int, default=0, help="smoke tests only")
    r.add_argument("--latency-n", type=int, default=1000)
    r.add_argument("--latency-warmup", type=int, default=50)
    r.add_argument("--lc-n", type=int, default=5000)
    r.add_argument("--skip-lc", action="store_true")
    r.add_argument("--threads", type=int, default=2, help="TF intra-op threads")
    r.add_argument("--inter-threads", type=int, default=1, help="TF inter-op threads")
    r.add_argument("--out-dir", default=os.path.join(OUT_DIR, "centralized"))
    r.add_argument("--tag", default="")
    r.add_argument("--force", action="store_true")
    r.add_argument("--verbose", type=int, default=2)
    r.add_argument("--no-save-model", action="store_true", help="do not save weights (models/*.weights.h5)")
    dp = sp.add_parser("deploy", help="re-measure deployment metrics in isolation from saved weights")
    dp.add_argument("--runs", nargs="+", required=True, help="glob(s) of run JSON files")
    dp.add_argument("--threads", type=int, default=2)
    dp.add_argument("--inter-threads", type=int, default=1)
    dp.add_argument("--latency-n", type=int, default=1000)
    dp.add_argument("--latency-warmup", type=int, default=50)
    dp.add_argument("--lc-n", type=int, default=5000)
    dp.add_argument("--force", action="store_true")
    s = sp.add_parser("sweep")
    s.add_argument("--grid", required=True)
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--threads", type=int, default=2)
    s.add_argument("--inter-threads", type=int, default=1)
    s.add_argument("--dry-run", action="store_true")
    return ap


def main() -> None:
    a = build_parser().parse_args()
    if a.cmd == "run":
        run_single(a)
    elif a.cmd == "deploy":
        redeploy(a)
    else:
        sweep(a)


if __name__ == "__main__":
    main()
