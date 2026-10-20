"""Leakage-free data pipeline for the CONQUEST revision (ACI-IoT-2023, binary Benign vs Attack).

Pipeline (each step fitted on TRAIN only where applicable):
  1. Load CSV (only needed columns, float32), parse Timestamp (three formats, see below),
     stable-sort chronologically (ties -> file order).
  2. Feature matrix: drop Flow ID, Src IP, Dst IP, Timestamp, Connection Type, Label.
     Src Port / Dst Port / Protocol are kept as numeric.  The four CICFlowMeter
     ``Idle Mean/Std/Max/Min`` columns are ALSO dropped by default because in this dataset
     they contain absolute epoch micro-seconds (Idle Max == flow start time in UTC us for
     100% of rows), i.e. a wall-clock proxy that identifies attack sessions
     (use --keep-time-proxies to keep them for an ablation).
  3. Context features F_c (17 dims, see context.py for the table) computed on the FULL
     sorted stream before subsampling, strictly causal, grouped by Src IP.
  4. Balanced subsample (default 40k benign + 40k attack, seeded, without replacement,
     uniform within class -> attack-type mix preserved), re-sorted chronologically.
  5. Split chronological 70/15/15; if any class < 25 % in any split -> blocked split
     (blocks of 500 consecutive rows, seeded shuffle of blocks 70/15/15, first T rows of
     every block dropped as a purge gap).  The mode and per-split counts are logged.
  6. MinMax scaler fitted on train rows only, applied to val/test, clipped to [0, 1].
  7. Windows of T=10 flows built inside each contiguous segment (no window crosses a
     split or block boundary); label = label of the last flow.

Timestamp formats found in the CSV (verified against the epoch value in ``Idle Max``,
which equals local time + 4 h for every row):
  * 'YYYY-DD-MM HH:MM:SS'  (e.g. '2023-01-11 09:43:40' = 1 Nov 2023)  - year-DAY-month!
  * 'YYYY-MM-DD HH:MM:SS'  only for '2023-10-30 ...' (unambiguous)
  * 'DD/MM/YYYY hh:mm:ss AM/PM' (162,050 Benign-only rows, 31 Oct and 3 Nov)

CLI:
  python data.py prepare [--n-sub 80000] [--data-seed 42]   # build caches + split report
  python data.py selftest                                    # streaming == batch on real flows
  python data.py table                                       # print context-feature table
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from common import CACHE_DIR, CSV_PATH, DATA_SEED_DEFAULT, ensure_dir, read_json, write_json
from context import CONTEXT_NAMES, N_CONTEXT, compute_context_batch, compute_context_streaming, markdown_table

FULL_DIR = os.path.join(CACHE_DIR, "full_v1")
ID_COLS = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Connection Type", "Label"]
TIME_PROXY_COLS = ["Idle Mean", "Idle Std", "Idle Max", "Idle Min"]
BLOCK_SIZE = 500
MIN_CLASS_FRAC = 0.25
SPLIT_FRACS = (0.70, 0.15, 0.15)


# ======================================================================================
# 1-3. Full-stream cache
# ======================================================================================

def _parse_timestamps(cats: pd.Index) -> Tuple[np.ndarray, Dict[str, int]]:
    s = pd.Series(cats.astype(str))
    iso = s.str.match(r"^\d{4}-")
    ydm = pd.to_datetime(s.where(iso), format="%Y-%d-%m %H:%M:%S", errors="coerce")
    ymd = pd.to_datetime(s.where(iso), format="%Y-%m-%d %H:%M:%S", errors="coerce")
    dmy = pd.to_datetime(s.where(~iso), format="%d/%m/%Y %I:%M:%S %p", errors="coerce")
    parsed = ydm.fillna(ymd).fillna(dmy)
    if parsed.isna().any():
        raise ValueError(f"Unparsed timestamps: {s[parsed.isna()].head().tolist()}")
    stats = {
        "unique_ts": int(len(s)),
        "fmt_year_day_month": int(ydm.notna().sum()),
        "fmt_year_month_day": int((ydm.isna() & ymd.notna()).sum()),
        "fmt_day_month_year_ampm": int(dmy.notna().sum()),
    }
    secs = (parsed.values.astype("datetime64[s]").astype(np.int64))
    return secs, stats


def build_full_cache(csv_path: str = CSV_PATH, out_dir: str = FULL_DIR, chunksize: int = 200_000) -> Dict:
    """Read the CSV once, sort chronologically, compute context features, store .npy files."""
    if os.path.exists(os.path.join(out_dir, "meta.json")):
        return read_json(os.path.join(out_dir, "meta.json"))
    ensure_dir(out_dir)
    t0 = time.time()
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    feat_cols = [c for c in header if c not in ID_COLS]

    ids = pd.read_csv(csv_path, usecols=["Src IP", "Dst IP", "Timestamp", "Label", "Idle Max"],
                      dtype={"Src IP": "category", "Dst IP": "category", "Timestamp": "category",
                             "Label": "category", "Idle Max": "float64"})
    n = len(ids)
    ts_secs_cat, ts_stats = _parse_timestamps(ids["Timestamp"].cat.categories)
    t = ts_secs_cat[ids["Timestamp"].cat.codes.values]
    # Consistency check against the epoch micro-seconds hidden in Idle Max (UTC = local + 4h)
    off = ids["Idle Max"].values / 1e6 - t
    ts_stats["idle_max_minus_ts_s_min"] = float(np.min(off))
    ts_stats["idle_max_minus_ts_s_max"] = float(np.max(off))
    ts_stats["frac_rows_idle_max_is_epoch"] = float(np.mean(np.abs(off - 14400.5) <= 1.0))

    label_str = ids["Label"].astype(str).values
    attack_names = sorted(set(label_str) - {"Benign"})
    label_names = ["Benign"] + attack_names
    lab_map = {k: i for i, k in enumerate(label_names)}
    atk = np.array([lab_map[v] for v in label_str], dtype=np.int8)
    y = (atk > 0).astype(np.int8)
    src = ids["Src IP"].cat.codes.values.astype(np.int32)
    dst = ids["Dst IP"].cat.codes.values.astype(np.int32)
    del ids, label_str

    order = np.lexsort((np.arange(n), t))  # stable chronological sort, ties -> file order

    # numeric features, chunked, float32, nan/inf -> 0 (as in the original pipeline)
    X = np.lib.format.open_memmap(os.path.join(out_dir, "X_unsorted.npy"), mode="w+",
                                  dtype=np.float32, shape=(n, len(feat_cols)))
    col_max = np.full(len(feat_cols), -np.inf)
    r0 = 0
    for ch in pd.read_csv(csv_path, usecols=feat_cols, dtype={c: "float64" for c in feat_cols},
                          chunksize=chunksize):
        a = ch[feat_cols].to_numpy(dtype=np.float64)
        a[~np.isfinite(a)] = 0.0
        col_max = np.maximum(col_max, np.abs(a).max(axis=0))
        X[r0:r0 + len(a)] = a.astype(np.float32)
        r0 += len(a)
    X.flush()
    assert r0 == n

    Xs = np.lib.format.open_memmap(os.path.join(out_dir, "X_base.npy"), mode="w+",
                                   dtype=np.float32, shape=(n, len(feat_cols)))
    for j in range(len(feat_cols)):
        Xs[:, j] = X[order, j]
    Xs.flush()
    del X
    os.remove(os.path.join(out_dir, "X_unsorted.npy"))

    t_s, y_s, atk_s, src_s, dst_s = t[order], y[order], atk[order], src[order], dst[order]
    ci = {c: feat_cols.index(c) for c in feat_cols}
    dport = Xs[:, ci["Dst Port"]].astype(np.int64)
    proto = Xs[:, ci["Protocol"]].astype(np.int64)
    syn = Xs[:, ci["SYN Flag Count"]].astype(np.float64)
    pkts = Xs[:, ci["Total Fwd Packet"]].astype(np.float64) + Xs[:, ci["Total Bwd packets"]]
    nbytes = (Xs[:, ci["Total Length of Fwd Packet"]].astype(np.float64)
              + Xs[:, ci["Total Length of Bwd Packet"]])
    tc = time.time()
    ctx = compute_context_batch(t_s, src_s, dst_s, dport, proto, syn, pkts, nbytes)
    ctx_time = time.time() - tc

    np.save(os.path.join(out_dir, "ctx.npy"), ctx.astype(np.float32))
    np.save(os.path.join(out_dir, "t.npy"), t_s)
    np.save(os.path.join(out_dir, "y.npy"), y_s)
    np.save(os.path.join(out_dir, "atk.npy"), atk_s)
    np.save(os.path.join(out_dir, "src.npy"), src_s)
    np.save(os.path.join(out_dir, "dst.npy"), dst_s)
    np.save(os.path.join(out_dir, "file_idx.npy"), order.astype(np.int64))

    suspicious = {c: float(m) for c, m in zip(feat_cols, col_max) if m > 1e14}
    meta = {
        "n_rows": int(n),
        "base_feature_names": feat_cols,
        "context_names": CONTEXT_NAMES,
        "label_names": label_names,
        "timestamp_parse": ts_stats,
        "t_min": str(np.datetime64(int(t_s[0]), "s")),
        "t_max": str(np.datetime64(int(t_s[-1]), "s")),
        "n_src_ips": int(len(np.unique(src_s))),
        "columns_with_abs_max_gt_1e14": suspicious,
        "time_proxy_cols_dropped_by_default": TIME_PROXY_COLS,
        "context_batch_time_s": ctx_time,
        "build_time_s": time.time() - t0,
        "label_counts": {label_names[i]: int(c) for i, c in enumerate(np.bincount(atk_s, minlength=len(label_names)))},
    }
    write_json(os.path.join(out_dir, "meta.json"), meta)
    return meta


def load_full(mmap: bool = True) -> Dict[str, np.ndarray]:
    meta = build_full_cache()
    mode = "r" if mmap else None
    d = {k: np.load(os.path.join(FULL_DIR, f"{k}.npy"), mmap_mode=mode)
         for k in ["X_base", "ctx", "t", "y", "atk", "src", "dst", "file_idx"]}
    d["meta"] = meta
    return d


# ======================================================================================
# 4. Subsample
# ======================================================================================

def build_subsample(n_sub: int = 80_000, data_seed: int = DATA_SEED_DEFAULT) -> str:
    path = os.path.join(CACHE_DIR, f"sub_n{n_sub}_s{data_seed}.npz")
    if os.path.exists(path):
        return path
    full = load_full()
    y = np.asarray(full["y"])
    rng = np.random.default_rng(data_seed)
    per = n_sub // 2
    ben = np.flatnonzero(y == 0)
    att = np.flatnonzero(y == 1)
    idx = np.sort(np.concatenate([rng.choice(ben, per, replace=False), rng.choice(att, per, replace=False)]))
    np.savez(path, idx=idx, X_base=np.asarray(full["X_base"][idx]), ctx=np.asarray(full["ctx"][idx]),
             y=y[idx], atk=np.asarray(full["atk"][idx]), t=np.asarray(full["t"][idx]),
             src=np.asarray(full["src"][idx]))
    return path


# ======================================================================================
# 5. Split
# ======================================================================================

Segments = List[Tuple[int, int]]  # half-open row ranges [s, e) into the subsample


def _counts(y: np.ndarray, atk: np.ndarray, segs: Segments, label_names: List[str]) -> Dict:
    rows = np.concatenate([np.arange(s, e) for s, e in segs]) if segs else np.array([], int)
    yy, aa = y[rows], atk[rows]
    n = len(rows)
    return {
        "n_rows": int(n),
        "n_segments": len(segs),
        "benign": int((yy == 0).sum()),
        "attack": int((yy == 1).sum()),
        "attack_frac": float(yy.mean()) if n else float("nan"),
        "attack_types": {label_names[i]: int(c) for i, c in enumerate(np.bincount(aa, minlength=len(label_names))) if c > 0},
    }


def make_split(y: np.ndarray, atk: np.ndarray, label_names: List[str], time_steps: int,
               data_seed: int, force_mode: Optional[str] = None, block_size: Optional[int] = None) -> Dict:
    n = len(y)
    # 2000-row blocks for the real 80k subsample; tiny smoke subsamples use n/40.
    block_size = block_size or (BLOCK_SIZE if n >= 40_000 else max(50, n // 40))
    c1 = int(round(SPLIT_FRACS[0] * n))
    c2 = int(round((SPLIT_FRACS[0] + SPLIT_FRACS[1]) * n))
    chrono = {"train": [(0, c1)], "val": [(c1, c2)], "test": [(c2, n)]}
    chrono_counts = {k: _counts(y, atk, v, label_names) for k, v in chrono.items()}
    fracs = [min(c["attack_frac"], 1 - c["attack_frac"]) for c in chrono_counts.values()]
    use_blocked = min(fracs) < MIN_CLASS_FRAC
    if force_mode is not None:
        use_blocked = force_mode == "blocked"
    info = {"chronological_counts": chrono_counts,
            "chronological_min_class_frac": float(min(fracs))}
    if not use_blocked:
        info.update(mode="chronological", segments=chrono, counts=chrono_counts)
        return info
    nb = int(np.ceil(n / block_size))
    rng = np.random.default_rng(data_seed)
    perm = rng.permutation(nb)
    n_tr = int(round(SPLIT_FRACS[0] * nb))
    n_va = int(round(SPLIT_FRACS[1] * nb))
    assign = {}
    for i, b in enumerate(perm):
        assign[int(b)] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    segs: Dict[str, Segments] = {"train": [], "val": [], "test": []}
    for b in range(nb):
        s, e = b * block_size, min((b + 1) * block_size, n)
        if b > 0:
            s += time_steps  # purge gap at every block boundary
        if e - s >= time_steps:
            segs[assign[b]].append((s, e))
    counts = {k: _counts(y, atk, v, label_names) for k, v in segs.items()}
    info.update(mode="blocked", segments=segs, counts=counts,
                block_size=block_size, block_assignment={k: sorted(b for b, a in assign.items() if a == k) for k in segs})
    return info


# ======================================================================================
# 6-7. Dataset object: scaling + windows
# ======================================================================================

@dataclass
class Dataset:
    X: np.ndarray  # (n_sub, d) MinMax-scaled (train-fitted), clipped to [0,1]
    X_raw: np.ndarray  # (n_sub, d) unscaled
    y: np.ndarray
    atk: np.ndarray
    t: np.ndarray
    full_idx: np.ndarray  # positions in the full sorted stream
    feature_names: List[str]
    label_names: List[str]
    split: Dict
    scaler_min: np.ndarray
    scaler_scale: np.ndarray
    context: bool
    data_seed: int
    time_steps: int
    extra: Dict = field(default_factory=dict)

    def rows(self, part: str) -> np.ndarray:
        return np.concatenate([np.arange(s, e) for s, e in self.split["segments"][part]])

    def segments(self, part: str) -> Segments:
        return self.split["segments"][part]


def load_dataset(context: bool = True, n_sub: int = 80_000, data_seed: int = DATA_SEED_DEFAULT,
                 time_steps: int = 10, keep_time_proxies: bool = False,
                 force_split_mode: Optional[str] = None, block_size: Optional[int] = None,
                 drop_constant: bool = True) -> Dataset:
    meta = build_full_cache()
    sub = np.load(build_subsample(n_sub, data_seed))
    base_names = list(meta["base_feature_names"])
    keep = [i for i, c in enumerate(base_names) if keep_time_proxies or c not in TIME_PROXY_COLS]
    Xb = sub["X_base"][:, keep]
    names = [base_names[i] for i in keep]
    if context:
        X_raw = np.concatenate([Xb, sub["ctx"]], axis=1).astype(np.float32)
        names = names + list(CONTEXT_NAMES)
    else:
        X_raw = Xb.astype(np.float32)
    y = sub["y"].astype(np.int64)
    atk = sub["atk"].astype(np.int64)
    split = make_split(y, atk, meta["label_names"], time_steps, data_seed, force_split_mode, block_size)
    tr = np.concatenate([np.arange(s, e) for s, e in split["segments"]["train"]])
    n_base_pre = Xb.shape[1]
    nonconst = np.arange(X_raw.shape[1])
    dropped = []
    if drop_constant:
        # Zero-variance filter fitted on D_tr only: constant columns carry no information and
        # are "free" under the trace-ratio (Fisher) fitness, so QPSO would otherwise select them.
        v = X_raw[tr].max(axis=0) > X_raw[tr].min(axis=0)
        nonconst = np.flatnonzero(v)
        dropped = [names[i] for i in np.flatnonzero(~v)]
        X_raw = np.ascontiguousarray(X_raw[:, nonconst])
        names = [names[i] for i in nonconst]
    n_base = int(np.sum(nonconst < n_base_pre))
    mn = X_raw[tr].min(axis=0).astype(np.float64)
    mx = X_raw[tr].max(axis=0).astype(np.float64)
    rng_ = mx - mn
    scale = np.where(rng_ > 0, 1.0 / np.where(rng_ > 0, rng_, 1.0), 0.0)
    X = np.clip((X_raw - mn) * scale, 0.0, 1.0).astype(np.float32)
    return Dataset(X=X, X_raw=X_raw, y=y, atk=atk, t=sub["t"], full_idx=sub["idx"], feature_names=names,
                   label_names=meta["label_names"], split=split, scaler_min=mn, scaler_scale=scale,
                   context=context, data_seed=data_seed, time_steps=time_steps,
                   extra={"base_cols_kept": keep, "nonconst_cols": nonconst.tolist(),
                          "dropped_constant": dropped, "n_base": n_base})


def make_windows(X: np.ndarray, y: np.ndarray, segments: Segments, T: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Windows of T consecutive rows built inside each segment; label = label of last row.
    Returns (Xw [N,T,d], yw [N], end_row [N])."""
    Xs, ys, ends = [], [], []
    for s, e in segments:
        if e - s < T:
            continue
        v = np.lib.stride_tricks.sliding_window_view(X[s:e], T, axis=0)  # (L-T+1, d, T)
        Xs.append(np.transpose(v, (0, 2, 1)))
        end = np.arange(s + T - 1, e)
        ys.append(y[end])
        ends.append(end)
    if not Xs:
        return np.zeros((0, T, X.shape[1]), np.float32), np.zeros(0, np.int64), np.zeros(0, np.int64)
    return (np.ascontiguousarray(np.concatenate(Xs)).astype(np.float32), np.concatenate(ys),
            np.concatenate(ends))


def split_report(ds: Dataset) -> Dict:
    return {"mode": ds.split["mode"], "counts": ds.split["counts"],
            "chronological_counts": ds.split["chronological_counts"],
            "chronological_min_class_frac": ds.split["chronological_min_class_frac"],
            "time_range": {p: [str(np.datetime64(int(ds.t[ds.rows(p)].min()), "s")),
                               str(np.datetime64(int(ds.t[ds.rows(p)].max()), "s"))]
                           for p in ("train", "val", "test")}}


# ======================================================================================
# Self-test: streaming == batch on real flows
# ======================================================================================

def selftest(n: int = 5000) -> Dict:
    full = load_full()
    meta = full["meta"]
    names = meta["base_feature_names"]
    ci = {c: names.index(c) for c in names}
    N = meta["n_rows"]
    res = {}
    y = np.asarray(full["y"])
    # slice 1: start of stream; slice 2: around the median attack flow; slice 3: middle
    att_pos = np.flatnonzero(y == 1)
    att_mid = int(att_pos[len(att_pos) // 2])
    for tag, s in (("stream_start", 0), ("attack_dense_region", max(0, att_mid - n // 2)),
                   ("mid_stream", N // 2)):
        sl = slice(s, s + n)
        Xb = np.asarray(full["X_base"][sl])
        args = (np.asarray(full["t"][sl]), np.asarray(full["src"][sl]), np.asarray(full["dst"][sl]),
                Xb[:, ci["Dst Port"]].astype(np.int64), Xb[:, ci["Protocol"]].astype(np.int64),
                Xb[:, ci["SYN Flag Count"]].astype(np.float64),
                Xb[:, ci["Total Fwd Packet"]].astype(np.float64) + Xb[:, ci["Total Bwd packets"]],
                Xb[:, ci["Total Length of Fwd Packet"]].astype(np.float64) + Xb[:, ci["Total Length of Bwd Packet"]])
        a = compute_context_batch(*args)
        b = compute_context_streaming(*args)
        d = np.abs(a - b).max(axis=0)
        res[tag] = {"start": s, "n": n, "max_abs_diff_per_feature": dict(zip(CONTEXT_NAMES, d.tolist())),
                    "identical": bool(np.array_equal(a, b)), "allclose": bool(np.allclose(a, b, rtol=1e-12, atol=1e-9)),
                    "n_attack": int(y[sl].sum())}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prepare")
    p.add_argument("--n-sub", type=int, default=80_000)
    p.add_argument("--data-seed", type=int, default=DATA_SEED_DEFAULT)
    p.add_argument("--time-steps", type=int, default=10)
    p.add_argument("--context", choices=["on", "off"], default="on")
    p.add_argument("--block-size", type=int, default=None, help="default 500")
    p.add_argument("--out", default=None, help="split report JSON path")
    sp.add_parser("selftest")
    sp.add_parser("table")
    a = ap.parse_args()
    if a.cmd == "table":
        print(markdown_table())
        return
    if a.cmd == "selftest":
        r = selftest()
        print(json.dumps(r, indent=1))
        write_json(os.path.join(CACHE_DIR, "context_selftest.json"), r)
        return
    t0 = time.time()
    meta = build_full_cache()
    print(f"full cache ready ({time.time() - t0:.1f}s): {meta['n_rows']} rows, "
          f"{len(meta['base_feature_names'])} base cols, t in [{meta['t_min']}, {meta['t_max']}]")
    ds = load_dataset(context=a.context == "on", n_sub=a.n_sub, data_seed=a.data_seed, time_steps=a.time_steps,
                      block_size=a.block_size)
    rep = split_report(ds)
    rep["n_features"] = len(ds.feature_names)
    rep["feature_names"] = ds.feature_names
    for part in ("train", "val", "test"):
        Xw, yw, _ = make_windows(ds.X, ds.y, ds.segments(part), ds.time_steps)
        rep["counts"][part]["n_windows"] = int(len(yw))
        rep["counts"][part]["window_attack_frac"] = float(yw.mean()) if len(yw) else float("nan")
    out = a.out or os.path.join(CACHE_DIR, f"split_report_n{a.n_sub}_s{a.data_seed}_ctx{a.context}_b{a.block_size or 'def'}.json")
    write_json(out, rep)
    print(json.dumps({k: rep[k] for k in ("mode", "counts", "chronological_counts", "time_range")}, indent=1))
    print("report:", out)


if __name__ == "__main__":
    main()
