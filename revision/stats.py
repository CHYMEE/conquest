"""Aggregation + statistics for the revision experiments.

  python stats.py centralized --dir outputs/A [--dir outputs/B ...] --out outputs/stats/A [--ref-q 6]
  python stats.py federated   --dir outputs/E --out outputs/stats/E
  python stats.py qstar       --dir outputs/A            # Q* = argmax mean VALIDATION F1

Centralised outputs:
  runs.csv                      one row per run (all scalar metrics, deployment numbers)
  summary.csv / summary.tex     mean +- std (ddof=1) per configuration (seed aggregated)
  wilcoxon_q.csv                Wilcoxon signed-rank, paired by seed: Q=ref vs every other Q
                                (within identical context/particles/iterations/k)
  friedman_q.csv                Friedman test across Q (seeds = blocks, complete seeds only)
  qpso_vs_pso.csv               Wilcoxon paired by seed, QPSO (each Q) vs PSO with the same budget
  context_selector.csv          2x2 (context x selector) paired Wilcoxon tests
  mask_stability.csv            mean pairwise Jaccard of selected sets across seeds, per config
  feature_frequency.csv         per-feature selection frequency per config
Note: MetricsCalculator's FAR is a one-vs-rest FPR, so in the binary task FAR(Benign) == MD(Attack);
the operational false-alarm rate (benign flagged as attack) is ``far_attack``.
Federated outputs: fl_runs.csv, fl_summary.csv/.tex (final F1, AUC, min-F1, sigma(F1), R90).
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from common import REV_DIR, ensure_dir

CFG_COLS = ["selector", "context", "Q", "particles", "iterations", "k"]
METRICS = ["test_f1", "test_acc", "test_prec", "test_rec", "test_auc", "test_mse", "test_f1_attack",
           "far_benign", "far_attack", "md_benign", "md_attack", "val_f1", "epochs", "selection_time_s",
           "train_time_s", "mean_epoch_time_s", "latency_b1_median_ms", "latency_b1_p95_ms",
           "throughput_b64_sps", "throughput_b64_ms_per_sample", "T_infer_te_s", "rss_train_mb",
           "rss_infer_mb", "L_C_median_ms", "L_C_p95_ms", "n_context_selected"]


def _abs(d: str) -> str:
    return d if os.path.isabs(d) else os.path.join(REV_DIR, d)


# ======================================================================================
# loading
# ======================================================================================

def load_centralized(dirs: List[str]) -> pd.DataFrame:
    rows = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(_abs(d), "*.json"))):
            try:
                r = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            if "test_metrics" not in r:
                continue
            c, tm, vm = r["config"], r["test_metrics"], r["val_metrics"]
            # prefer deployment numbers re-measured in isolation (centralized.py deploy), if present
            dep = {**r["deployment"], **r.get("deployment_isolated", {})}
            sel = c["selector"]
            far, md = tm.get("far_per_class", {}), tm.get("missed_detections_per_class", {})
            lc = dep.get("L_C_per_flow", {}) or {}
            rows.append({
                "run_name": r["run_name"], "dir": d, "selector": sel, "context": c["context"],
                "Q": c["Q"] if sel == "qpso" else -1,
                "particles": c["particles"] if sel in ("qpso", "pso") else -1,
                "iterations": c["iterations"] if sel in ("qpso", "pso") else -1,
                "k": r["selection"]["k"], "seed": r["seeds"]["run_seed"], "data_seed": r["seeds"]["data_seed"],
                "test_f1": tm["f1_weighted"], "test_acc": tm["accuracy"], "test_prec": tm["precision_weighted"],
                "test_rec": tm["recall_weighted"], "test_auc": tm["auc_attack"], "test_mse": tm["mse"],
                "test_f1_attack": tm["f1_attack"],
                "far_benign": far.get("Benign"), "far_attack": far.get("Attack"),
                "md_benign": md.get("Benign", {}).get("rate"), "md_attack": md.get("Attack", {}).get("rate"),
                "val_f1": vm["f1_weighted"], "val_f1_attack": vm["f1_attack"],
                "epochs": r["training"]["epochs_run"], "selection_time_s": r["timing"]["selection_time_s"],
                "train_time_s": r["training"]["train_time_s"], "mean_epoch_time_s": r["training"]["mean_epoch_time_s"],
                "latency_b1_median_ms": dep["latency_b1"]["median_ms"], "latency_b1_p95_ms": dep["latency_b1"]["p95_ms"],
                "throughput_b64_sps": dep["throughput_b64"]["samples_per_s"],
                "throughput_b64_ms_per_sample": dep["throughput_b64"]["ms_per_sample"],
                "T_infer_te_s": dep["T_infer_te_s"], "rss_train_mb": dep["peak_rss_training_mb"],
                "rss_infer_mb": dep["peak_rss_inference_mb"], "L_C_median_ms": lc.get("median_ms"),
                "deploy_isolated": "deployment_isolated" in r,
                "L_C_p95_ms": lc.get("p95_ms"), "n_context_selected": r["selection"].get("n_context_selected"),
                "indices": r["selection"]["indices"], "feature_names_sel": r["selection"]["names"],
            })
    return pd.DataFrame(rows)


def load_federated(dirs: List[str]) -> pd.DataFrame:
    rows = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(_abs(d), "fl_*.json"))):
            r = json.load(open(f, encoding="utf-8"))
            if "final" not in r:
                continue
            c, fi = r["config"], r["final"]
            rows.append({"run_name": r["run_name"], "partition": c["partition"],
                         "alpha": c["alpha"] if c["partition"] == "dirichlet" else np.nan, "K": c["K"],
                         "aggregator": c["aggregator"], "mask_mode": c["mask_mode"],
                         "proxy_mode": c.get("proxy_mode", "last"), "selector": c["selector"], "Q": c["Q"],
                         "context": c["context"], "rounds": c["rounds"], "seed": c["seed"],
                         "final_f1": fi["test_f1"], "final_auc": fi["test_auc"], "final_acc": fi["test_accuracy"],
                         "client_f1_min": fi["client_f1_min"], "client_f1_mean": fi["client_f1_mean"],
                         "client_f1_sigma": fi["client_f1_std_pop"], "R90": fi["R90"],
                         "best_f1": fi["best_test_f1"], "wall_s": r["timing"]["wall_total_s"],
                         "mean_round_s": r["timing"]["mean_round_time_s"]})
    return pd.DataFrame(rows)


# ======================================================================================
# statistics
# ======================================================================================

def mean_std_table(df: pd.DataFrame, group_cols: List[str], metrics: List[str]) -> pd.DataFrame:
    metrics = [m for m in metrics if m in df.columns]
    g = df.groupby(group_cols, dropna=False)
    out = g[metrics].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out.insert(0, "n_seeds", g.size())
    return out.reset_index()


def _wilcoxon(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    from scipy.stats import wilcoxon

    d = a - b
    res = {"n_pairs": int(len(d)), "mean_diff": float(np.mean(d)) if len(d) else np.nan,
           "median_diff": float(np.median(d)) if len(d) else np.nan}
    if len(d) < 2 or np.allclose(d, 0):
        res.update(statistic=np.nan, p_value=1.0 if len(d) else np.nan)
        return res
    try:
        s = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        res.update(statistic=float(s.statistic), p_value=float(s.pvalue))
    except ValueError:
        res.update(statistic=np.nan, p_value=np.nan)
    return res


def wilcoxon_q(df: pd.DataFrame, metric: str = "test_f1", ref_q: int = 6) -> pd.DataFrame:
    q = df[df.selector == "qpso"]
    rows = []
    for key, grp in q.groupby(["context", "particles", "iterations", "k"]):
        piv = grp.pivot_table(index="seed", columns="Q", values=metric)
        if ref_q not in piv.columns:
            continue
        for other in piv.columns:
            if other == ref_q:
                continue
            pr = piv[[ref_q, other]].dropna()
            rows.append({"context": key[0], "particles": key[1], "iterations": key[2], "k": key[3],
                         "metric": metric, "Q_ref": ref_q, "Q_other": other,
                         **_wilcoxon(pr[ref_q].values, pr[other].values)})
    return pd.DataFrame(rows)


def friedman_q(df: pd.DataFrame, metric: str = "test_f1") -> pd.DataFrame:
    from scipy.stats import friedmanchisquare

    q = df[df.selector == "qpso"]
    rows = []
    for key, grp in q.groupby(["context", "particles", "iterations", "k"]):
        piv = grp.pivot_table(index="seed", columns="Q", values=metric).dropna()
        if piv.shape[1] < 3 or piv.shape[0] < 2:
            continue
        s = friedmanchisquare(*[piv[c].values for c in piv.columns])
        ranks = piv.rank(axis=1, ascending=False).mean()
        rows.append({"context": key[0], "particles": key[1], "iterations": key[2], "k": key[3], "metric": metric,
                     "n_seeds": piv.shape[0], "n_Q": piv.shape[1], "chi2": float(s.statistic),
                     "p_value": float(s.pvalue), "mean_rank_per_Q": json.dumps({int(k): round(float(v), 3) for k, v in ranks.items()})})
    return pd.DataFrame(rows)


def qpso_vs_pso(df: pd.DataFrame, metric: str = "test_f1") -> pd.DataFrame:
    rows = []
    pso = df[df.selector == "pso"]
    qp = df[df.selector == "qpso"]
    for key, pg in pso.groupby(["context", "particles", "iterations", "k"]):
        qg = qp[(qp.context == key[0]) & (qp.particles == key[1]) & (qp.iterations == key[2]) & (qp.k == key[3])]
        for Q, qq in qg.groupby("Q"):
            m = pd.merge(qq[["seed", metric]], pg[["seed", metric]], on="seed", suffixes=("_qpso", "_pso"))
            rows.append({"context": key[0], "particles": key[1], "iterations": key[2], "k": key[3], "Q": Q,
                         "metric": metric, "qpso_mean": m[f"{metric}_qpso"].mean(), "pso_mean": m[f"{metric}_pso"].mean(),
                         **_wilcoxon(m[f"{metric}_qpso"].values, m[f"{metric}_pso"].values)})
    return pd.DataFrame(rows)


def context_selector_tests(df: pd.DataFrame, metric: str = "test_f1") -> pd.DataFrame:
    """Paired tests for the 2x2 design (context on/off x selector variance/qpso)."""
    rows = []
    cells = {}
    base = df[df.selector.isin(["variance", "qpso"]) & (df.k == 40)
              & ((df.selector == "variance") | ((df.particles == 20) & (df.iterations == 30)))]
    for (sel, ctx, Q), g in base.groupby(["selector", "context", "Q"]):  # block-B budgets excluded
        cells[(sel, ctx, Q)] = g.set_index("seed")[metric]
    for (c1, s1), (c2, s2) in itertools.combinations(cells.items(), 2):
        if not ((c1[0] == c2[0]) or (c1[1] == c2[1])):  # only one factor differs
            continue
        j = pd.concat([s1, s2], axis=1, join="inner").dropna()
        if len(j) == 0:
            continue
        rows.append({"A": f"{c1[0]}/ctx{c1[1]}/Q{c1[2]}", "B": f"{c2[0]}/ctx{c2[1]}/Q{c2[2]}", "metric": metric,
                     "mean_A": float(j.iloc[:, 0].mean()), "mean_B": float(j.iloc[:, 1].mean()),
                     **_wilcoxon(j.iloc[:, 0].values, j.iloc[:, 1].values)})
    return pd.DataFrame(rows)


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def mask_stability(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(CFG_COLS):
        sets = list(g["indices"])
        pairs = [jaccard(x, y) for x, y in itertools.combinations(sets, 2)]
        rows.append({**dict(zip(CFG_COLS, key)), "n_seeds": len(sets),
                     "mean_pairwise_jaccard": float(np.mean(pairs)) if pairs else np.nan,
                     "std_pairwise_jaccard": float(np.std(pairs)) if pairs else np.nan})
    return pd.DataFrame(rows)


def feature_frequency(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(CFG_COLS):
        cnt: Dict[str, int] = {}
        for names in g["feature_names_sel"]:
            for n in names:
                cnt[n] = cnt.get(n, 0) + 1
        for n, c in cnt.items():
            rows.append({**dict(zip(CFG_COLS, key)), "feature": n, "count": c, "frequency": c / len(g)})
    out = pd.DataFrame(rows)
    return out.sort_values(CFG_COLS + ["frequency"], ascending=[True] * len(CFG_COLS) + [False]) if len(out) else out


def select_qstar(d, metric: str = "val_f1", context: str = "on", particles: int = 20, iterations: int = 30,
                 q_values=tuple(range(1, 9)), k: int = 40) -> int:
    """Q* = argmax over Q of the mean VALIDATION F1 (never test). Ties -> smaller Q.
    Restricted to the block-A configuration (context on, 20 particles, 30 iterations, Q in 1..8),
    so that block-B runs (Q 9/10, 5 seeds) in the same directory do not influence Q*."""
    df = load_centralized([d] if isinstance(d, str) else d)
    q = df[(df.selector == "qpso") & (df.context == context) & (df.particles == particles)
           & (df.iterations == iterations) & (df.Q.isin(list(q_values))) & (df.k == k)]
    if q.empty:
        raise RuntimeError(f"no qpso runs in {d} to select Q* from")
    m = q.groupby("Q")[metric].mean()
    best = m.max()
    return int(min(m[m >= best - 1e-12].index))


# ======================================================================================
# LaTeX
# ======================================================================================

def latex_mean_std(summary: pd.DataFrame, keys: List[str], metrics: List[str], caption: str, label: str,
                   scale: Optional[Dict[str, float]] = None) -> str:
    scale = scale or {}
    cols = keys + metrics
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{" + "l" * len(keys) + "c" * len(metrics) + "}",
             "\\hline", " & ".join(c.replace("_", "\\_") for c in cols) + " \\\\ \\hline"]
    for _, r in summary.iterrows():
        cells = [str(r[k]) for k in keys]
        for m in metrics:
            s = scale.get(m, 1.0)
            mu, sd = r.get(f"{m}_mean"), r.get(f"{m}_std")
            if mu is None or pd.isna(mu):
                cells.append("-")
            else:
                cells.append(f"{mu * s:.2f} $\\pm$ {0 if pd.isna(sd) else sd * s:.2f}")
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(lines)


def cmd_centralized(a) -> None:
    df = load_centralized(a.dir)
    out = _abs(a.out)
    ensure_dir(out)
    if df.empty:
        print("no runs found")
        return
    df.drop(columns=["indices", "feature_names_sel"]).to_csv(os.path.join(out, "runs.csv"), index=False)
    summ = mean_std_table(df, CFG_COLS, METRICS)
    summ.to_csv(os.path.join(out, "summary.csv"), index=False)
    pct = {m: 100.0 for m in ["test_f1", "test_acc", "test_prec", "test_rec", "val_f1"]}
    tex = latex_mean_std(summ, CFG_COLS, ["test_acc", "test_prec", "test_rec", "test_f1", "test_auc", "far_attack",
                                          "md_attack", "val_f1"],
                         "Leakage-free results (mean $\\pm$ std over seeds).", "tab:rev_summary", pct)
    tex2 = latex_mean_std(summ, CFG_COLS, ["selection_time_s", "train_time_s", "latency_b1_median_ms",
                                           "latency_b1_p95_ms", "throughput_b64_sps", "L_C_median_ms",
                                           "rss_infer_mb"],
                          "Deployment cost (mean $\\pm$ std over seeds).", "tab:rev_deploy")
    open(os.path.join(out, "summary.tex"), "w", encoding="utf-8").write(tex + "\n\n" + tex2 + "\n")
    tests = []
    for metric in a.metrics:
        w = wilcoxon_q(df, metric, a.ref_q)
        f = friedman_q(df, metric)
        p = qpso_vs_pso(df, metric)
        c = context_selector_tests(df, metric)
        for nm, t in (("wilcoxon_q", w), ("friedman_q", f), ("qpso_vs_pso", p), ("context_selector", c)):
            if len(t):
                t.to_csv(os.path.join(out, f"{nm}_{metric}.csv"), index=False)
                tests.append((nm, metric, len(t)))
    mask_stability(df).to_csv(os.path.join(out, "mask_stability.csv"), index=False)
    ff = feature_frequency(df)
    ff.to_csv(os.path.join(out, "feature_frequency.csv"), index=False)
    print(f"{len(df)} runs -> {out}; tests: {tests}")
    try:
        print("Q* (val F1):", select_qstar(a.dir))
    except RuntimeError:
        pass


def cmd_federated(a) -> None:
    df = load_federated(a.dir)
    out = _abs(a.out)
    ensure_dir(out)
    if df.empty:
        print("no FL runs found")
        return
    df.to_csv(os.path.join(out, "fl_runs.csv"), index=False)
    keys = ["partition", "alpha", "K", "aggregator", "mask_mode", "proxy_mode", "selector", "Q", "context"]
    ms = ["final_f1", "final_auc", "client_f1_min", "client_f1_mean", "client_f1_sigma", "R90", "best_f1",
          "mean_round_s"]
    summ = mean_std_table(df, keys, ms)
    summ.to_csv(os.path.join(out, "fl_summary.csv"), index=False)
    tex = latex_mean_std(summ, ["partition", "alpha", "K", "aggregator", "mask_mode"],
                         ["final_f1", "final_auc", "client_f1_min", "client_f1_sigma", "R90"],
                         "Federated results (mean $\\pm$ std over seeds), $R=30$, $E=2$.", "tab:rev_fl",
                         {"final_f1": 100.0, "client_f1_min": 100.0, "client_f1_sigma": 100.0})
    open(os.path.join(out, "fl_summary.tex"), "w", encoding="utf-8").write(tex + "\n")
    print(f"{len(df)} FL runs -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("centralized")
    c.add_argument("--dir", action="append", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--ref-q", type=int, default=6)
    c.add_argument("--metrics", nargs="+", default=["test_f1", "test_auc", "val_f1"])
    f = sp.add_parser("federated")
    f.add_argument("--dir", action="append", required=True)
    f.add_argument("--out", required=True)
    q = sp.add_parser("qstar")
    q.add_argument("--dir", action="append", required=True)
    q.add_argument("--particles", type=int, default=20)
    q.add_argument("--iterations", type=int, default=30)
    a = ap.parse_args()
    if a.cmd == "centralized":
        cmd_centralized(a)
    elif a.cmd == "federated":
        cmd_federated(a)
    else:
        print(select_qstar(a.dir, particles=a.particles, iterations=a.iterations))


if __name__ == "__main__":
    main()
