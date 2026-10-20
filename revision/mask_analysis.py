"""Reviewer 1, Comment 4: selected feature subsets, k, and consistency across seeds.

Uses block-A runs (QPSO, context on, P=20, I=30, k=40, Q=1..8, 10 seeds each).
  * per-feature selection frequency (per Q and pooled), Q* = 2 highlighted
  * feature-level stability: mean pairwise Jaccard across seeds vs. the exact random baseline
    for two uniformly random k-subsets of d features
  * family-level stability: features are grouped into redundancy families by hierarchical
    clustering of |Spearman rho| on D_tr (average linkage, cut at 1-|rho| = 0.1, i.e. |rho|>=0.9);
    two masks agree on a family if both contain at least one member
  * the deployed mask m*: the Q* run with the highest VALIDATION F1 (test never used)
Outputs: revision/outputs/stats_masks/*.json|csv, paper/images/feature_selection_heatmap.png,
         paper/tables/table_selected_features.tex
"""
import glob
import itertools
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

import common  # noqa: F401
from data import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "outputs", "stats_masks")
IMG = os.path.join(ROOT, "paper", "images")
TAB = os.path.join(ROOT, "paper", "tables")
QSTAR = 6
N_CONTEXT = 17


def jaccard(a, b):
    return len(a & b) / len(a | b)


def random_jaccard(d, k, n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.empty(n)
    for i in range(n):
        a = set(rng.choice(d, k, replace=False).tolist())
        b = set(rng.choice(d, k, replace=False).tolist())
        vals[i] = jaccard(a, b)
    return float(vals.mean()), float(vals.std())


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = []
    for f in glob.glob(os.path.join(HERE, "outputs", "central", "qpso_ctxon_Q*_P20_I30_k40_s*.json")):
        r = json.load(open(f))
        runs.append(dict(Q=r["config"]["Q"], seed=r["config"]["seed"], idx=set(r["selection"]["indices"]),
                         val_f1=r["val_metrics"]["f1_weighted"], test_f1=r["test_metrics"]["f1_weighted"],
                         run=r["run_name"]))
    qs = sorted({r["Q"] for r in runs})
    ds = load_dataset(context=True)
    names = list(ds.feature_names)
    d = len(names)
    k = len(runs[0]["idx"])
    assert all(len(r["idx"]) == k for r in runs)
    ctx = set(range(d - N_CONTEXT, d))
    assert names[d - N_CONTEXT] == "n_flows_src_10", names[d - N_CONTEXT]

    # ---- redundancy families on the training split
    tr = ds.rows("train")
    X = np.asarray(ds.X[tr], dtype=np.float64)
    const = X.std(0) == 0
    rho = np.nan_to_num(spearmanr(X).correlation, nan=0.0)
    rho[const, :] = 0
    rho[:, const] = 0
    np.fill_diagonal(rho, 1.0)
    dist = np.clip(1 - np.abs(rho), 0, None)
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    fam = fcluster(linkage(squareform(dist, checks=False), "average"), t=0.1, criterion="distance")
    n_fam = len(set(fam))

    def fams(idx):
        return {int(fam[i]) for i in idx}

    rand_feat = random_jaccard(d, k)
    # family-level random baseline by simulation
    rng = np.random.default_rng(1)
    rf = [jaccard(fams(rng.choice(d, k, replace=False)), fams(rng.choice(d, k, replace=False)))
          for _ in range(20000)]
    rand_fam = (float(np.mean(rf)), float(np.std(rf)))

    per_q = []
    freq = np.zeros((len(qs), d))
    for qi, q in enumerate(qs):
        rq = [r for r in runs if r["Q"] == q]
        for r in rq:
            for i in r["idx"]:
                freq[qi, i] += 1 / len(rq)
        jf = [jaccard(a["idx"], b["idx"]) for a, b in itertools.combinations(rq, 2)]
        jm = [jaccard(fams(a["idx"]), fams(b["idx"])) for a, b in itertools.combinations(rq, 2)]
        per_q.append(dict(Q=q, n=len(rq), feat_jaccard=float(np.mean(jf)), feat_jaccard_std=float(np.std(jf)),
                          fam_jaccard=float(np.mean(jm)), fam_jaccard_std=float(np.std(jm)),
                          n_families_selected=float(np.mean([len(fams(r["idx"])) for r in rq])),
                          context_selected=float(np.mean([len(r["idx"] & ctx) for r in rq]))))

    pooled = freq.mean(0)
    qi = qs.index(QSTAR)
    rq = sorted([r for r in runs if r["Q"] == QSTAR], key=lambda r: -r["val_f1"])
    deployed = rq[0]
    core = [i for i in range(d) if freq[qi, i] >= 0.8]
    never = [i for i in range(d) if pooled[i] == 0]
    summary = dict(d=d, k=k, n_families=n_fam, family_threshold_abs_rho=0.9,
                   random_feat_jaccard=rand_feat, random_family_jaccard=rand_fam, per_q=per_q,
                   deployed=dict(run=deployed["run"], seed=deployed["seed"], val_f1=deployed["val_f1"],
                                 test_f1=deployed["test_f1"], indices=sorted(deployed["idx"]),
                                 names=[names[i] for i in sorted(deployed["idx"])],
                                 n_context=len(deployed["idx"] & ctx)),
                   qstar_core_ge80=[names[i] for i in core],
                   never_selected_any_run=[names[i] for i in never],
                   families={int(f): [names[i] for i in range(d) if fam[i] == f] for f in sorted(set(fam))})
    json.dump(summary, open(os.path.join(OUT, "mask_summary.json"), "w"), indent=1)
    with open(os.path.join(OUT, "feature_frequency.csv"), "w") as fh:
        fh.write("feature,is_context,family," + ",".join(f"Q{q}" for q in qs) + ",pooled\n")
        for i in range(d):
            fh.write(f"{names[i]},{int(i in ctx)},{fam[i]}," + ",".join(f"{freq[j, i]:.2f}" for j in range(len(qs)))
                     + f",{pooled[i]:.3f}\n")

    from feature_selectors import select_consensus
    cons = select_consensus(X, ds.y[tr], k=k, seed=0, Q=QSTAR)
    cons_idx = set(cons.indices)
    # ---- heatmap: features (sorted by pooled frequency) x Q  (style: figstyle, Fig. 6 reference)
    import figstyle
    figstyle.apply()
    order = np.argsort(-pooled, kind="stable")
    fig, ax = plt.subplots(figsize=(22, 5.6))
    im = ax.imshow(freq[:, order], aspect="auto", cmap="plasma", vmin=0, vmax=1)
    ax.set_xticks(np.arange(-0.5, d, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(qs), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    ax.set_yticks(range(len(qs)))
    ax.set_yticklabels([f"Q={q}" + ("*" if q == QSTAR else "") for q in qs], fontsize=13)
    ax.set_xticks(range(d))
    ax.set_xticklabels([("● " if i in cons_idx else "") + names[i] for i in order],
                       rotation=90, fontsize=8.5, fontweight="bold")
    for t, i in zip(ax.get_xticklabels(), order):
        if i in ctx:
            t.set_color("#e8112d")
    for sp in ax.spines.values():
        sp.set_linewidth(2.0)
    cb = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cb.set_label("Selection frequency\n(10 seeds; 5 for Q=9,10)", fontsize=13, fontweight="bold")
    cb.outline.set_linewidth(1.6)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(IMG, f"feature_selection_heatmap.{ext}"))
    plt.close(fig)

    # ---- table of the consensus mask (two columns of 20), with Q* selection frequency


    run_fit = [json.load(open(f))["selection"]["best_fitness"] for f in glob.glob(os.path.join(
        HERE, "outputs", "central", f"qpso_ctxon_Q{QSTAR}_P20_I30_k40_s*.json"))]
    summary["consensus"] = dict(indices=cons.indices, names=[names[i] for i in cons.indices],
                                fitness=cons.final_mask_fitness, n_context=len(set(cons.indices) & ctx),
                                min_freq=min(cons.extra["selection_frequency"].values()))
    summary["qstar_single_run_fitness"] = (float(np.mean(run_fit)), float(np.std(run_fit, ddof=1)), len(run_fit))
    json.dump(summary, open(os.path.join(OUT, "mask_summary.json"), "w"), indent=1)
    tex_name = lambda n: n.replace("_", r"\_").replace("/", "/\\allowbreak ")
    items = [(names[i], freq[qi, i], i in ctx) for i in sorted(cons.indices, key=lambda i: (-freq[qi, i], i))]
    half = (len(items) + 1) // 2
    rows = []
    for a, b in itertools.zip_longest(items[:half], items[half:]):
        cell = lambda it: ("", "") if it is None else (
            (r"\textit{" + tex_name(it[0]) + "}" if it[2] else tex_name(it[0])), f"{it[1]:.1f}")
        ca, cb_ = cell(a), cell(b)
        rows.append(f"{ca[0]} & {ca[1]} & {cb_[0]} & {cb_[1]} \\\\")
    open(os.path.join(TAB, "table_selected_features.tex"), "w").write("\n".join(rows) + "\n")

    print(json.dumps({k_: v for k_, v in summary.items() if k_ not in ("families", "deployed")}, indent=1))
    print("deployed:", deployed["run"], "val", round(deployed["val_f1"], 4), "test", round(deployed["test_f1"], 4),
          "n_context", summary["deployed"]["n_context"])
    big = {f: m for f, m in summary["families"].items() if len(m) > 1}
    print("families with >1 member:", len(big))
    for f, m in big.items():
        print(" ", m)


if __name__ == "__main__":
    main()
