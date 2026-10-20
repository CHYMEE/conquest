"""Reviewer 1, Comment 2: Q-ablation table (Table V) + box-plot figure with statistics.

Uses block-A runs (QPSO, context on, P=20, I=30, k=40; 10 seeds per Q). If block-B runs at
Q=9/10 with the same budget exist, they are added to the figure (fewer seeds, shown hatched).
FAR / MD are recomputed from each run's confusion matrix (rows = true, cols = predicted,
order [Benign, Attack]): FAR = benign flagged as attack / benign; MD = attacks missed / attacks.

Outputs (paper/images, paper/tables, revision/outputs/stats_q):
  q_ablation_boxplot.png/.pdf, table_v_q_ablation.tex, q_ablation_stats.json
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(HERE, "outputs", "central")
OUT = os.path.join(HERE, "outputs", "stats_q")
IMG = os.path.join(ROOT, "paper", "images")
TAB = os.path.join(ROOT, "paper", "tables")


def load():
    rows = []
    for f in glob.glob(os.path.join(RUNS, "qpso_ctxon_Q*_P20_I30_k40_s*.json")):
        r = json.load(open(f))
        cm = np.array(r["test_metrics"]["confusion_matrix"], dtype=float)
        t = r["test_metrics"]
        rows.append(dict(Q=int(r["config"]["Q"]), seed=int(r["config"]["seed"]),
                         acc=t["accuracy"], f1=t["f1_weighted"], auc=t["auc_attack"], mse=t["mse"],
                         far=cm[0, 1] / cm[0].sum(), md=cm[1, 0] / cm[1].sum(),
                         val_f1=r["val_metrics"]["f1_weighted"],
                         fit=r["selection"].get("final_mask_fitness")))
    return rows


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    rows = load()
    qs_all = sorted({r["Q"] for r in rows})
    by = {q: sorted([r for r in rows if r["Q"] == q], key=lambda r: r["seed"]) for q in qs_all}
    qs = [q for q in qs_all if q <= 8 and len(by[q]) == 10]          # block A (complete)
    extra = [q for q in qs_all if q > 8]                              # block B, Q=9/10
    seeds = sorted({r["seed"] for r in by[qs[0]]})

    def vec(q, m):
        d = {r["seed"]: r[m] for r in by[q]}
        return np.array([d[s] for s in seeds])

    # Pre-declared rule: Q* = argmax mean validation F1 (ties -> smaller Q)
    val_means = {q: float(np.mean(vec(q, "val_f1"))) for q in qs}
    qstar = max(qs, key=lambda q: (round(val_means[q], 12), -q))

    stats = {"Q_star": qstar, "seeds": seeds, "friedman": {}, "wilcoxon": {}}
    for m in ["f1", "auc", "val_f1"]:
        chi2, p = friedmanchisquare(*[vec(q, m) for q in qs])
        stats["friedman"][m] = {"chi2": float(chi2), "p": float(p), "df": len(qs) - 1}
    for ref in [qstar, 6]:
        others = [q for q in qs if q != ref]
        raw = [float(wilcoxon(vec(ref, "f1"), vec(q, "f1")).pvalue) for q in others]
        adj = holm(raw)
        stats["wilcoxon"][f"ref{ref}"] = {
            str(q): {"p_raw": raw[i], "p_holm": float(adj[i]),
                     "mean_diff_pp": float(100 * np.mean(vec(ref, "f1") - vec(q, "f1")))}
            for i, q in enumerate(others)}
    # span of means (pp) and pooled seed std (pp) for the text
    f1m = {q: 100 * np.mean(vec(q, "f1")) for q in qs}
    stats["f1_mean_span_pp"] = float(max(f1m.values()) - min(f1m.values()))
    stats["f1_mean_seed_std_pp"] = float(np.mean([100 * np.std(vec(q, "f1"), ddof=1) for q in qs]))
    json.dump(stats, open(os.path.join(OUT, "q_ablation_stats.json"), "w"), indent=1)

    # ---------------- Table V ----------------
    def ms(q, m, scale=100, nd=2):
        v = vec(q, m) * scale
        return f"{v.mean():.{nd}f}$\\pm${v.std(ddof=1):.{nd}f}"

    pq = stats["wilcoxon"][f"ref{qstar}"]
    p6 = stats["wilcoxon"]["ref6"]
    fmt_p = lambda d, q: "--" if str(q) not in d else f"{d[str(q)]['p_holm']:.2f}"
    lines = []
    for q in qs:
        cells = [ms(q, "acc"), ms(q, "f1"), ms(q, "auc", 1, 4), ms(q, "mse", 1, 4), ms(q, "far"), ms(q, "md"),
                 ms(q, "val_f1"), fmt_p(pq, q)] + ([] if qstar == 6 else [fmt_p(p6, q)])
        lab = f"\\textbf{{{q}}}$^{{*}}$" if q == qstar else str(q)
        if q == qstar:
            cells = [f"\\textbf{{{c}}}" if "pm" in c else c for c in cells]
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
    fr = stats["friedman"]
    if qstar == 6:
        pdesc = (r"$p_{Q^*}$: Holm-adjusted two-sided Wilcoxon signed-rank $p$-value of test F1 against "
                 r"$Q^{*}{=}6$, selected by validation F1 (it coincides with the operating point of the "
                 r"original submission).")
        pcols, colspec = r"$p_{Q^*}$", "c" * 9
    else:
        pdesc = (r"$p_{Q^*}$, $p_{6}$: Holm-adjusted two-sided Wilcoxon signed-rank $p$-values of test F1 "
                 r"against $Q^{*}{=}%d$ (selected by validation F1) and against $Q{=}6$ (the operating point "
                 r"of the original submission)." % qstar)
        pcols, colspec = r"$p_{Q^*}$ & $p_{6}$", "c" * 10
    tex = r"""\begin{table*}[!t]
\centering
\caption{\textcolor{red}{Variable-qubit QPSO ablation for LSTM(+VAE)+SGD on ACI-IoT (context features on,
$P{=}20$, $I{=}30$, $k{=}40$; mean$\pm$std over ten seeds, QPSO re-executed per seed). FAR: benign flows
flagged as attack; MD: attacks missed (both \%%). %(pdesc)s Friedman test across $Q$: test F1
$\chi^2_{%(df)d}{=}%(c1).2f$, $p{=}%(p1).2f$; test AUC $p{=}%(p2).2f$; validation F1 $p{=}%(p3).2f$.}}
\label{tab:qubit_ablation_improved_robust}
\color{red}
\setlength{\tabcolsep}{3.5pt}
\resizebox{\textwidth}{!}{
\begin{tabular}{%(colspec)s}
\hline\toprule
$Q$ & Acc (\%%) & F1 (\%%) & AUC & MSE & FAR (\%%) & MD (\%%) & Val.\ F1 (\%%) & %(pcols)s \\
\hline\toprule
""" % dict(pdesc=pdesc, pcols=pcols, colspec=colspec, qstar=qstar, df=fr["f1"]["df"], c1=fr["f1"]["chi2"], p1=fr["f1"]["p"], p2=fr["auc"]["p"],
           p3=fr["val_f1"]["p"])
    tex += "\n".join(lines) + "\n" + r"""\hline\toprule
\end{tabular}}
\end{table*}
"""
    open(os.path.join(TAB, "table_v_q_ablation.tex"), "w").write(tex)

    # ---------------- Box plot (style: figstyle, Fig. 6 reference) ----------------
    import figstyle
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    allq = qs + extra
    data = [100 * np.array([r["f1"] for r in by[q]]) for q in allq]
    bp = ax.boxplot(data, positions=range(len(allq)), widths=0.58, patch_artist=True, showfliers=False)
    cols = ["#e8112d" if q == qstar else ("#b0b0b0" if q in extra else "#8e2ce0") for q in allq]
    for med in bp["medians"]:
        med.set_color("#111111")
        med.set_zorder(5)
    ax.set_xticks(range(len(allq)))
    ax.set_xticklabels([f"{q}" + ("*" if q == qstar else "") + (f"\n(n={len(by[q])})" if q in extra else "")
                        for q in allq])
    ax.set_xlabel("Register size $Q$")
    ax.set_ylabel("Test F1 (%)")
    figstyle.panel_title(ax, f"Friedman test across Q=1-8: $\\chi^2_{{{fr['f1']['df']}}}$={fr['f1']['chi2']:.2f}, "
                         f"p={fr['f1']['p']:.2f}  (10 seeds per Q)")
    figstyle.finish(ax, grid_axis="y")
    figstyle.glossy_boxes(ax, bp["boxes"], cols)
    for i, q in enumerate(allq):
        if q in extra:
            bp["boxes"][i].set_hatch("///")
    rng = np.random.default_rng(0)
    for i, v in enumerate(data):
        figstyle.balls(ax, i + rng.uniform(-0.12, 0.12, len(v)), v, "#ffd400", s=42, zorder=6)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(IMG, f"q_ablation_boxplot.{ext}"))
    print(json.dumps({k: v for k, v in stats.items() if k != "wilcoxon"}, indent=1))
    print("Holm p vs Q*:", {q: round(v["p_holm"], 3) for q, v in pq.items()})
    print("Holm p vs Q=6:", {q: round(v["p_holm"], 3) for q, v in p6.items()})
    print("extra Q in figure:", extra)


if __name__ == "__main__":
    main()
