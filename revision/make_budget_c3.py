"""Reviewer 1, Comment 3: QPSO search-budget sensitivity.

(a) Trained models (block B): Q in 6..10 x particles {20,40} x iterations {30,60}, seeds 42..46.
    Blocks for tests = (Q, seed) pairs (n = 25): Friedman across the four budgets; Wilcoxon
    20x30 vs 40x60 for test F1, validation F1 and best search fitness. Q* per budget = argmax
    mean validation F1 over Q (within Q 6..10 unless block B2 adds Q 1..5 at 40x60).
(b) Search only (no training): best Fisher fitness vs Q at 20x30, 40x60, 80x60 (seeds 42..46).
Outputs: paper/tables/table_budget.tex, paper/images/budget_sensitivity.png/.pdf,
         revision/outputs/stats_budget/budget_stats.json
"""
import collections
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
OUT = os.path.join(HERE, "outputs", "stats_budget")
SEEDS = [42, 43, 44, 45, 46]
BUDGETS = [(20, 30), (40, 30), (20, 60), (40, 60)]
QS = [6, 7, 8, 9, 10]


def evals(p, i):
    return p * (i + 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = {}
    for f in glob.glob(os.path.join(HERE, "outputs", "central", "qpso_ctxon_Q*_k40_s*.json")):
        r = json.load(open(f))
        c = r["config"]
        if c["seed"] not in SEEDS:
            continue
        runs[(c["Q"], c["particles"], c["iterations"], c["seed"])] = dict(
            test=r["test_metrics"]["f1_weighted"], val=r["val_metrics"]["f1_weighted"],
            fit=r["selection"]["best_fitness"], auc=r["test_metrics"]["auc_attack"])

    def vec(b, m, qs=QS):
        return np.array([runs[(q, b[0], b[1], s)][m] for q in qs for s in SEEDS])

    stats = {"n_blocks": len(QS) * len(SEEDS), "friedman": {}, "wilcoxon_20x30_vs_40x60": {}, "qstar": {}}
    for m in ["test", "val", "fit", "auc"]:
        chi2, p = friedmanchisquare(*[vec(b, m) for b in BUDGETS])
        stats["friedman"][m] = {"chi2": float(chi2), "p": float(p)}
        a, b_ = vec(BUDGETS[0], m), vec(BUDGETS[3], m)
        stats["wilcoxon_20x30_vs_40x60"][m] = {"p": float(wilcoxon(a, b_).pvalue),
                                              "mean_diff": float(np.mean(b_ - a))}
    for b in BUDGETS:
        vm = {q: np.mean([runs[(q, b[0], b[1], s)]["val"] for s in SEEDS]) for q in QS}
        stats["qstar"][f"{b[0]}x{b[1]}"] = int(max(vm, key=lambda q: (vm[q], -q)))
    # B2 (Q 1..5 at 40x60), if present: full-range Q* at the largest trained budget
    full = [q for q in range(1, 11) if all((q, 40, 60, s) in runs for s in SEEDS)]
    if len(full) == 10:
        vm = {q: np.mean([runs[(q, 40, 60, s)]["val"] for s in SEEDS]) for q in full}
        stats["qstar_full_range_40x60"] = int(max(vm, key=lambda q: (vm[q], -q)))
        stats["friedman_full_range_40x60_test"] = dict(zip(
            ["chi2", "p"], map(float, friedmanchisquare(*[[runs[(q, 40, 60, s)]["test"] for s in SEEDS]
                                                         for q in full]))))

    # search-only sweeps
    search = {}
    for (p, i), sub in [((20, 30), ""), ((40, 60), "budget_P40_I60"), ((80, 60), "budget_P80_I60")]:
        by = collections.defaultdict(list)
        for f in glob.glob(os.path.join(HERE, "outputs", "qpso_diagnostics", sub, "Q*.json")):
            r = json.load(open(f))
            if r["seed"] in SEEDS:
                by[r["Q"]].append(r["best_found_fitness"])
        search[(p, i)] = {q: (float(np.mean(v)), float(np.std(v, ddof=1))) for q, v in sorted(by.items())}
    base = search[(20, 30)]
    stats["search_gain_pct_80x60_vs_20x30"] = {
        q: 100 * (search[(80, 60)][q][0] / base[q][0] - 1) for q in base}
    stats["search_best_Q"] = {f"{p}x{i}": int(max(v, key=lambda q: v[q][0])) for (p, i), v in search.items()}
    json.dump(stats, open(os.path.join(OUT, "budget_stats.json"), "w"), indent=1)

    # ---------- table ----------
    def cell(q, b, m, scale=100, nd=2):
        v = np.array([runs[(q, b[0], b[1], s)][m] for s in SEEDS]) * scale
        return f"{v.mean():.{nd}f}$\\pm${v.std(ddof=1):.{nd}f}"

    rows = []
    for q in QS:
        for bi, b in enumerate(BUDGETS):
            lab = f"\\multirow{{4}}{{*}}{{{q}}}" if bi == 0 else ""
            rows.append(f"{lab} & {b[0]}$\\times${b[1]} & {evals(*b)} & {cell(q, b, 'fit', 1, 0)} & "
                        f"{cell(q, b, 'test')} & {cell(q, b, 'val')} \\\\")
        rows.append(r"\midrule" if q != QS[-1] else "")
    fr, wx = stats["friedman"], stats["wilcoxon_20x30_vs_40x60"]
    times = r"{\times}"
    qs_txt = ", ".join("$" + k.replace("x", times) + "$: $Q^*{=}" + str(v) + "$"
                       for k, v in stats["qstar"].items())

    def fp(p):
        for e in range(3, 9):
            if p < 10 ** -e:
                continue
            return f"{{<}}10^{{-{e - 1}}}" if e > 3 else f"{{=}}{p:.3f}"
        return "{<}10^{-8}"
    tex = (r"""\begin{table}[!t]
\centering
\scriptsize
\caption{\rev{R1.3}\textcolor{red}{QPSO search-budget sensitivity ($P$ particles $\times$ $I$ iterations;
five seeds per cell; mean$\pm$std). Fitness: best trace-ratio fitness found on $\mathcal{D}_{\rm tr}$.
Friedman test across the four budgets, blocked by ($Q$, seed), $n{=}25$: fitness $p%s$;
test F1 $p{=}%.2f$; validation F1 $p{=}%.2f$. Wilcoxon $20{\times}30$ vs.\ $40{\times}60$: fitness
$p%s$, test F1 $p{=}%.2f$. Validation-selected $Q^*$ within $Q\in\{6,\dots,10\}$ per budget: %s.}}
\label{tab:budget_sensitivity}
\color{red}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{cccccc}
\hline\toprule
$Q$ & $P{\times}I$ & Evals & Fitness & Test F1 (\%%) & Val.\ F1 (\%%) \\
\hline\toprule
""" % (fp(fr["fit"]["p"]), fr["test"]["p"], fr["val"]["p"], fp(wx["fit"]["p"]), wx["test"]["p"], qs_txt))
    tex += "\n".join(r for r in rows if r) + "\n" + r"""\hline\toprule
\end{tabular}
\end{table}
"""
    open(os.path.join(ROOT, "paper", "tables", "table_budget.tex"), "w").write(tex)

    # ---------- figure (style: figstyle, Fig. 6 reference) ----------
    import figstyle
    figstyle.apply()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.2))
    cols = {(20, 30): "#1565ff", (40, 60): "#ff7f00", (80, 60): "#e8112d"}
    for (p, i), v in search.items():
        qs = sorted(v)
        m = np.array([v[q][0] for q in qs])
        sd = np.array([v[q][1] for q in qs])
        eb = a1.errorbar(qs, m, yerr=sd, marker="o", markersize=10, capsize=5, elinewidth=2,
                         color=cols[(p, i)], markeredgecolor="#111111",
                         label=f"{p}$\\times${i} ({evals(p, i)} evals)")
        figstyle.shadow([eb.lines[0]])
    a1.set_xlabel("Register size $Q$")
    a1.set_ylabel("Best fitness found")
    figstyle.panel_title(a1, "(a) Search only, no training (5 seeds)")
    a1.set_xticks(range(1, 11))
    a1.legend()
    figstyle.finish(a1)
    mk = {(20, 30): "o", (40, 30): "s", (20, 60): "^", (40, 60): "D"}
    bc = {(20, 30): "#1565ff", (40, 30): "#18b33c", (20, 60): "#ff7f00", (40, 60): "#e8112d"}
    for bi, b in enumerate(BUDGETS):
        m = [100 * np.mean([runs[(q, b[0], b[1], s)]["test"] for s in SEEDS]) for q in QS]
        sd = [100 * np.std([runs[(q, b[0], b[1], s)]["test"] for s in SEEDS], ddof=1) for q in QS]
        eb = a2.errorbar(np.array(QS) + (bi - 1.5) * 0.08, m, yerr=sd, marker=mk[b], markersize=10,
                         capsize=5, elinewidth=2, color=bc[b], markeredgecolor="#111111",
                         label=f"{b[0]}$\\times${b[1]}", lw=2.6)
        figstyle.shadow([eb.lines[0]])
    a2.set_xlabel("Register size $Q$")
    a2.set_ylabel("Test F1 (%)")
    figstyle.panel_title(a2, f"(b) Trained detector (5 seeds); Friedman over budgets p={fr['test']['p']:.2f}")
    a2.set_xticks(QS)
    a2.legend(ncol=2)
    figstyle.finish(a2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "paper", "images", f"budget_sensitivity.{ext}"))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
