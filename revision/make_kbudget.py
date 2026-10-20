"""Feature-budget scenario (R1.4 compactness, R1.6/R2.3 trade-off): k in {10, 20, 40} x selector.

Context features on; QPSO Q*=6 and PSO at 20x30; five paired seeds (42..46) for all cells, plus
the consensus mask at k=40 over ten seeds. Wilcoxon with n=5 cannot fall below p=0.0625, so the
table reports how many of the five seeds favour QPSO instead of p-values.
Output: paper/tables/table_kbudget.tex
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SEEDS = [42, 43, 44, 45, 46]


def load(stem):
    d = {}
    for f in glob.glob(os.path.join(HERE, "outputs", "central", f"{stem}_s*.json")):
        r = json.load(open(f))
        cm = np.array(r["test_metrics"]["confusion_matrix"], float)
        d[r["config"]["seed"]] = dict(f1=r["test_metrics"]["f1_weighted"], far=cm[0, 1] / cm[0].sum(),
                                      md=cm[1, 0] / cm[1].sum(), ctx=r["selection"].get("n_context_selected"))
    return d


def main():
    rows = []
    for k in (40, 20, 10):
        cells = {"QPSO": load(f"qpso_ctxon_Q6_P20_I30_k{k}"), "PSO": load(f"pso_ctxon_P20_I30_k{k}"),
                 "Variance": load(f"variance_ctxon_k{k}")}
        best = max(cells, key=lambda n: np.mean([cells[n][s]["f1"] for s in SEEDS]))
        for i, (n, c) in enumerate(cells.items()):
            v = lambda m: np.array([c[s][m] for s in SEEDS])
            f1 = f"{100 * v('f1').mean():.2f}$\\pm${100 * v('f1').std(ddof=1):.2f}"
            if n == best:
                f1 = r"\textbf{" + f1 + "}"
            if n == "QPSO":
                win = "--"
            else:
                win = f"{int(np.sum([cells['QPSO'][s]['f1'] > c[s]['f1'] for s in SEEDS]))}/5"
            lab = f"\\multirow{{3}}{{*}}{{{k}}}" if i == 0 else ""
            rows.append(f"{lab} & {n} & {f1} & {100 * v('far').mean():.2f} & {100 * v('md').mean():.2f} & "
                        f"{v('ctx').mean():.1f} & {win} \\\\")
        rows.append(r"\midrule")
    cons = load("consensus_ctxon_Q6_P20_I30_k40")
    s10 = sorted(cons)
    v = lambda m: np.array([cons[s][m] for s in s10])
    rows.append(f"40 & Consensus mask ({len(s10)} seeds) & {100 * v('f1').mean():.2f}$\\pm${100 * v('f1').std(ddof=1):.2f} & "
                f"{100 * v('far').mean():.2f} & {100 * v('md').mean():.2f} & {v('ctx').mean():.1f} & -- \\\\")
    tex = r"""\begin{table}[!t]
\centering
\scriptsize
\caption{\rev{R1.4}\rev{R1.6}\textcolor{red}{Feature-budget scenario: test F1 (\%), FAR and MD (\%), and number
of the 17 context features selected, for $k\in\{40,20,10\}$ (context features available; five paired seeds;
QPSO $Q^{*}{=}6$, PSO at an equal budget). Wins: seeds on which QPSO has the higher F1. Bold: best F1 per $k$.
Last row: consensus mask of Table~\ref{tab:selected_features} over ten seeds.}}
\label{tab:kbudget}
\color{red}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{clccccc}
\hline\toprule
$k$ & Selector & F1 & FAR & MD & Ctx. & Wins \\
\hline\toprule
""" + "\n".join(rows) + r"""
\hline\toprule
\end{tabular}
\end{table}
"""
    open(os.path.join(ROOT, "paper", "tables", "table_kbudget.tex"), "w").write(tex)
    print(tex)


if __name__ == "__main__":
    main()
