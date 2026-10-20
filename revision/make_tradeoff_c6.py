"""Reviewer 1 Comment 6 + Reviewer 2 Comment 3: centralized pipeline comparison and trade-offs.

Pipelines (k = 40, ten seeds 42..51, paired by seed):
  OS-CIDS           variance ranking, static features only          variance_ctxoff_k40
  Var+context       variance ranking, static + context              variance_ctxon_k40
  QPSO, no context  QPSO Q*=6, static only                          qpso_ctxoff_Q6_P20_I30_k40
  PSO+context       classical PSO, equal budget                     pso_ctxon_P20_I30_k40
  CONQUEST (QCIDS)  QPSO Q*=6, static + context                     qpso_ctxon_Q6_P20_I30_k40
FAR = benign flagged as attack / benign; MD = attacks missed / attacks (from confusion matrices).
Outputs: paper/tables/table_iv_pipelines.tex, paper/tables/table_factorial.tex,
         paper/images/tradeoff_far_md.png/.pdf, paper/images/cm_conquest_vs_os.png/.pdf,
         revision/outputs/stats_tradeoff/tradeoff_stats.json
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "outputs", "stats_tradeoff")
PIPES = [("OS-CIDS", "variance_ctxoff_k40"), ("Var.+context", "variance_ctxon_k40"),
         ("QPSO, no context", "qpso_ctxoff_Q6_P20_I30_k40"), ("PSO+context", "pso_ctxon_P20_I30_k40"),
         ("CONQUEST (QCIDS)", "qpso_ctxon_Q6_P20_I30_k40")]
REF = "CONQUEST (QCIDS)"
COL = {"OS-CIDS": "#1565ff", "Var.+context": "#00a7a7", "QPSO, no context": "#ff7f00",
       "PSO+context": "#18b33c", "CONQUEST (QCIDS)": "#e8112d"}


def load(stem):
    d = {}
    for f in glob.glob(os.path.join(HERE, "outputs", "central", f"{stem}_s*.json")):
        r = json.load(open(f))
        s = r["config"]["seed"]
        if not 42 <= s <= 51:
            continue
        cm = np.array(r["test_metrics"]["confusion_matrix"], float)
        t = r["test_metrics"]
        d[s] = dict(acc=t["accuracy"], f1=t["f1_weighted"], auc=t["auc_attack"], mse=t["mse"],
                    far=cm[0, 1] / cm[0].sum(), md=cm[1, 0] / cm[1].sum(), cm=cm / cm.sum(1, keepdims=True),
                    sel=r["timing"]["selection_time_s"], train=r["timing"]["train_time_s"] / 60,
                    val=r["val_metrics"]["f1_weighted"])
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {n: load(s) for n, s in PIPES}
    seeds = sorted(set.intersection(*[set(v) for v in data.values()]))
    assert len(seeds) == 10, seeds
    vec = lambda n, m: np.array([data[n][s][m] for s in seeds])
    stats = {"seeds": seeds, "pipelines": {}, "vs_ref": {}, "factorial": {}}
    for n, _ in PIPES:
        stats["pipelines"][n] = {m: (float(vec(n, m).mean()), float(vec(n, m).std(ddof=1)))
                                 for m in ["acc", "f1", "auc", "mse", "far", "md", "sel", "train", "val"]}
        if n != REF:
            stats["vs_ref"][n] = {m: {"diff": float(np.mean(vec(REF, m) - vec(n, m))),
                                      "p": float(wilcoxon(vec(REF, m), vec(n, m)).pvalue)}
                                  for m in ["f1", "auc", "far", "md"]}
    # 2x2: context {off,on} x selector {variance, QPSO}
    cell = {("off", "var"): "OS-CIDS", ("on", "var"): "Var.+context",
            ("off", "qpso"): "QPSO, no context", ("on", "qpso"): REF}
    f = {k: vec(v, "f1") for k, v in cell.items()}
    stats["factorial"] = {
        "context_effect_var": [float(np.mean(f["on", "var"] - f["off", "var"])),
                               float(wilcoxon(f["on", "var"], f["off", "var"]).pvalue)],
        "context_effect_qpso": [float(np.mean(f["on", "qpso"] - f["off", "qpso"])),
                                float(wilcoxon(f["on", "qpso"], f["off", "qpso"]).pvalue)],
        "selector_effect_ctxoff": [float(np.mean(f["off", "qpso"] - f["off", "var"])),
                                   float(wilcoxon(f["off", "qpso"], f["off", "var"]).pvalue)],
        "selector_effect_ctxon": [float(np.mean(f["on", "qpso"] - f["on", "var"])),
                                  float(wilcoxon(f["on", "qpso"], f["on", "var"]).pvalue)],
    }
    inter = (f["on", "qpso"] - f["off", "qpso"]) - (f["on", "var"] - f["off", "var"])
    stats["factorial"]["interaction"] = [float(inter.mean()), float(wilcoxon(inter).pvalue)]
    json.dump(stats, open(os.path.join(OUT, "tradeoff_stats.json"), "w"), indent=1)

    # ---------------- Table IV ----------------
    def ms(n, m, sc=100, nd=2):
        v = vec(n, m) * sc
        return f"{v.mean():.{nd}f}$\\pm${v.std(ddof=1):.{nd}f}"

    def pv(n, m):
        if n == REF:
            return "--"
        p = stats["vs_ref"][n][m]["p"]
        return f"{p:.3f}" if p >= 0.001 else "$<$0.001"

    rows = [r"S-CIDS~\cite{ajakweconquest}$^{\dagger}$ & 85.14 & 80.62 & -- & -- & -- & -- & -- & -- & -- \\",
            r"\midrule"]
    for n, _ in PIPES:
        lab = r"\textbf{" + n + "}" if n == REF else n
        cells = [ms(n, "acc"), ms(n, "f1"), ms(n, "auc", 1, 4), ms(n, "mse", 1, 4), ms(n, "far"), ms(n, "md"),
                 pv(n, "f1"), pv(n, "md"), f"{np.median(vec(n, 'sel')):.1f}"]
        if n == REF:
            cells = [r"\textbf{" + c + "}" if "pm" in c else c for c in cells]
        rows.append(lab + " & " + " & ".join(cells) + r" \\")
    tex = r"""\begin{table*}[!t]
\centering
\caption{\rev{R1.6}\rev{R2.3}\textcolor{red}{Centralized detection on ACI-IoT under the leakage-free protocol ($k{=}40$;
mean$\pm$std over ten seeds; QPSO at $Q^{*}{=}6$, PSO at the same budget of 620 fitness evaluations). FAR: benign
flows flagged as attack; MD: attacks missed (both \%). $p_{\rm F1}$, $p_{\rm MD}$: two-sided Wilcoxon signed-rank
$p$-values against CONQUEST, paired by seed. Sel.: median feature-selection time (s, three experiments sharing
the CPU). $^{\dagger}$Conference result~\cite{ajakweconquest}, obtained with a random split of pooled data and a single
run; quoted for reference only and not comparable.}}
\label{tab:cyberworm_high_impact}
\color{red}
\setlength{\tabcolsep}{3pt}
\resizebox{\textwidth}{!}{
\begin{tabular}{lccccccccc}
\hline\toprule
\textbf{Pipeline} & Acc (\%) & F1 (\%) & AUC & MSE & FAR (\%) & MD (\%) & $p_{\rm F1}$ & $p_{\rm MD}$ & Sel. (s) \\
\hline\toprule
""" + "\n".join(rows) + r"""
\hline\toprule
\end{tabular}}
\end{table*}
"""
    open(os.path.join(ROOT, "paper", "tables", "table_iv_pipelines.tex"), "w").write(tex)

    # ---------------- 2x2 factorial table ----------------
    def c2(n):
        return (f"{100 * vec(n, 'f1').mean():.2f}", f"{100 * vec(n, 'far').mean():.2f}",
                f"{100 * vec(n, 'md').mean():.2f}")
    fx = stats["factorial"]
    fp = lambda x: f"{100 * x[0]:+.2f}\\,pp ($p{{=}}{x[1]:.3f}$)"
    tex2 = r"""\begin{table}[!t]
\centering
\scriptsize
\caption{\rev{R1.6}\textcolor{red}{Factorial ablation separating context-aware representation from QPSO
($k{=}40$, ten seeds). Each cell: test F1 / FAR / MD (\%%). Effects are mean paired differences in F1 with
Wilcoxon $p$-values.}}
\label{tab:centralized_compare}
\color{red}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcc}
\hline\toprule
 & \textbf{Variance ranking} & \textbf{QPSO ($Q^{*}{=}6$)} \\
\hline\toprule
Static features only & %s / %s / %s & %s / %s / %s \\
Static + context & %s / %s / %s & \textbf{%s} / %s / \textbf{%s} \\
\midrule
Effect of context & %s & %s \\
Effect of QPSO & \multicolumn{2}{l}{static: %s;\ \ with context: %s} \\
Interaction & \multicolumn{2}{l}{%s} \\
\hline\toprule
\end{tabular}
\end{table}
""" % (*c2("OS-CIDS"), *c2("QPSO, no context"), *c2("Var.+context"), *c2(REF),
       fp(fx["context_effect_var"]), fp(fx["context_effect_qpso"]),
       fp(fx["selector_effect_ctxoff"]), fp(fx["selector_effect_ctxon"]), fp(fx["interaction"]))
    open(os.path.join(ROOT, "paper", "tables", "table_factorial.tex"), "w").write(tex2)

    # ---------------- FAR vs MD scatter (style: figstyle, Fig. 6 reference) ----------------
    import figstyle
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for n, _ in PIPES:
        x, y = 100 * vec(n, "far"), 100 * vec(n, "md")
        figstyle.balls(ax, x, y, COL[n], s=70, zorder=3, alpha=0.75)
        h = ax.scatter([x.mean()], [y.mean()], s=260, color=COL[n], edgecolor="#111111", linewidth=2.0,
                       marker="D", label=n, zorder=6)
        figstyle.shadow([h])
    ax.set_xlabel("False alarms: benign flagged as attack (%)")
    ax.set_ylabel("Missed attacks (%)")
    figstyle.finish(ax)
    ax.legend(loc="upper right", markerscale=0.8)
    figstyle.panel_title(ax, "Error trade-off per pipeline (dots: seeds; diamonds: means)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "paper", "images", f"tradeoff_far_md.{ext}"))
    plt.close(fig)

    # ---------------- mean normalized confusion matrices ----------------
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 4.0))
    for ax, n in zip(axs, ["OS-CIDS", REF]):
        m = np.mean([data[n][s]["cm"] for s in seeds], axis=0) * 100
        ax.imshow(m, cmap=figstyle.bright_seq(), vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{m[i, j]:.2f}%", ha="center", va="center", fontsize=17, fontweight="bold",
                        color="white" if m[i, j] > 50 else "#111111")
        for k in (0.5,):
            ax.axhline(k, color="#111111", lw=2.0)
            ax.axvline(k, color="#111111", lw=2.0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Benign", "Attack"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Benign", "Attack"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for sp in ax.spines.values():
            sp.set_linewidth(2.2)
        figstyle.panel_title(ax, ("(a) " if n == "OS-CIDS" else "(b) ") + n)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "paper", "images", f"cm_conquest_vs_os.{ext}"))
    print(json.dumps({k: v for k, v in stats.items() if k != "seeds"}, indent=1))


if __name__ == "__main__":
    main()
