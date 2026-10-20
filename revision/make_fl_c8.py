"""Federated results (R1.8 proxy/global mask, R1.9 consistent configuration, R2.2 client count).

All runs: R=30, E=2, batch 64, full participation, QPSO Q*=6 mask, seeds 42..44.
Global metrics from the run JSONs; per-client metrics from fl_client_eval.py (client-matched
test sets from D_te). Outputs: paper/tables/table_fl.tex, paper/images/fl_results.png/.pdf,
revision/outputs/stats_fl/fl_stats.json
"""
import collections
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FED = os.path.join(HERE, "outputs", "federated")
CE = os.path.join(HERE, "outputs", "federated_clienteval")
OUT = os.path.join(HERE, "outputs", "stats_fl")
CENTRAL_F1 = 0.97130  # CONQUEST centralized, ten seeds (Table IV)


def key(run):
    return run.rsplit("_s", 1)[0].replace("_qpsoQ6_ctxon", "").replace("fl_", "")


def load():
    g = collections.defaultdict(list)
    for f in glob.glob(os.path.join(FED, "fl_*.json")):
        r = json.load(open(f))
        if r.get("dp"):
            continue
        ce = json.load(open(os.path.join(CE, r["run_name"] + ".json")))
        g[key(r["run_name"])].append(dict(
            f1=r["final"]["test_f1"], auc=r["final"]["test_auc"], curve=[x["test_f1"] for x in r["rounds"]],
            cmin=ce["client_f1_min"], csd=ce["client_f1_std_pop"], mdmax=ce["client_md_max"],
            cf1=[c["f1"] for c in ce["clients"]], r99=ce["R99"], rh=ce["R_half_pp"],
            cov=ce["min_type_coverage"], round_s=r["timing"]["mean_round_time_s"]))
    return g


ROWS = [
    ("Heterogeneity (FedAvg, proxy mask, $K{=}5$)", [
        ("IID", "iid_K5_fedavg_proxy"), ("Label skew", "label_skew_K5_fedavg_proxy"),
        ("Quantity skew", "quantity_skew_K5_fedavg_proxy"),
        ("Dirichlet $\\alpha{=}0.1$", "dirichlet0.1_K5_fedavg_proxy"),
        ("Dirichlet $\\alpha{=}0.3$", "dirichlet0.3_K5_fedavg_proxy"),
        ("Dirichlet $\\alpha{=}0.5$", "dirichlet0.5_K5_fedavg_proxy"),
        ("Dirichlet $\\alpha{=}1.0$", "dirichlet1_K5_fedavg_proxy")]),
    ("Aggregator (quantity skew, $K{=}5$)", [
        ("FedAvg", "quantity_skew_K5_fedavg_proxy"), ("SCAFFOLD", "quantity_skew_K5_scaffold_proxy"),
        ("FLAME", "quantity_skew_K5_flame_proxy")]),
    ("Number of clients (quantity skew, FedAvg)", [
        ("$K{=}5$", "quantity_skew_K5_fedavg_proxy"), ("$K{=}10$", "quantity_skew_K10_fedavg_proxy"),
        ("$K{=}20$", "quantity_skew_K20_fedavg_proxy")]),
    ("Mask construction (quantity skew)", [
        ("Proxy", "quantity_skew_K5_fedavg_proxy"), ("Vote", "quantity_skew_K5_fedavg_vote"),
        ("Central (oracle)", "quantity_skew_K5_fedavg_central")]),
    ("Mask construction (Dirichlet $\\alpha{=}0.1$)", [
        ("Proxy", "dirichlet0.1_K5_fedavg_proxy"), ("Vote", "dirichlet0.1_K5_fedavg_vote"),
        ("Central (oracle)", "dirichlet0.1_K5_fedavg_central")]),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    g = load()
    summ = {}
    for k, v in g.items():
        a = lambda m: np.array([x[m] for x in v], float)
        summ[k] = {m: (float(a(m).mean()), float(a(m).std(ddof=1)) if len(v) > 1 else 0.0)
                   for m in ["f1", "auc", "cmin", "csd", "mdmax", "r99", "rh", "cov", "round_s"]}
        summ[k]["n"] = len(v)
    json.dump({"central_f1": CENTRAL_F1, "configs": summ}, open(os.path.join(OUT, "fl_stats.json"), "w"), indent=1)

    lines = []
    for block, rows in ROWS:
        lines.append(r"\multicolumn{7}{l}{\textit{" + block + r"}} \\")
        for lab, k in rows:
            s = summ[k]
            lines.append(f"{lab} & {100 * s['f1'][0]:.2f}$\\pm${100 * s['f1'][1]:.2f} & {s['auc'][0]:.4f} & "
                         f"{100 * s['cmin'][0]:.2f} & {100 * s['csd'][0]:.2f} & {100 * s['mdmax'][0]:.2f} & "
                         f"{s['r99'][0]:.1f} \\\\")
        lines.append(r"\midrule")
    lines.append(f"Centralized CONQUEST & {100 * CENTRAL_F1:.2f} & 0.9924 & -- & -- & -- & -- \\\\")
    tex = r"""\begin{table*}[!t]
\centering
\caption{\rev{R1.8}\rev{R1.9}\rev{R2.2}\textcolor{red}{Federated CONQUEST on ACI-IoT under one configuration for every
row ($R{=}30$ rounds, $E{=}2$ local epochs, batch 64, full participation, QPSO $Q^{*}{=}6$ mask, $k{=}40$; mean over
three seeds, $\pm$std for global F1). Global F1 and AUC: final global model on $\mathcal{D}_{\rm te}$. Per-client
columns: the final global model on a client-specific test set drawn from $\mathcal{D}_{\rm te}$ with the class and
attack-type mix of that client's training data. Min-F1: worst client; $\sigma$(F1): population std across clients;
Max MD: largest per-client missed-attack rate; $R_{99}$: first round reaching 99\% of the final global F1.}}
\label{tab:fl_performance_extended}
\color{red}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.05}
\begin{tabular}{lcccccc}
\hline\toprule
\textbf{Setting} & \textbf{Global F1 (\%)} & \textbf{AUC} & \textbf{Min-F1 (\%)} & \textbf{$\sigma$(F1) (pp)} & \textbf{Max MD (\%)} & \textbf{$R_{99}$} \\
\hline\toprule
""" + "\n".join(lines) + r"""
\hline\toprule
\end{tabular}
\end{table*}
"""
    open(os.path.join(ROOT, "paper", "tables", "table_fl.tex"), "w").write(tex)

    # ---------------- figure (style: figstyle, Fig. 6 reference) ----------------
    import figstyle
    figstyle.apply()
    fig, ax = plt.subplots(2, 2, figsize=(15, 10.5))
    regimes = [("IID", "iid_K5_fedavg_proxy"), ("Quantity skew", "quantity_skew_K5_fedavg_proxy"),
               ("Label skew", "label_skew_K5_fedavg_proxy"), ("Dir. 0.1", "dirichlet0.1_K5_fedavg_proxy"),
               ("Dir. 1.0", "dirichlet1_K5_fedavg_proxy")]
    rc = ["#1565ff", "#ff7f00", "#e8112d", "#18b33c", "#8e2ce0"]
    for (lab, k), c in zip(regimes, rc):
        cv = 100 * np.mean([x["curve"] for x in g[k]], axis=0)
        (ln,) = ax[0, 0].plot(range(1, len(cv) + 1), cv, label=lab, color=c, lw=3)
        figstyle.shadow([ln])
    ax[0, 0].axhline(100 * CENTRAL_F1, ls="--", color="#111111", lw=2, label="Centralized")
    ax[0, 0].set(xlabel="Round", ylabel="Global test F1 (%)")
    figstyle.panel_title(ax[0, 0], "(a) Convergence (FedAvg, K=5)")
    ax[0, 0].legend(loc="lower right")
    figstyle.finish(ax[0, 0])

    data = [100 * np.concatenate([x["cf1"] for x in g[k]]) for _, k in regimes]
    bp = ax[0, 1].boxplot(data, patch_artist=True, showfliers=True, widths=0.55,
                          flierprops=dict(marker="o", markerfacecolor="#ffd400", markeredgecolor="#111111",
                                          markersize=7))
    for med in bp["medians"]:
        med.set_color("#111111")
        med.set_zorder(5)
    ax[0, 1].set_xticks(range(1, len(regimes) + 1))
    ax[0, 1].set_xticklabels([l for l, _ in regimes])
    ax[0, 1].set(ylabel="Per-client F1 (%)")
    figstyle.panel_title(ax[0, 1], "(b) Per-client F1 (all clients x 3 seeds)")
    figstyle.finish(ax[0, 1], grid_axis="y")
    figstyle.glossy_boxes(ax[0, 1], bp["boxes"], rc)

    Ks = [5, 10, 20]
    for m, lab, mk, c in [("f1", "Global F1", "o", "#1565ff"), ("cmin", "Worst-client F1", "s", "#ff7f00")]:
        y = [100 * summ[f"quantity_skew_K{K}_fedavg_proxy"][m][0] for K in Ks]
        (ln,) = ax[1, 0].plot(Ks, y, marker=mk, markersize=12, label=lab, color=c, lw=3)
        figstyle.shadow([ln])
    ax2 = ax[1, 0].twinx()
    (ln,) = ax2.plot(Ks, [summ[f"quantity_skew_K{K}_fedavg_proxy"]["r99"][0] for K in Ks], ":^", color="#111111",
                     markerfacecolor="#18b33c", markersize=12, lw=2.6, label="$R_{99}$")
    figstyle.shadow([ln])
    ax2.set_ylabel("$R_{99}$ (rounds)")
    for sp in ax2.spines.values():
        sp.set_linewidth(1.8)
    ax[1, 0].set(xlabel="Number of clients K", ylabel="F1 (%)")
    figstyle.panel_title(ax[1, 0], "(c) Client count (quantity skew)")
    ax[1, 0].set_xticks(Ks)
    h1, l1 = ax[1, 0].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax[1, 0].legend(h1 + h2, l1 + l2, loc="center left")
    figstyle.finish(ax[1, 0])

    modes = ["proxy", "vote", "central"]
    mcol = ["#00a7a7", "#8e2ce0", "#ff3d9a"]
    w = 0.25
    bars = []
    for j, reg in enumerate(["quantity_skew_K5_fedavg", "dirichlet0.1_K5_fedavg"]):
        for i, m in enumerate(modes):
            st = summ[f"{reg}_{m}"]
            bc = ax[1, 1].bar(j + (i - 1) * w, 100 * st["f1"][0], w, yerr=100 * st["f1"][1], capsize=5,
                              error_kw=dict(elinewidth=2, capthick=2), color=mcol[i],
                              label=m.capitalize() if j == 0 else None)
            bars.append((bc[0], mcol[i]))
            ax[1, 1].text(j + (i - 1) * w, 95.08, f"{100 * st['f1'][0]:.2f}", ha="center", va="bottom",
                          fontsize=11, fontweight="bold", color="#111111", zorder=8)
            ax[1, 1].scatter(j + (i - 1) * w, 100 * st["cmin"][0], marker="D", s=110, color="#ffd400",
                             edgecolor="#111111", linewidth=1.6, zorder=9,
                             label="Worst client" if (i, j) == (0, 0) else None)
    ax[1, 1].set_xticks([0, 1])
    ax[1, 1].set_xticklabels(["Quantity skew", "Dirichlet 0.1"])
    ax[1, 1].set_ylim(95, 98.2)
    ax[1, 1].set(ylabel="F1 (%)")
    figstyle.panel_title(ax[1, 1], "(d) Mask construction")
    ax[1, 1].legend(ncol=2, loc="upper right")
    figstyle.finish(ax[1, 1], grid_axis="y")
    fig.tight_layout()
    figstyle.glossy_bars(ax[1, 1], [b for b, _ in bars], colors=[c for _, c in bars])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "paper", "images", f"fl_results.{ext}"))
    print("ok")


if __name__ == "__main__":
    main()
