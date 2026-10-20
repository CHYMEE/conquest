"""Deployment cost (Reviewer 1 Comment 10, Reviewer 2 Comment 4): table + figure.

Isolated measurements (``centralized.py deploy``: idle CPU host, 2 intra-op threads) stored as
``deployment_isolated`` in each run JSON. Per pipeline the median over the measured seeds is
reported. Units: per-window latency at batch size 1 (ms/window, median and p95 over 1,000 test
windows after 50 warm-up calls); batched cost at B=64 (ms/window); whole-test-set time
(s, N_te windows); context-feature update (us/flow); peak resident memory of the inference
process (MB). Feature-selection time is a training-time cost and is reported separately.
Outputs: paper/tables/table_latency.tex, paper/images/deployment_cost.png/.pdf
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ROWS = [("OS-CIDS", "variance_ctxoff_k40"), ("Var.+context", "variance_ctxon_k40"),
        ("PSO+context", "pso_ctxon_P20_I30_k40"), ("CONQUEST ($Q^{*}{=}6$)", "qpso_ctxon_Q6_P20_I30_k40"),
        ("CONQUEST, consensus mask", "consensus_ctxon_Q6_P20_I30_k40"),
        ("CONQUEST, $Q{=}1$", "qpso_ctxon_Q1_P20_I30_k40"), ("CONQUEST, $Q{=}8$", "qpso_ctxon_Q8_P20_I30_k40"),
        ("CONQUEST, $k{=}20$", "qpso_ctxon_Q6_P20_I30_k20"), ("CONQUEST, $k{=}10$", "qpso_ctxon_Q6_P20_I30_k10")]
SHORT = ["OS-CIDS", "Var.+context", "PSO+context", "CONQUEST", "Consensus mask", "Q = 1", "Q = 8", "k = 20", "k = 10"]


def collect():
    out = []
    for lab, stem in ROWS:
        v = []
        for f in sorted(glob.glob(os.path.join(HERE, "outputs", "central", f"{stem}_s*.json"))):
            r = json.load(open(f))
            d = r.get("deployment_isolated")
            if not d:
                continue
            v.append([d["latency_b1"]["median_ms"], d["latency_b1"]["p95_ms"], d["throughput_b64"]["ms_per_sample"],
                      d["throughput_b64"]["samples_per_s"], d["T_infer_te_s"], d["throughput_b64"]["n_samples"],
                      1000 * d["L_C_per_flow"]["median_ms"], d["peak_rss_inference_mb"],
                      r["model"]["params"]["trainable"], r["timing"]["selection_time_s"]])
        a = np.median(np.array(v), axis=0)
        out.append((lab, len(v), a))
    return out


def main():
    rows = collect()
    lines = []
    for lab, n, a in rows:
        bold = lab.startswith("CONQUEST ($Q")
        cells = [f"{a[0]:.2f}", f"{a[1]:.2f}", f"{a[2]:.3f}", f"{a[4]:.2f}", f"{a[6]:.1f}", f"{a[7]:.0f}",
                 f"{a[8] / 1000:.0f}k", f"{a[9]:.1f}"]
        if bold:
            cells = [r"\textbf{" + c + "}" for c in cells]
            lab = r"\textbf{" + lab + "}"
        lines.append(f"{lab} ({n}) & " + " & ".join(cells) + r" \\")
    n_te = int(rows[3][2][5])
    tex = r"""\begin{table*}[!t]
\centering
\caption{\rev{R1.10}\rev{R2.4}\textcolor{red}{Deployment cost measured in isolation; medians over seeds, with the number of seeds in parentheses.}}
\label{tab:latency_profile_qcids}
\color{red}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccccc}
\hline\toprule
\textbf{Pipeline} & \multicolumn{2}{c}{Latency (ms/window)} & Batched & Test set & Context update & Memory & Params & Selection \\
 & median & p95 & (ms/window) & (s) & ($\mu$s/flow) & (MB) & & (s) \\
\hline\toprule
""" + "\n".join(lines) + r"""
\hline\toprule
\end{tabular}
\end{table*}
"""
    open(os.path.join(ROOT, "paper", "tables", "table_latency.tex"), "w").write(tex)

    import figstyle
    figstyle.apply()
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.4))
    x = np.arange(len(rows))
    cols = ["#1565ff", "#00a7a7", "#18b33c", "#e8112d", "#ff3d9a", "#8e2ce0", "#8e2ce0", "#ff7f00", "#ff7f00"]
    med = [a[0] for _, _, a in rows]
    p95 = [a[1] for _, _, a in rows]
    b0 = ax[0].bar(x, med, 0.62, color=cols)
    ax[0].scatter(x, p95, marker="D", s=90, color="#ffd400", edgecolor="#111111", linewidth=1.5, zorder=8, label="p95")
    for xi, m in zip(x, med):
        ax[0].text(xi, m + 0.12, f"{m:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold", zorder=9)
    ax[0].set_ylabel("ms per window (batch 1)")
    ax[0].set_ylim(0, max(p95) * 1.25)
    ax[0].legend(loc="upper left")
    figstyle.panel_title(ax[0], "(a) Per-window detection latency")
    thr = [a[3] for _, _, a in rows]
    b1 = ax[1].bar(x, thr, 0.62, color=cols)
    for xi, t in zip(x, thr):
        ax[1].text(xi, t + 60, f"{t:.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold", zorder=9)
    ax[1].set_ylabel("windows per second (B=64)")
    ax[1].set_ylim(0, max(thr) * 1.18)
    figstyle.panel_title(ax[1], "(b) Batched throughput")
    mem = [a[7] for _, _, a in rows]
    b2 = ax[2].bar(x, mem, 0.62, color=cols)
    ax[2].axhline(1024, ls="--", color="#e8112d", lw=2.4, label="1 GB edge budget")
    for xi, m in zip(x, mem):
        ax[2].text(xi, m + 15, f"{m:.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold", zorder=9)
    ax[2].set_ylabel("peak inference memory (MB)")
    ax[2].set_ylim(0, 1150)
    ax[2].legend(loc="upper left")
    figstyle.panel_title(ax[2], "(c) Peak memory")
    for a_ in ax:
        a_.set_xticks(x)
        a_.set_xticklabels(SHORT, rotation=35, ha="right")
        figstyle.finish(a_, grid_axis="y")
        # bars 5-9 are variants of CONQUEST (index 3): other mask, register size, or feature budget
        a_.axvspan(3.5, 8.5, color="#d9d9d9", alpha=0.45, zorder=0, lw=0)
        a_.text(6.0, 0.97, "CONQUEST variants", transform=a_.get_xaxis_transform(), ha="center", va="top",
                fontsize=12, fontweight="bold", color="#333333", zorder=10)
    fig.tight_layout()
    for a_, bars in zip(ax, (b0, b1, b2)):
        figstyle.glossy_bars(a_, bars, colors=cols)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "paper", "images", f"deployment_cost.{ext}"))
    print(tex)


if __name__ == "__main__":
    main()
