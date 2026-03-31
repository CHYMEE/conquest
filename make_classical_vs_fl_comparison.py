# -*- coding: utf-8 -*-
"""
Generate comparison plots and tables: Classical (Centralized) vs FL Best
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Output directory for comparison artifacts
OUT_DIR = "outputs/classical_vs_fl_comparison"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================================
# Data: Classical (Centralized QPSO-LSTM-VAE+SGD) vs FL Best (FedAvg tuned)
# ============================================================================

classical = {
    "name": "Classical (Centralized)",
    "accuracy": 0.9449,
    "precision": 0.9450,
    "recall": 0.9449,
    "f1_score": 0.9449,
    "auc_attack": 0.9887,
    "training_time_sec": 1776.76,
}

fl_best = {
    "name": "FL Best (FedAvg, non-IID qty)",
    "accuracy": 0.9162,
    "precision": 0.9178,
    "recall": 0.9162,
    "f1_score": 0.9162,
    "auc_attack": 0.9708,
    "training_time_sec": 6004.03,
}

# ============================================================================
# 1) Bar chart comparison of key metrics
# ============================================================================
metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC (Attack)"]
classical_vals = [classical["accuracy"], classical["precision"], classical["recall"], classical["f1_score"], classical["auc_attack"]]
fl_vals = [fl_best["accuracy"], fl_best["precision"], fl_best["recall"], fl_best["f1_score"], fl_best["auc_attack"]]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, classical_vals, width, label="Classical (Centralized)", color="#2ecc71", edgecolor="black")
bars2 = ax.bar(x + width/2, fl_vals, width, label="FL Best (FedAvg)", color="#3498db", edgecolor="black")

ax.set_ylabel("Score", fontsize=12)
ax.set_title("Classical vs Federated Learning: Performance Comparison", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0.85, 1.0)
ax.legend(loc="lower right", fontsize=10)
ax.grid(axis="y", alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
bar_path = os.path.join(OUT_DIR, "classical_vs_fl_metrics_bar.png")
plt.savefig(bar_path, dpi=200)
plt.close()
print(f"Saved: {bar_path}")

# ============================================================================
# 2) Radar/Spider chart for multi-metric comparison
# ============================================================================
from math import pi

categories = metrics
N = len(categories)

# Normalize values to 0-1 scale (already in 0-1 for these metrics)
classical_radar = classical_vals + [classical_vals[0]]  # close the loop
fl_radar = fl_vals + [fl_vals[0]]

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

plt.xticks(angles[:-1], categories, fontsize=11)
ax.set_rlabel_position(0)
plt.yticks([0.85, 0.90, 0.95, 1.0], ["0.85", "0.90", "0.95", "1.0"], color="grey", size=9)
plt.ylim(0.80, 1.0)

ax.plot(angles, classical_radar, 'o-', linewidth=2, label="Classical (Centralized)", color="#2ecc71")
ax.fill(angles, classical_radar, alpha=0.25, color="#2ecc71")

ax.plot(angles, fl_radar, 'o-', linewidth=2, label="FL Best (FedAvg)", color="#3498db")
ax.fill(angles, fl_radar, alpha=0.25, color="#3498db")

plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
plt.title("Classical vs FL: Multi-Metric Radar Comparison", size=13, fontweight="bold", y=1.08)

radar_path = os.path.join(OUT_DIR, "classical_vs_fl_radar.png")
plt.tight_layout()
plt.savefig(radar_path, dpi=200)
plt.close()
print(f"Saved: {radar_path}")

# ============================================================================
# 3) Training time comparison (bar)
# ============================================================================
fig, ax = plt.subplots(figsize=(6, 5))
names = ["Classical\n(Centralized)", "FL Best\n(FedAvg)"]
times = [classical["training_time_sec"], fl_best["training_time_sec"]]
colors = ["#2ecc71", "#3498db"]

bars = ax.bar(names, times, color=colors, edgecolor="black", width=0.5)
ax.set_ylabel("Training Time (seconds)", fontsize=12)
ax.set_title("Training Time Comparison", fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}s\n({height/60:.1f} min)', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
time_path = os.path.join(OUT_DIR, "classical_vs_fl_training_time.png")
plt.savefig(time_path, dpi=200)
plt.close()
print(f"Saved: {time_path}")

# ============================================================================
# 4) Summary table (CSV)
# ============================================================================
import csv

summary_rows = [
    {
        "Approach": "Classical (Centralized)",
        "Accuracy": f"{classical['accuracy']:.4f}",
        "Precision": f"{classical['precision']:.4f}",
        "Recall": f"{classical['recall']:.4f}",
        "F1-Score": f"{classical['f1_score']:.4f}",
        "AUC (Attack)": f"{classical['auc_attack']:.4f}",
        "Training Time (s)": f"{classical['training_time_sec']:.2f}",
    },
    {
        "Approach": "FL Best (FedAvg, non-IID qty)",
        "Accuracy": f"{fl_best['accuracy']:.4f}",
        "Precision": f"{fl_best['precision']:.4f}",
        "Recall": f"{fl_best['recall']:.4f}",
        "F1-Score": f"{fl_best['f1_score']:.4f}",
        "AUC (Attack)": f"{fl_best['auc_attack']:.4f}",
        "Training Time (s)": f"{fl_best['training_time_sec']:.2f}",
    },
    {
        "Approach": "Difference (Classical - FL)",
        "Accuracy": f"{classical['accuracy'] - fl_best['accuracy']:+.4f}",
        "Precision": f"{classical['precision'] - fl_best['precision']:+.4f}",
        "Recall": f"{classical['recall'] - fl_best['recall']:+.4f}",
        "F1-Score": f"{classical['f1_score'] - fl_best['f1_score']:+.4f}",
        "AUC (Attack)": f"{classical['auc_attack'] - fl_best['auc_attack']:+.4f}",
        "Training Time (s)": f"{classical['training_time_sec'] - fl_best['training_time_sec']:+.2f}",
    },
]

csv_path = os.path.join(OUT_DIR, "classical_vs_fl_summary.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    w.writeheader()
    w.writerows(summary_rows)
print(f"Saved: {csv_path}")

# ============================================================================
# 5) LaTeX table
# ============================================================================
tex_path = os.path.join(OUT_DIR, "classical_vs_fl_summary.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write("\\begin{table}[t]\n")
    f.write("\\centering\n")
    f.write("\\caption{Performance Comparison: Classical (Centralized) vs Federated Learning Best}\n")
    f.write("\\label{tab:classical_vs_fl}\n")
    f.write("\\begin{tabular}{l c c c c c c}\n")
    f.write("\\hline\n")
    f.write("Approach & Acc & Prec & Rec & F1 & AUC & Time(s) \\\\ \\hline\n")
    f.write(f"Classical (Centralized) & {classical['accuracy']:.4f} & {classical['precision']:.4f} & {classical['recall']:.4f} & {classical['f1_score']:.4f} & {classical['auc_attack']:.4f} & {classical['training_time_sec']:.1f} \\\\ \n")
    f.write(f"FL Best (FedAvg) & {fl_best['accuracy']:.4f} & {fl_best['precision']:.4f} & {fl_best['recall']:.4f} & {fl_best['f1_score']:.4f} & {fl_best['auc_attack']:.4f} & {fl_best['training_time_sec']:.1f} \\\\ \n")
    f.write("\\hline\n")
    diff_acc = classical['accuracy'] - fl_best['accuracy']
    diff_prec = classical['precision'] - fl_best['precision']
    diff_rec = classical['recall'] - fl_best['recall']
    diff_f1 = classical['f1_score'] - fl_best['f1_score']
    diff_auc = classical['auc_attack'] - fl_best['auc_attack']
    diff_time = classical['training_time_sec'] - fl_best['training_time_sec']
    f.write(f"$\\Delta$ (Classical $-$ FL) & {diff_acc:+.4f} & {diff_prec:+.4f} & {diff_rec:+.4f} & {diff_f1:+.4f} & {diff_auc:+.4f} & {diff_time:+.1f} \\\\ \n")
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")
print(f"Saved: {tex_path}")

# ============================================================================
# 6) JSON summary
# ============================================================================
json_summary = {
    "classical": classical,
    "fl_best": fl_best,
    "difference": {
        "accuracy": classical["accuracy"] - fl_best["accuracy"],
        "precision": classical["precision"] - fl_best["precision"],
        "recall": classical["recall"] - fl_best["recall"],
        "f1_score": classical["f1_score"] - fl_best["f1_score"],
        "auc_attack": classical["auc_attack"] - fl_best["auc_attack"],
        "training_time_sec": classical["training_time_sec"] - fl_best["training_time_sec"],
    },
    "conclusion": "Classical (Centralized) outperforms FL Best by ~2.9% in F1-score, but FL provides privacy-preserving distributed training capability."
}

json_path = os.path.join(OUT_DIR, "classical_vs_fl_summary.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(json_summary, f, indent=2)
print(f"Saved: {json_path}")

print("\n" + "="*60)
print("COMPARISON COMPLETE")
print("="*60)
print(f"\nAll artifacts saved to: {OUT_DIR}/")
print("\nFiles generated:")
print("  - classical_vs_fl_metrics_bar.png   (bar chart)")
print("  - classical_vs_fl_radar.png         (radar/spider chart)")
print("  - classical_vs_fl_training_time.png (time comparison)")
print("  - classical_vs_fl_summary.csv       (CSV table)")
print("  - classical_vs_fl_summary.tex       (LaTeX table)")
print("  - classical_vs_fl_summary.json      (JSON summary)")
