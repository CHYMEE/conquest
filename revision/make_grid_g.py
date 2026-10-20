"""Builds grids/G.json (R2.1, client-level DP-FedAvg) with the clip norm C fixed BEFORE any DP run:
C = median client-update L2 norm over all rounds of the non-private FedAvg runs (quantity skew,
K=5, proxy mask, seeds 42-44) from block E. No DP-run data is used to choose C."""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
norms = []
for f in glob.glob(os.path.join(HERE, "outputs", "federated", "fl_quantity_skew_K5_fedavg_proxy_qpsoQ*_ctxon_s4[2-4].json")):
    r = json.load(open(f))
    for row in r["rounds"]:
        norms += row["agg_info"]["update_norms"]
if not norms:
    raise SystemExit("no non-private FedAvg runs found to calibrate C")
C = float(np.round(np.median(norms), 3))
grid = {"blocks": [{"out_dir": "outputs/federated",
                    "base": {"selector": "qpso", "Q": "auto:outputs/central", "particles": 20, "iterations": 30,
                             "k": 40, "context": "on", "rounds": 30, "local_epochs": 2, "batch_size": 64, "K": 5,
                             "aggregator": "fedavg", "partition": "quantity_skew", "mask_mode": "proxy",
                             "dp_clip": C},
                    "grid": {"dp_noise": [1.0, 3.0, 8.0], "seed": [42, 43, 44]}}]}
json.dump(grid, open(os.path.join(HERE, "grids", "G.json"), "w"), indent=1)
print(f"C = {C} (median of {len(norms)} update norms)")
