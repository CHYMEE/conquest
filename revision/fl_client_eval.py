"""Post-hoc client-specific evaluation of the final federated models (R1.8, R2.2, FL robustness).

Why: the in-run per-client evaluation sets were each client's chronologically last chunks.
ACI-IoT attacks are clustered in time, so those sets were often a single chunk of one class
whose mix differed from the client's own training data, which made worst-client F1 and
sigma(F1) unreliable. Here every client gets a test set drawn from the central test split
D_te whose class / attack-type mix matches that client's TRAINING data:

  * client partitions are rebuilt exactly (same code path and seed as federated.run_single);
  * p_k(t) = share of type t (Benign or an attack type) among the client's training windows;
  * from the test windows of each type present in D_te, n_t = round(N * p_k(t) / coverage_k)
    windows are drawn (without replacement when possible; N = 2000; seeded per run and client);
    coverage_k = sum of p_k(t) over types present in D_te is reported;
  * the run's FINAL global model (checkpoint) is evaluated once on D_te and the per-client
    metrics are computed on the client's subset.
Also reports R99 (first round with global test F1 >= 0.99 x final) and R_0.5pp (first round
within 0.5 pp of the final F1), since R90 is reached in round 1 under the corrected protocol.
Outputs: outputs/federated_clienteval/<run>.json and summary.csv
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

import common

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs", "federated_clienteval")
N_PER_CLIENT = 2000


def rebuild(cfg, ds):
    from federated import _rows, carve_eval, make_chunks, partition_chunks  # noqa: F401
    T = cfg["time_steps"]
    chunks = make_chunks(ds.segments("train"), cfg["chunk_rows"], T)
    n_train_rows = sum(e - s for s, e in chunks)
    by_time = sorted(range(len(chunks)), key=lambda i: chunks[i][0])
    proxy_ids, acc = [], 0
    step = max(1, int(round(1.0 / cfg["proxy_frac"])))
    sys_pick = by_time[step // 2::step]
    cand = sys_pick + [i for i in by_time if i not in set(sys_pick)]
    for i in cand:
        if acc >= cfg["proxy_frac"] * n_train_rows:
            break
        proxy_ids.append(i)
        acc += chunks[i][1] - chunks[i][0]
    pool = [i for i in range(len(chunks)) if i not in set(proxy_ids)]
    pool_chunks = [chunks[i] for i in pool]
    assign = partition_chunks(pool_chunks, ds.y, cfg["K"], cfg["partition"], cfg["alpha"], cfg["seed"],
                              cfg["min_chunks"])
    clients = []
    for ids in assign:
        tr_ids, _ = carve_eval(ids, pool_chunks, cfg["client_eval_frac"])
        clients.append([pool_chunks[i] for i in tr_ids])
    return clients


def metrics(y, p):
    from sklearn.metrics import f1_score
    pred = p.argmax(1)
    ben, att = y == 0, y == 1
    return {"n": int(len(y)), "attack_frac": float(att.mean()),
            "f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
            "acc": float((pred == y).mean()),
            "far": float((pred[ben] == 1).mean()) if ben.any() else None,
            "md": float((pred[att] == 0).mean()) if att.any() else None}


def evaluate_run(path, threads):
    common.configure_threads(threads, 1)
    from data import load_dataset, make_windows
    from model import build_model

    r = json.load(open(path))
    cfg = r["config"]
    name = r["run_name"]
    out = os.path.join(OUT, name + ".json")
    if os.path.exists(out):
        return json.load(open(out))
    ds = load_dataset(context=cfg["context"] == "on", n_sub=cfg["n_sub"], data_seed=cfg["data_seed"],
                      time_steps=cfg["time_steps"], block_size=cfg["block_size"])
    T = cfg["time_steps"]
    idx = r["mask"]["indices"]
    Xs = ds.X[:, idx]
    Xte, yte, end_te = make_windows(Xs, ds.y, ds.segments("test"), T)
    t_te = ds.atk[end_te]
    clients = rebuild(cfg, ds)

    z = np.load(os.path.join(os.path.dirname(path), "ckpt", name + ".npz"))
    m = build_model((T, len(idx)), 2, learning_rate=cfg["lr"], kl_weight=cfg["kl_weight"]).model
    m.set_weights([z[f"w{i}"] for i in range(int(z["n"]))])
    p_te = m.predict(Xte, batch_size=512, verbose=0)

    present = set(np.unique(t_te).tolist())
    per_client = []
    for k, segs in enumerate(clients):
        _, ytr, end_tr = make_windows(Xs, ds.y, segs, T)
        types, counts = np.unique(ds.atk[end_tr], return_counts=True)
        share = dict(zip(types.tolist(), (counts / counts.sum()).tolist()))
        cov = sum(v for t, v in share.items() if t in present)
        rng = np.random.default_rng(cfg["seed"] * 100 + k)
        pick = []
        for t, v in share.items():
            if t not in present or v <= 0:
                continue
            pool = np.flatnonzero(t_te == t)
            n_t = max(1, int(round(N_PER_CLIENT * v / cov)))
            pick.append(rng.choice(pool, size=n_t, replace=n_t > len(pool)))
        sel = np.concatenate(pick)
        mc = metrics(yte[sel], p_te[sel])
        mc.update(client=k, train_windows=int(len(ytr)), train_attack_frac=float(ytr.mean()),
                  type_coverage=float(cov))
        per_client.append(mc)

    f1s = np.array([c["f1"] for c in per_client])
    f_round = [row["test_f1"] for row in r["rounds"]]
    fin = f_round[-1]
    res = {"run_name": name, "global": metrics(yte, p_te), "clients": per_client,
           "client_f1_min": float(f1s.min()), "client_f1_mean": float(f1s.mean()),
           "client_f1_std_pop": float(f1s.std(ddof=0)),
           "client_md_max": float(max(c["md"] for c in per_client if c["md"] is not None)),
           "client_far_max": float(max(c["far"] for c in per_client if c["far"] is not None)),
           "min_type_coverage": float(min(c["type_coverage"] for c in per_client)),
           "R99": next(i + 1 for i, f in enumerate(f_round) if f >= 0.99 * fin),
           "R_half_pp": next(i + 1 for i, f in enumerate(f_round) if f >= fin - 0.005),
           "final_test_f1_logged": fin}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="fl_*.json")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    files = sorted(f for f in glob.glob(os.path.join(HERE, "outputs", "federated", a.pattern))
                   if not f.endswith("_rounds.json"))
    rows = []
    for f in files:
        res = evaluate_run(f, a.threads)
        g = res["global"]
        rows.append(f"{res['run_name']},{g['f1']:.5f},{abs(g['f1'] - res['final_test_f1_logged']):.1e},"
                    f"{res['client_f1_min']:.5f},{res['client_f1_mean']:.5f},{res['client_f1_std_pop']:.5f},"
                    f"{res['client_md_max']:.5f},{res['client_far_max']:.5f},{res['min_type_coverage']:.3f},"
                    f"{res['R99']},{res['R_half_pp']}")
        print(rows[-1], flush=True)
    with open(os.path.join(OUT, "summary.csv"), "w") as fh:
        fh.write("run,global_f1,global_f1_check_diff,client_f1_min,client_f1_mean,client_f1_std_pop,"
                 "client_md_max,client_far_max,min_type_coverage,R99,R_half_pp\n" + "\n".join(rows) + "\n")


if __name__ == "__main__":
    sys.exit(main())
