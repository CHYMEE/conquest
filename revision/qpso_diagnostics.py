"""QPSO search-dynamics diagnostics vs register size Q (Reviewer 1, Comment 1).

Re-runs the ORIGINAL QuantumInspiredFeatureSelector loop (same methods, same RNG call
order) on the train split only and records, per iteration:
  * best_fitness   : global-best Fisher fitness so far
  * coverage       : cumulative number of distinct features measured by any particle
  * entropy        : mean normalised Shannon entropy of the measurement distribution
                     pi_i = a_i^2 / sum(a_i^2)   (1 = uniform over d features)
  * diversity      : mean pairwise Jaccard distance between the P masks measured this iteration
  * gbest_updates  : cumulative number of global-best improvements
Per run it also stores the returned mask, its fitness, and the best fitness found, and checks
that the returned mask is identical to select_qpso() for the same seed (equivalence check).

No classifier is trained. Usage:
    python qpso_diagnostics.py run   [--qs 1..10] [--seeds 42..51]
    python qpso_diagnostics.py plot
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import itertools
import json
import time

import numpy as np

import common  # noqa: F401  (sets sys.path for the repo root)
from data import load_dataset
from feature_selectors import select_qpso
from qubit_ablation_study import QuantumInspiredFeatureSelector

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "qpso_diagnostics")


def _entropy(states: np.ndarray) -> float:
    p = states ** 2
    p = p / (p.sum(axis=1, keepdims=True) + 1e-12)
    h = -(p * np.log(p + 1e-300)).sum(axis=1)
    return float(np.mean(h / np.log(states.shape[1])))


def _diversity(masks) -> float:
    sets = [set(m.tolist()) for m in masks]
    d = [1 - len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2)]
    return float(np.mean(d))


def instrumented_fit(sel: QuantumInspiredFeatureSelector, X: np.ndarray, y: np.ndarray):
    """Exact copy of QuantumInspiredFeatureSelector.fit control flow, with logging."""
    n_features = X.shape[1]
    sel.quantum_states = sel._initialize_quantum_states(n_features)
    p_best_states = sel.quantum_states.copy()
    p_best_fitness = np.zeros(sel.num_particles)
    g_best_state = sel.quantum_states[0].copy()
    g_best_fitness = 0
    seen, updates, hist = set(), 0, []

    masks = []
    for i in range(sel.num_particles):
        features = sel._measure_quantum_state(sel.quantum_states[i])
        masks.append(features)
        seen.update(features.tolist())
        fitness = sel._fitness_function(features, X, y)
        p_best_fitness[i] = fitness
        if fitness > g_best_fitness:
            g_best_fitness = fitness
            g_best_state = sel.quantum_states[i].copy()
            best_idx = np.array(features, dtype=int)
            updates += 1
    hist.append(dict(it=0, best_fitness=float(g_best_fitness), coverage=len(seen),
                     entropy=_entropy(sel.quantum_states), diversity=_diversity(masks),
                     gbest_updates=updates))

    for iteration in range(sel.iterations):
        masks = []
        for i in range(sel.num_particles):
            sel.quantum_states[i] = sel._quantum_rotation_gate(
                sel.quantum_states[i], p_best_states[i], g_best_state, iteration)
            features = sel._measure_quantum_state(sel.quantum_states[i])
            masks.append(features)
            seen.update(features.tolist())
            fitness = sel._fitness_function(features, X, y)
            if fitness > p_best_fitness[i]:
                p_best_fitness[i] = fitness
                p_best_states[i] = sel.quantum_states[i].copy()
            if fitness > g_best_fitness:
                g_best_fitness = fitness
                g_best_state = sel.quantum_states[i].copy()
                best_idx = np.array(features, dtype=int)
                updates += 1
        hist.append(dict(it=iteration + 1, best_fitness=float(g_best_fitness), coverage=len(seen),
                         entropy=_entropy(sel.quantum_states), diversity=_diversity(masks),
                         gbest_updates=updates))

    random_draw = sel._measure_quantum_state(g_best_state)  # original final-measurement behaviour
    random_draw_fitness = float(sel._fitness_function(random_draw, X, y))
    sel.best_features = best_idx  # v2: best evaluated subset is returned
    return hist, float(g_best_fitness), random_draw_fitness


def budget_dir(particles, iterations):
    """Default budget (20 x 30) keeps the original folder; other budgets get a subfolder."""
    if (particles, iterations) == (20, 30):
        return OUT_DIR
    return os.path.join(OUT_DIR, f"budget_P{particles}_I{iterations}")


def run(qs, seeds, k, particles, iterations, check_seeds):
    out = budget_dir(particles, iterations)
    os.makedirs(out, exist_ok=True)
    ds = load_dataset(context=True)
    tr = ds.rows("train")
    X, y = np.asarray(ds.X[tr], dtype=np.float64), np.asarray(ds.y[tr])
    d = X.shape[1]
    print(f"train rows={len(y)} d={d}")
    for q, s in itertools.product(qs, seeds):
        path = os.path.join(out, f"Q{q}_s{s}.json")
        if os.path.exists(path):
            continue
        t0 = time.time()
        np.random.seed(s)  # same seeding as feature_selectors.select_qpso
        sel = QuantumInspiredFeatureSelector(num_qubits=q, num_particles=particles,
                                             iterations=iterations, num_features_to_select=k)
        hist, best_found, rand_fit = instrumented_fit(sel, X, y)
        mask = sorted(int(i) for i in sel.best_features)
        rec = dict(Q=q, seed=s, k=k, d=d, particles=particles, iterations=iterations,
                   history=hist, best_found_fitness=best_found, random_draw_fitness=rand_fit,
                   mask=mask, mask_names=[ds.feature_names[i] for i in mask],
                   expected_resets_per_particle_iter=d * 2.0 ** (-q),
                   rotation_bound_rad=np.pi / (2 * q), wall_s=time.time() - t0)
        if s in check_seeds:
            ref = select_qpso(X, y, k=k, seed=s, Q=q, particles=particles, iterations=iterations)
            rec["equivalent_to_select_qpso"] = sorted(int(i) for i in ref.indices) == mask
            assert rec["equivalent_to_select_qpso"], f"instrumented loop diverged at Q={q} s={s}"
        with open(path, "w") as f:
            json.dump(rec, f)
        print(f"Q={q} seed={s} best={best_found:.1f} random_draw={rand_fit:.1f} "
              f"coverage={hist[-1]['coverage']}/{d} {rec['wall_s']:.0f}s", flush=True)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import figstyle
    figstyle.apply()
    recs = [json.load(open(os.path.join(OUT_DIR, f))) for f in os.listdir(OUT_DIR)
            if f.startswith("Q") and f.endswith(".json")]
    qs = sorted({r["Q"] for r in recs})
    d = recs[0]["d"]
    keys = [("entropy", "Amplitude entropy (norm.)"),
            ("diversity", "Swarm diversity (Jaccard dist.)"), ("best_fitness", "Best Fisher fitness")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    cmap = plt.get_cmap("turbo")
    cq = lambda qi: cmap(0.05 + 0.9 * qi / max(1, len(qs) - 1))
    summary = []
    for qi, q in enumerate(qs):
        rq = [r for r in recs if r["Q"] == q]
        for ax, (key, lab) in zip(axes, keys):
            arr = np.array([[h[key] for h in r["history"]] for r in rq])
            m, sd = arr.mean(0), arr.std(0)
            x = np.arange(arr.shape[1])
            (ln,) = ax.plot(x, m, color=cq(qi), label=f"Q={q}", lw=2.8)
            figstyle.shadow([ln])
            ax.fill_between(x, m - sd, m + sd, color=cq(qi), alpha=0.10, lw=0)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(lab)
        masks = [set(r["mask"]) for r in rq]
        jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(masks, 2)]
        summary.append(dict(
            Q=q, n=len(rq),
            coverage_final=float(np.mean([r["history"][-1]["coverage"] for r in rq])),
            entropy_final=float(np.mean([r["history"][-1]["entropy"] for r in rq])),
            diversity_final=float(np.mean([r["history"][-1]["diversity"] for r in rq])),
            best_found=float(np.mean([r["best_found_fitness"] for r in rq])),
            best_found_std=float(np.std([r["best_found_fitness"] for r in rq])),
            random_draw=float(np.mean([r["random_draw_fitness"] for r in rq])),
            gbest_updates=float(np.mean([r["history"][-1]["gbest_updates"] for r in rq])),
            mask_jaccard_across_seeds=float(np.mean(jac)) if jac else float("nan"),
            expected_resets=rq[0]["expected_resets_per_particle_iter"]))
    for ax, t in zip(axes, ["(a) Amplitude entropy", "(b) Swarm diversity", "(c) Best fitness found"]):
        figstyle.finish(ax)
        figstyle.panel_title(ax, t)
    axes[0].legend(ncol=2, fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"qpso_dynamics_vs_Q.{ext}"))
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    for s in summary:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--qs", default="1,2,3,4,5,6,7,8,9,10")
    r.add_argument("--seeds", default=",".join(str(s) for s in range(42, 52)))
    r.add_argument("--k", type=int, default=40)
    r.add_argument("--particles", type=int, default=20)
    r.add_argument("--iterations", type=int, default=30)
    r.add_argument("--check-seeds", default="42", help="seeds verified against select_qpso")
    sub.add_parser("plot")
    a = p.parse_args()
    if a.cmd == "run":
        ints = lambda s: [int(x) for x in s.split(",") if x]
        run(ints(a.qs), ints(a.seeds), a.k, a.particles, a.iterations, set(ints(a.check_seeds)))
    else:
        plot()


if __name__ == "__main__":
    main()
