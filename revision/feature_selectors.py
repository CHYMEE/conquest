"""(File named feature_selectors.py, not selectors.py: a top-level module called
`selectors` would shadow the Python stdlib module used by subprocess/asyncio.)

Feature selectors with a common interface:

    select(name, X_train, y_train, k, seed, **params) -> SelectionResult(indices, wall_time_s,
                                                                         n_fitness_evals, ...)

  * 'qpso'     : the ORIGINAL QuantumInspiredFeatureSelector from qubit_ablation_study.py
                 (imported, not rewritten): init amplitude 1/sqrt(2^Q) + N(0, 1/(Q^2+1)) noise,
                 rotation angle theta_max = pi/(2Q) decaying linearly, reset ("collapse")
                 probability 1/2^Q, Fisher-ratio fitness, top-k sampling without replacement
                 with probability ~ amplitude^2.  Params: Q, particles, iterations.
                 Fitness evaluations = particles * (iterations + 1).
  * 'pso'      : classical k-constrained binary PSO baseline with the SAME particle count,
                 iterations, Fisher fitness (the QPSO object's own _fitness_function) and the
                 SAME k-subset decoding (sampling k features without replacement with
                 probability ~ sigmoid(v)); inertia w=0.7, c1=c2=1.5, |v| <= 4 -> identical
                 budget of particles * (iterations + 1) fitness evaluations.
  * 'variance' : top-k train-only variance.
  * 'all'      : no selection.

All selectors only ever see TRAIN rows (2-D flow matrix, train-fitted MinMax scaling).
The QPSO uses NumPy's global RNG, so ``np.random.seed(seed)`` is set right before fit().
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

import common  # noqa: F401  (sys.path)


@dataclass
class SelectionResult:
    indices: List[int]
    wall_time_s: float
    n_fitness_evals: int
    best_fitness: Optional[float] = None
    fitness_history: List[float] = field(default_factory=list)
    final_mask_fitness: Optional[float] = None  # Fisher fitness of the returned subset
    extra: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


def _fisher(X: np.ndarray, y: np.ndarray, idx: np.ndarray) -> float:
    from qubit_ablation_study import QuantumInspiredFeatureSelector

    return float(QuantumInspiredFeatureSelector._fitness_function(None, np.asarray(idx), X, y))


def _best_tracking_qpso_cls():
    """Original QPSO with one change: the returned subset is the best subset actually
    evaluated (the measurement that produced g_best), as in standard PSO, instead of a fresh
    measurement of g_best_state drawn after the search (original behaviour, which returned
    subsets with ~1/3 of the best-found fitness). Search dynamics and RNG call order are
    unchanged, because the original fit() never evaluates its final measurement."""
    from qubit_ablation_study import QuantumInspiredFeatureSelector

    class BestTrackingQPSO(QuantumInspiredFeatureSelector):
        def _fitness_function(self, feature_indices, X, y):
            f = super()._fitness_function(feature_indices, X, y)
            if f > getattr(self, "_best_eval_fit", -np.inf):
                self._best_eval_fit = f
                self._best_eval_idx = np.array(feature_indices, dtype=int).copy()
            return f

    return BestTrackingQPSO


def select_qpso(X: np.ndarray, y: np.ndarray, k: int, seed: int, Q: int = 6, particles: int = 20,
                iterations: int = 30, **_) -> SelectionResult:
    np.random.seed(seed)
    sel = _best_tracking_qpso_cls()(num_qubits=Q, num_particles=particles, iterations=iterations,
                                    num_features_to_select=k)
    t0 = time.perf_counter()
    sel.fit(X, y)
    wall = time.perf_counter() - t0
    random_draw_fitness = _fisher(X, y, np.asarray(sel.best_features))  # original behaviour, logged
    sel.best_features = sel._best_eval_idx
    idx = sorted(int(i) for i in sel.best_features)
    return SelectionResult(indices=idx, wall_time_s=wall, n_fitness_evals=particles * (iterations + 1),
                           best_fitness=float(sel.fitness_history[-1]) if sel.fitness_history else None,
                           fitness_history=[float(f) for f in sel.fitness_history],
                           final_mask_fitness=_fisher(X, y, np.asarray(idx)),
                           extra={"Q": Q, "particles": particles, "iterations": iterations,
                                  "qpso_internal_time_s": float(sel.computation_time),
                                  "qpso_tracemalloc_peak_mb": float(sel.memory_usage),
                                  "random_draw_fitness": float(random_draw_fitness),
                                  "note": "final subset = best evaluated subset (v2); "
                                          "random_draw_fitness = original final-measurement behaviour"})


def select_pso(X: np.ndarray, y: np.ndarray, k: int, seed: int, particles: int = 20, iterations: int = 30,
               w: float = 0.7, c1: float = 1.5, c2: float = 1.5, vmax: float = 4.0, **_) -> SelectionResult:
    rng = np.random.default_rng(seed)
    n = X.shape[1]
    k = min(k, n)

    def decode(v: np.ndarray) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-v))
        p = p / p.sum()
        return np.sort(rng.choice(n, size=k, replace=False, p=p))

    def as_bits(idx: np.ndarray) -> np.ndarray:
        b = np.zeros(n)
        b[idx] = 1.0
        return b

    t0 = time.perf_counter()
    V = rng.uniform(-vmax, vmax, size=(particles, n)) * 0.1
    pos = [decode(V[i]) for i in range(particles)]
    fit = np.array([_fisher(X, y, p) for p in pos])
    evals = particles
    pbest = [p.copy() for p in pos]
    pbest_fit = fit.copy()
    g = int(np.argmax(fit))
    gbest, gbest_fit = pos[g].copy(), float(fit[g])
    hist = []
    for _it in range(iterations):
        for i in range(particles):
            xb, pb, gb = as_bits(pos[i]), as_bits(pbest[i]), as_bits(gbest)
            r1, r2 = rng.random(n), rng.random(n)
            V[i] = np.clip(w * V[i] + c1 * r1 * (pb - xb) + c2 * r2 * (gb - xb), -vmax, vmax)
            pos[i] = decode(V[i])
            f = _fisher(X, y, pos[i])
            evals += 1
            if f > pbest_fit[i]:
                pbest_fit[i], pbest[i] = f, pos[i].copy()
            if f > gbest_fit:
                gbest_fit, gbest = f, pos[i].copy()
        hist.append(gbest_fit)
    wall = time.perf_counter() - t0
    idx = [int(i) for i in gbest]
    return SelectionResult(indices=idx, wall_time_s=wall, n_fitness_evals=evals, best_fitness=gbest_fit,
                           fitness_history=hist, final_mask_fitness=_fisher(X, y, np.asarray(idx)),
                           extra={"particles": particles, "iterations": iterations, "w": w, "c1": c1,
                                  "c2": c2, "vmax": vmax})


def select_variance(X: np.ndarray, y: np.ndarray, k: int, seed: int, **_) -> SelectionResult:
    t0 = time.perf_counter()
    var = X.var(axis=0)
    idx = sorted(int(i) for i in np.argsort(-var, kind="stable")[: min(k, X.shape[1])])
    return SelectionResult(indices=idx, wall_time_s=time.perf_counter() - t0, n_fitness_evals=0,
                           final_mask_fitness=_fisher(X, y, np.asarray(idx)))


def select_all(X: np.ndarray, y: np.ndarray, k: int, seed: int, **_) -> SelectionResult:
    idx = list(range(X.shape[1]))
    return SelectionResult(indices=idx, wall_time_s=0.0, n_fitness_evals=0,
                           final_mask_fitness=_fisher(X, y, np.asarray(idx)))


def select_consensus(X: np.ndarray, y: np.ndarray, k: int, seed: int, Q: int = 6, particles: int = 20,
                     iterations: int = 30, source_seeds=tuple(range(42, 52)), **_) -> SelectionResult:
    """Consensus mask (Reviewer 1, Comment 4): the k features most frequently selected by the
    QPSO runs at (Q, particles, iterations, k) over ``source_seeds`` (context on). Those masks were
    fitted on D_tr only, so no validation or test information is used. Ties in frequency are
    broken by the summed best fitness of the runs that selected the feature, then by index.
    The mask does not depend on ``seed`` (which only controls model training)."""
    import json
    import os

    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "central")
    t0 = time.perf_counter()
    count = np.zeros(X.shape[1])
    fit_sum = np.zeros(X.shape[1])
    used = []
    for s in source_seeds:
        f = os.path.join(src, f"qpso_ctxon_Q{Q}_P{particles}_I{iterations}_k{k}_s{s}.json")
        r = json.load(open(f))
        if r["data"]["n_features_total"] != X.shape[1]:
            raise ValueError(f"feature layout mismatch with {f}")
        for i in r["selection"]["indices"]:
            count[i] += 1
            fit_sum[i] += r["selection"]["best_fitness"]
        used.append(int(s))
    order = sorted(range(X.shape[1]), key=lambda i: (-count[i], -fit_sum[i], i))
    idx = sorted(order[:k])
    return SelectionResult(indices=idx, wall_time_s=time.perf_counter() - t0, n_fitness_evals=0,
                           final_mask_fitness=_fisher(X, y, np.asarray(idx)),
                           extra={"Q": Q, "particles": particles, "iterations": iterations,
                                  "source_seeds": used,
                                  "selection_frequency": {int(i): float(count[i] / len(used)) for i in idx}})


SELECTORS = {"qpso": select_qpso, "pso": select_pso, "variance": select_variance, "all": select_all,
             "consensus": select_consensus}


def select(name: str, X_train: np.ndarray, y_train: np.ndarray, k: int = 40, seed: int = 0,
           **params) -> SelectionResult:
    if name not in SELECTORS:
        raise ValueError(f"unknown selector {name}; choose from {list(SELECTORS)}")
    return SELECTORS[name](np.asarray(X_train, dtype=np.float64), np.asarray(y_train), k, seed, **params)
