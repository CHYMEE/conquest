"""Shared helpers for the CONQUEST revision pipeline (paths, seeding, threading, I/O).

Import this module FIRST (before tensorflow) and call ``configure_threads`` so that
TF intra/inter-op thread counts take effect for parallel worker processes.
"""
from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
from typing import Any, Dict, List

REV_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(REV_DIR)
CACHE_DIR = os.path.join(REV_DIR, "cache")
OUT_DIR = os.path.join(REV_DIR, "outputs")
CSV_PATH = os.path.join(REPO_DIR, "0.ACI-IoT-2023.csv")

# Make the legacy modules (qubit_ablation_study.py) importable without modification.
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

DATA_SEED_DEFAULT = 42  # subsample + split are FIXED across all runs (coordinator spec)


def configure_threads(intra: int, inter: int) -> None:
    """Must be called before TensorFlow is imported for the env vars to apply."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(intra)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(inter)
    os.environ["OMP_NUM_THREADS"] = str(intra)
    os.environ["MKL_NUM_THREADS"] = str(intra)
    os.environ["OPENBLAS_NUM_THREADS"] = str(intra)
    import tensorflow as tf  # noqa: WPS433

    try:
        tf.config.threading.set_intra_op_parallelism_threads(intra)
        tf.config.threading.set_inter_op_parallelism_threads(inter)
    except RuntimeError:
        # Runtime already initialised (e.g. called twice) - env vars still documented.
        pass


def set_seed(seed: int) -> None:
    import numpy as np
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        import keras

        keras.utils.set_random_seed(seed)
    except Exception:
        pass


def ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)


def to_serializable(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_serializable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def write_json(path: str, payload: Any) -> None:
    """Atomic JSON write (tmp + rename) so that resumable sweeps never see half files."""
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(to_serializable(payload), f, indent=2)
    os.replace(tmp, path)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    import csv

    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(to_serializable(r))


class RSSMonitor:
    """Samples process RSS (psutil) in a background thread; reports the peak in MB.

    Used separately for the training phase and the inference phase, so the two peaks
    are not confounded (psutil's OS-level peak_wset cannot be reset on Windows).
    """

    def __init__(self, interval: float = 0.05):
        import psutil

        self._proc = psutil.Process(os.getpid())
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, self._proc.memory_info().rss)
            except Exception:
                pass
            time.sleep(self.interval)

    def __enter__(self) -> "RSSMonitor":
        self.peak = self._proc.memory_info().rss
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._t is not None:
            self._t.join()
        self.peak = max(self.peak, self._proc.memory_info().rss)

    @property
    def peak_mb(self) -> float:
        return self.peak / (1024 * 1024)
