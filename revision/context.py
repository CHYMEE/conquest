"""Causal, per-source context features F_c for CONQUEST (batch + streaming implementations).

All features are computed on the FULL chronologically sorted flow stream (before any
subsampling), grouped by Src IP.  Ordering: stable sort by Timestamp (1 s resolution);
flows with equal timestamps keep their file order, so "earlier" means "earlier position
in the sorted stream".

Causality / window convention (identical in batch and streaming code):
  * For flow i of source s at time t_i, the *history window* H_W(i) is the set of flows
    j of the same source with position j < i (strictly earlier in the sorted stream,
    i.e. the current flow is EXCLUDED) and t_j >= t_i - W  (inclusive window of W s).
  * Nothing after flow i is ever used.

Feature table (17 dims).  W in {10 s, 60 s} where marked.

  | #  | name                  | group        | window   | dim | definition                                                                                  |
  |----|-----------------------|--------------|----------|-----|---------------------------------------------------------------------------------------------|
  | 1  | n_flows_src_10        | burst        | 10 s     | 1   | |H_10(i)|  number of earlier flows from same src                                          |
  | 2  | pkt_rate_src_10       | burst        | 10 s     | 1   | sum_{j in H_10} (TotFwdPkts_j + TotBwdPkts_j) / 10                                           |
  | 3  | bytes_rate_src_10     | burst        | 10 s     | 1   | sum_{j in H_10} (TotLenFwd_j + TotLenBwd_j) / 10                                             |
  | 4  | n_flows_src_60        | burst        | 60 s     | 1   | as #1 with W = 60                                                                            |
  | 5  | pkt_rate_src_60       | burst        | 60 s     | 1   | as #2 with W = 60 (divided by 60)                                                            |
  | 6  | bytes_rate_src_60     | burst        | 60 s     | 1   | as #3 with W = 60 (divided by 60)                                                            |
  | 7  | dt_prev_src           | dormant      | -        | 1   | t_i - t_prev (s) since previous flow of same src; 1e6 s sentinel if none                     |
  | 8  | dt_prev_src_log       | dormant      | -        | 1   | log1p(dt_prev_src)                                                                           |
  | 9  | max_gap_src_60        | dormant      | 60 s     | 1   | max consecutive inter-flow gap over H_60(i) U {i}; 60 if H_60(i) is empty                    |
  | 10 | n_distinct_dst_10     | diversity    | 10 s     | 1   | number of distinct Dst IP over H_10(i)                                                       |
  | 11 | n_distinct_dport_10   | diversity    | 10 s     | 1   | number of distinct Dst Port over H_10(i)                                                     |
  | 12 | n_distinct_dst_60     | diversity    | 60 s     | 1   | as #10 with W = 60                                                                           |
  | 13 | n_distinct_dport_60   | diversity    | 60 s     | 1   | as #11 with W = 60                                                                           |
  | 14 | proto_switch          | transition   | -        | 1   | 1 if Protocol differs from the src's previous flow, else 0 (0 if no previous flow)           |
  | 15 | n_proto_switch_60     | transition   | 60 s     | 1   | sum_{j in H_60(i)} proto_switch_j                                                            |
  | 16 | syn_frac_src_60       | transition   | 60 s     | 1   | fraction of j in H_60(i) with SYN Flag Count > 0 (0 if H_60 empty)                           |
  | 17 | dport_switch          | transition   | -        | 1   | 1 if Dst Port differs from the src's previous flow, else 0 (0 if no previous flow)           |
  Total = 6 (burst) + 3 (dormant) + 4 (diversity) + 4 (transition) = 17.

Notes: max_gap uses the gaps g_j = t_j - t_prev(j) of the flows j in H_60(i) except the
oldest one (whose predecessor lies outside the window), plus the current gap dt_prev_src(i);
all such gaps are <= 60 s by construction.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List

import numpy as np

DT_SENTINEL = 1.0e6  # seconds; larger than the whole capture span (~4.3 days)
WINDOWS = (10, 60)

CONTEXT_FEATURES: List[Dict[str, object]] = [
    {"name": "n_flows_src_10", "group": "burst", "window_s": 10, "dim": 1},
    {"name": "pkt_rate_src_10", "group": "burst", "window_s": 10, "dim": 1},
    {"name": "bytes_rate_src_10", "group": "burst", "window_s": 10, "dim": 1},
    {"name": "n_flows_src_60", "group": "burst", "window_s": 60, "dim": 1},
    {"name": "pkt_rate_src_60", "group": "burst", "window_s": 60, "dim": 1},
    {"name": "bytes_rate_src_60", "group": "burst", "window_s": 60, "dim": 1},
    {"name": "dt_prev_src", "group": "dormant", "window_s": None, "dim": 1},
    {"name": "dt_prev_src_log", "group": "dormant", "window_s": None, "dim": 1},
    {"name": "max_gap_src_60", "group": "dormant", "window_s": 60, "dim": 1},
    {"name": "n_distinct_dst_10", "group": "diversity", "window_s": 10, "dim": 1},
    {"name": "n_distinct_dport_10", "group": "diversity", "window_s": 10, "dim": 1},
    {"name": "n_distinct_dst_60", "group": "diversity", "window_s": 60, "dim": 1},
    {"name": "n_distinct_dport_60", "group": "diversity", "window_s": 60, "dim": 1},
    {"name": "proto_switch", "group": "transition", "window_s": None, "dim": 1},
    {"name": "n_proto_switch_60", "group": "transition", "window_s": 60, "dim": 1},
    {"name": "syn_frac_src_60", "group": "transition", "window_s": 60, "dim": 1},
    {"name": "dport_switch", "group": "transition", "window_s": None, "dim": 1},
]
CONTEXT_NAMES = [f["name"] for f in CONTEXT_FEATURES]
N_CONTEXT = len(CONTEXT_NAMES)
assert N_CONTEXT == 17


# --------------------------------------------------------------------------------------
# Batch (vectorised) implementation
# --------------------------------------------------------------------------------------

def _range_max_table(g: np.ndarray) -> np.ndarray:
    """Sparse table for O(1) range-max queries (values are small ints)."""
    n = len(g)
    levels = max(1, int(np.floor(np.log2(max(n, 1)))) + 1)
    st = np.empty((levels, n), dtype=g.dtype)
    st[0] = g
    for k in range(1, levels):
        h = 1 << (k - 1)
        st[k] = st[k - 1]
        st[k, : n - h] = np.maximum(st[k - 1, : n - h], st[k - 1, h:])
    return st


def _range_max(st: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """max(g[lo..hi]) inclusive, requires lo <= hi."""
    length = hi - lo + 1
    k = np.floor(np.log2(length)).astype(np.int64)
    return np.maximum(st[k, lo], st[k, hi - (1 << k) + 1])


def _distinct_in_window(s: np.ndarray, v: np.ndarray, left: np.ndarray) -> np.ndarray:
    """For every position p (src-sorted space) count distinct values of v in [left[p], p).

    Element j is the first occurrence of its value inside [left, p) iff prev_j < left <= j,
    where prev_j is the previous position with the same (src, value).  Because left[] is
    non-decreasing in p, the set of p for which j counts is a contiguous interval, so the
    answer is a sum of interval indicators -> difference array + cumsum (fully vectorised).
    """
    n = len(s)
    pos = np.arange(n)
    o = np.lexsort((pos, v, s))
    prev = np.full(n, -1, dtype=np.int64)
    same = (s[o][1:] == s[o][:-1]) & (v[o][1:] == v[o][:-1])
    prev[o[1:][same]] = o[:-1][same]
    a = np.searchsorted(left, prev, side="right")  # first p with left[p] > prev_j
    b = np.searchsorted(left, pos, side="right") - 1  # last p with left[p] <= j
    lo = np.maximum(a, pos + 1)
    hi = b
    ok = lo <= hi
    diff = np.zeros(n + 1, dtype=np.int64)
    np.add.at(diff, lo[ok], 1)
    np.add.at(diff, hi[ok] + 1, -1)
    return np.cumsum(diff)[:n]


def compute_context_batch(
    t: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    dport: np.ndarray,
    proto: np.ndarray,
    syn: np.ndarray,
    pkts: np.ndarray,
    nbytes: np.ndarray,
) -> np.ndarray:
    """Compute the 17 context features for a CHRONOLOGICALLY SORTED stream.

    Args are 1-D arrays aligned with the sorted stream: t (int seconds), src/dst (int codes),
    dport, proto (numeric), syn (SYN Flag Count), pkts (fwd+bwd packets), nbytes (fwd+bwd bytes).
    Returns float64 array (n, 17) in the stream's order.
    """
    n = len(t)
    t = np.asarray(t, dtype=np.int64)
    t = t - (t.min() if n else 0)
    src = np.asarray(src, dtype=np.int64)
    o = np.lexsort((np.arange(n), src))  # by src, then stream position (stable)
    s = src[o]
    tt = t[o]
    dd = np.asarray(dst, dtype=np.int64)[o]
    dp = np.asarray(dport, dtype=np.int64)[o]
    pr = np.asarray(proto, dtype=np.int64)[o]
    sy = (np.asarray(syn)[o] > 0).astype(np.int64)
    pk = np.asarray(pkts, dtype=np.float64)[o]
    by = np.asarray(nbytes, dtype=np.float64)[o]

    big = int(tt.max() + 10_000) if n else 1
    key = s * big + tt
    pos = np.arange(n)
    gs = np.searchsorted(s, s, side="left")
    has_prev = pos > gs
    prevpos = np.where(has_prev, pos - 1, 0)

    out = np.zeros((n, N_CONTEXT), dtype=np.float64)

    def psum(x: np.ndarray) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])

    P_pk, P_by = psum(pk), psum(by)
    lefts = {}
    for wi, W in enumerate(WINDOWS):
        left = np.searchsorted(key, key - W, side="left")
        lefts[W] = left
        cnt = pos - left
        out[:, 3 * wi + 0] = cnt
        out[:, 3 * wi + 1] = (P_pk[pos] - P_pk[left]) / W
        out[:, 3 * wi + 2] = (P_by[pos] - P_by[left]) / W

    dt = np.where(has_prev, tt - tt[prevpos], DT_SENTINEL).astype(np.float64)
    out[:, 6] = dt
    out[:, 7] = np.log1p(dt)

    # max gap over window (gaps of in-window flows except the oldest, plus current gap)
    left60 = lefts[60]
    g = np.where(has_prev, np.minimum(tt - tt[prevpos], 61), 0).astype(np.int16)
    st = _range_max_table(g)
    nonempty = left60 < pos
    mg = np.full(n, 60.0)
    lo = left60[nonempty] + 1
    hi = pos[nonempty]
    mg[nonempty] = _range_max(st, lo, hi)
    out[:, 8] = mg
    del st

    out[:, 9] = _distinct_in_window(s, dd, lefts[10])
    out[:, 10] = _distinct_in_window(s, dp, lefts[10])
    out[:, 11] = _distinct_in_window(s, dd, lefts[60])
    out[:, 12] = _distinct_in_window(s, dp, lefts[60])

    psw = (has_prev & (pr != pr[prevpos])).astype(np.float64)
    dsw = (has_prev & (dp != dp[prevpos])).astype(np.float64)
    out[:, 13] = psw
    P_psw, P_syn = psum(psw), psum(sy.astype(np.float64))
    out[:, 14] = P_psw[pos] - P_psw[left60]
    cnt60 = (pos - left60).astype(np.float64)
    out[:, 15] = np.where(cnt60 > 0, (P_syn[pos] - P_syn[left60]) / np.maximum(cnt60, 1), 0.0)
    out[:, 16] = dsw

    res = np.empty_like(out)
    res[o] = out
    return res


# --------------------------------------------------------------------------------------
# Streaming (incremental) implementation - used for deployment latency L_C
# --------------------------------------------------------------------------------------

class _WinState:
    __slots__ = ("W", "q", "sum_pk", "sum_by", "dst", "dport")

    def __init__(self, W: int):
        self.W = W
        self.q: deque = deque()  # entries: (t, pk, by, dst, dport)
        self.sum_pk = 0.0
        self.sum_by = 0.0
        self.dst: Dict[int, int] = {}
        self.dport: Dict[int, int] = {}

    def expire(self, t_now: int) -> None:
        lim = t_now - self.W
        q = self.q
        while q and q[0][0] < lim:
            _, pk, by, d, p = q.popleft()
            self.sum_pk -= pk
            self.sum_by -= by
            c = self.dst[d] - 1
            if c:
                self.dst[d] = c
            else:
                del self.dst[d]
            c = self.dport[p] - 1
            if c:
                self.dport[p] = c
            else:
                del self.dport[p]

    def add(self, t: int, pk: float, by: float, d: int, p: int) -> None:
        self.q.append((t, pk, by, d, p))
        self.sum_pk += pk
        self.sum_by += by
        self.dst[d] = self.dst.get(d, 0) + 1
        self.dport[p] = self.dport.get(p, 0) + 1


class _SrcState:
    __slots__ = ("w10", "w60", "last_t", "last_proto", "last_dport", "idx", "gapq", "meta60",
                 "psw_sum", "syn_sum")

    def __init__(self) -> None:
        self.w10 = _WinState(10)
        self.w60 = _WinState(60)
        self.last_t = None
        self.last_proto = None
        self.last_dport = None
        self.idx = 0  # running index of this src's flows
        self.gapq: deque = deque()  # monotonic (idx, gap), decreasing gap
        self.meta60: deque = deque()  # (t, idx, proto_switch, syn) for H_60 entries
        self.psw_sum = 0.0  # running sum of proto_switch over meta60
        self.syn_sum = 0.0  # running count of SYN>0 over meta60


class StreamingContext:
    """Incremental per-src state over a 60 s horizon; update() returns the 17 features
    for a newly arrived flow (history = earlier flows only), then inserts the flow.
    Produces values identical to ``compute_context_batch`` on the same stream."""

    def __init__(self) -> None:
        self.state: Dict[int, _SrcState] = {}

    def update(self, t: int, src: int, dst: int, dport: int, proto: int, syn: float,
               pkts: float, nbytes: float) -> np.ndarray:
        st = self.state.get(src)
        if st is None:
            st = _SrcState()
            self.state[src] = st
        out = np.empty(N_CONTEXT, dtype=np.float64)
        st.w10.expire(t)
        st.w60.expire(t)
        lim60 = t - 60
        m = st.meta60
        while m and m[0][0] < lim60:
            _, _, psw_old, syn_old = m.popleft()
            st.psw_sum -= psw_old
            st.syn_sum -= syn_old

        for k, w in enumerate((st.w10, st.w60)):
            out[3 * k + 0] = len(w.q)
            out[3 * k + 1] = w.sum_pk / w.W
            out[3 * k + 2] = w.sum_by / w.W

        has_prev = st.last_t is not None
        dt = float(t - st.last_t) if has_prev else DT_SENTINEL
        out[6] = dt
        out[7] = np.log1p(dt)

        # max gap: gaps of in-window entries except the oldest one, plus the current gap
        cnt60 = len(m)
        if cnt60 == 0:
            out[8] = 60.0
        else:
            oldest_idx = m[0][1]
            gq = st.gapq
            while gq and gq[0][0] <= oldest_idx:
                gq.popleft()
            cur_gap = min(t - st.last_t, 61)
            out[8] = float(max(gq[0][1], cur_gap) if gq else cur_gap)

        out[9] = len(st.w10.dst)
        out[10] = len(st.w10.dport)
        out[11] = len(st.w60.dst)
        out[12] = len(st.w60.dport)

        psw = 1.0 if (has_prev and proto != st.last_proto) else 0.0
        dsw = 1.0 if (has_prev and dport != st.last_dport) else 0.0
        out[13] = psw
        out[14] = st.psw_sum
        out[15] = (st.syn_sum / cnt60) if cnt60 > 0 else 0.0
        out[16] = dsw

        # ---- insert current flow into the state ----
        gap = min(t - st.last_t, 61) if has_prev else 0
        idx = st.idx
        gq = st.gapq
        while gq and gq[-1][1] <= gap:
            gq.pop()
        gq.append((idx, gap))
        synf = 1.0 if syn > 0 else 0.0
        m.append((t, idx, psw, synf))
        st.psw_sum += psw
        st.syn_sum += synf
        st.w10.add(t, pkts, nbytes, dst, dport)
        st.w60.add(t, pkts, nbytes, dst, dport)
        st.last_t = t
        st.last_proto = proto
        st.last_dport = dport
        st.idx = idx + 1
        return out


def compute_context_streaming(t, src, dst, dport, proto, syn, pkts, nbytes) -> np.ndarray:
    sc = StreamingContext()
    n = len(t)
    res = np.empty((n, N_CONTEXT), dtype=np.float64)
    for i in range(n):
        res[i] = sc.update(int(t[i]), int(src[i]), int(dst[i]), int(dport[i]), int(proto[i]),
                           float(syn[i]), float(pkts[i]), float(nbytes[i]))
    return res


def markdown_table() -> str:
    lines = ["| # | name | group | window | dim |", "|---|---|---|---|---|"]
    for i, f in enumerate(CONTEXT_FEATURES, 1):
        w = f"{f['window_s']} s" if f["window_s"] else "-"
        lines.append(f"| {i} | {f['name']} | {f['group']} | {w} | {f['dim']} |")
    lines.append(f"| | **total** | | | **{N_CONTEXT}** |")
    return "\n".join(lines)
