"""QUBO formulation of fair load shedding.

Minimise over binary x[f, t]:

    E(x) =   sum_{f,t} c_f * x[f,t]                              (social cost)
           + A * sum_t ( sum_f P_f * x[f,t] - D_t )^2             (cover the shortfall)
           + B * sum_f sum_t x[f,t] * x[f,t+1]                    (no back-to-back outages)
           + C * sum_f ( h_f + sum_t x[f,t] )^2                   (spread outages evenly)

h_f is the number of slots feeder f was already off earlier today (0 by
default), so fairness remembers outages that happened before this plan.

Every term is at most quadratic in x, so E(x) = x^T Q x + offset with an
upper-triangular matrix Q. No slack variables are needed.

Two independent implementations live here on purpose:
  * build_qubo()   builds the matrix Q that is handed to the quantum algorithm
  * breakdown()    evaluates each term directly from its definition
The test-suite checks that both agree on every one of the 2^n schedules.
"""

from __future__ import annotations

import numpy as np

from .instance import Instance, Weights


def build_qubo(inst: Instance, w: Weights) -> tuple[np.ndarray, float]:
    """Return (Q, offset) such that E(x) = x^T Q x + offset, Q upper-triangular."""
    n = inst.n_vars
    Q = np.zeros((n, n))
    offset = 0.0
    F, T = inst.n_feeders, inst.n_slots
    loads = [fd.load for fd in inst.feeders]

    # Social cost (linear terms live on the diagonal because x^2 = x).
    for f in range(F):
        for t in range(T):
            i = inst.index(f, t)
            Q[i, i] += inst.feeders[f].social_cost

    # A * (sum_f P_f x_ft - D_t)^2
    for t in range(T):
        D = inst.deficits[t]
        offset += w.deficit * D * D
        for f in range(F):
            i = inst.index(f, t)
            Q[i, i] += w.deficit * (loads[f] ** 2 - 2 * D * loads[f])
            for g in range(f + 1, F):
                j = inst.index(g, t)
                Q[i, j] += w.deficit * 2 * loads[f] * loads[g]

    # B * x_ft * x_f,t+1
    for f in range(F):
        for t in range(T - 1):
            Q[inst.index(f, t), inst.index(f, t + 1)] += w.consecutive

    # C * (h_f + sum_t x_ft)^2
    #   = C * (h_f^2 + (1 + 2 h_f) * sum_t x_ft + 2 * sum_{t<s} x_ft x_fs)
    for f in range(F):
        prior = inst.feeders[f].prior_outages
        offset += w.concentration * prior * prior
        for t in range(T):
            i = inst.index(f, t)
            Q[i, i] += w.concentration * (1 + 2 * prior)
            for s in range(t + 1, T):
                Q[i, inst.index(f, s)] += 2 * w.concentration

    return Q, offset


def to_matrix(inst: Instance, x: np.ndarray) -> np.ndarray:
    """Reshape a flat bit vector into a (feeders x slots) schedule matrix."""
    return np.asarray(x, dtype=int).reshape(inst.n_slots, inst.n_feeders).T


def breakdown(inst: Instance, w: Weights, x: np.ndarray) -> dict:
    """Evaluate each objective term directly from its definition."""
    X = to_matrix(inst, x)
    loads = np.array([fd.load for fd in inst.feeders])
    costs = np.array([fd.social_cost for fd in inst.feeders])
    deficits = np.array(inst.deficits)

    shed = loads @ X  # load removed in each slot
    outages = X.sum(axis=1)  # number of outage slots per feeder in this plan
    prior = np.array([fd.prior_outages for fd in inst.feeders])
    burden = prior + outages  # including outages earlier today
    consecutive = int((X[:, :-1] * X[:, 1:]).sum())

    social = float(costs @ outages)
    deficit_pen = float(w.deficit * ((shed - deficits) ** 2).sum())
    consec_pen = float(w.consecutive * consecutive)
    conc_pen = float(w.concentration * (burden**2).sum())

    return {
        "social_cost": social,
        "deficit_penalty": deficit_pen,
        "consecutive_penalty": consec_pen,
        "concentration_penalty": conc_pen,
        "total": social + deficit_pen + consec_pen + conc_pen,
        "shed_per_slot": shed.tolist(),
        "outages_per_feeder": outages.tolist(),
        "burden_per_feeder": burden.tolist(),
        "consecutive_pairs": consecutive,
        "max_outages": int(outages.max()),
        # A schedule is feasible if the shortfall is covered in every slot.
        "feasible": bool((shed >= deficits).all()),
        "over_shed_units": int(np.clip(shed - deficits, 0, None).sum()),
    }


def qubo_energy(Q: np.ndarray, offset: float, x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(x @ Q @ x + offset)


def index_to_bits(k: int, n: int) -> np.ndarray:
    """Integer k -> bit vector x with x[i] = bit i of k.

    This matches Qiskit's little-endian convention: qubit i is bit i of the
    basis-state index, i.e. the i-th character from the RIGHT of a counts key.
    """
    return np.array([(k >> i) & 1 for i in range(n)], dtype=int)


def bits_to_index(x: np.ndarray) -> int:
    return int(sum(int(b) << i for i, b in enumerate(x)))
