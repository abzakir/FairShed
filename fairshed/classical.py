"""Classical reference solvers.

* brute_force_energies: exact. Evaluates all 2^n schedules, so the true optimum
  is known and the quantum result can be verified against it.
* greedy_policy: a fixed priority list, the way load shedding is commonly run
  in practice (always cut the "cheapest" feeders first).
"""

from __future__ import annotations

import numpy as np

from .instance import Instance


def brute_force_energies(Q: np.ndarray, offset: float) -> np.ndarray:
    """Energy of every bitstring. Entry k is the energy of index_to_bits(k).

    Because index k uses Qiskit's little-endian bit order, this array is also
    exactly the diagonal of the cost Hamiltonian in the computational basis.
    """
    n = Q.shape[0]
    idx = np.arange(2**n)
    X = ((idx[:, None] >> np.arange(n)) & 1).astype(float)
    return np.einsum("bi,ij,bj->b", X, Q, X) + offset


def greedy_policy(inst: Instance) -> np.ndarray:
    """Per slot, switch feeders off in order of social cost per unit of load
    until the shortfall is covered. Ignores fairness entirely."""
    order = sorted(
        range(inst.n_feeders),
        key=lambda f: inst.feeders[f].social_cost / inst.feeders[f].load,
    )
    x = np.zeros(inst.n_vars, dtype=int)
    for t, deficit in enumerate(inst.deficits):
        shed = 0
        for f in order:
            if shed >= deficit:
                break
            x[inst.index(f, t)] = 1
            shed += inst.feeders[f].load
    return x
