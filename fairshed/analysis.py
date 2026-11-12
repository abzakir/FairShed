"""Experiment helpers shared by run_experiments.py and the Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .classical import brute_force_energies, greedy_policy
from .instance import Instance, Weights
from .qubo import breakdown, build_qubo, index_to_bits
from .quantum import QAOAResult, QAOASolver, score_samples


@dataclass
class Reference:
    """Exact enumeration of every schedule: the ground truth for evaluation."""

    inst: Instance
    weights: Weights
    energies: np.ndarray  # objective value of every bitstring index
    covers: np.ndarray  # shortfall covered in every slot (shed >= deficit)
    exact_cover: np.ndarray  # shed == deficit in every slot
    optimum_index: int

    @property
    def best_valid_index(self) -> int:
        """Best schedule among those that cover the shortfall exactly (the hard
        rule). Equals optimum_index whenever the deficit penalty A is large
        enough, which the default weights guarantee."""
        masked = np.where(self.exact_cover, self.energies, np.inf)
        return int(np.argmin(masked))

    @property
    def optimal_mask(self) -> np.ndarray:
        """All schedules tied for best among the valid ones."""
        return self.exact_cover & np.isclose(self.energies, self.energies[self.best_valid_index])


def exact_reference(inst: Instance, w: Weights) -> Reference:
    Q, offset = build_qubo(inst, w)
    energies = brute_force_energies(Q, offset)
    n = inst.n_vars
    idx = np.arange(2**n)
    X = ((idx[:, None] >> np.arange(n)) & 1).reshape(-1, inst.n_slots, inst.n_feeders)
    loads = np.array([fd.load for fd in inst.feeders])
    shed = np.einsum("f,btf->bt", loads, X)
    deficits = np.array(inst.deficits)
    return Reference(
        inst=inst,
        weights=w,
        energies=energies,
        covers=(shed >= deficits).all(axis=1),
        exact_cover=(shed == deficits).all(axis=1),
        optimum_index=int(np.argmin(energies)),
    )


def describe(ref: Reference, index: int) -> dict:
    """Objective breakdown of one schedule plus its rank among all schedules."""
    out = breakdown(ref.inst, ref.weights, index_to_bits(index, ref.inst.n_vars))
    out["index"] = int(index)
    out["rank"] = int((ref.energies < ref.energies[index] - 1e-9).sum()) + 1
    # Rank among schedules that obey the hard shortfall rule (None if it breaks it).
    out["valid"] = bool(ref.exact_cover[index])
    out["rank_valid"] = (
        int((ref.energies[ref.exact_cover] < ref.energies[index] - 1e-9).sum()) + 1
        if out["valid"] else None
    )
    return out


def distribution_metrics(probs: np.ndarray, ref: Reference) -> dict:
    """Metrics of an exact outcome distribution (noise-free, infinite shots)."""
    e = ref.energies
    mean_e = float(probs @ e)
    optimal = ref.optimal_mask
    return {
        "mean_energy": mean_e,
        "approximation_ratio": float((e.max() - mean_e) / (e.max() - e.min())),
        "p_optimal": float(probs[optimal].sum()),
        "p_exact_cover": float(probs[ref.exact_cover].sum()),
        "most_likely_is_optimal": bool(optimal[int(np.argmax(probs))]),
    }


def run_depths(solver: QAOASolver, depths, **optimize_kwargs) -> list[QAOAResult]:
    """Optimise depth by depth, warm-starting each depth from the previous one."""
    results, prev = [], None
    for p in depths:
        res = solver.optimize(p, warm_start=prev, **optimize_kwargs)
        prev = res.params
        results.append(res)
    return results


def sample_metrics(solver: QAOASolver, result: QAOAResult, ref: Reference, **sample_kwargs) -> dict:
    counts = solver.sample(result, **sample_kwargs)
    scored = score_samples(counts, ref.energies, ref.exact_cover)
    scored["p_exact_cover"] = scored.pop("p_feasible")
    return scored


def greedy_index(inst: Instance) -> int:
    x = greedy_policy(inst)
    return int(sum(int(b) << i for i, b in enumerate(x)))
