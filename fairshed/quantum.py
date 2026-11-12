"""QAOA for FairShed, written directly on core Qiskit.

Two encodings of the same problem are implemented so they can be compared.

"penalty"  (textbook QAOA)
    Every constraint is a penalty term inside the cost Hamiltonian.
    Start state |+>^n, mixer = RX on every qubit. The circuit explores all 2^n
    bitstrings, most of which do not even cover the shortfall.

"constrained"  (constraint-preserving QAOA, the main solver)
    The hard rule "switch off exactly D_t feeder blocks in slot t" is built
    into the circuit instead of being penalised:
      * start state  = a Dicke state |D^F_{D_t}> on the qubits of each slot
                       (equal superposition of all ways to pick D_t feeders)
      * mixer        = XY (XX+YY) rotations between qubits of the same slot,
                       which conserve the number of 1s in that slot
    Every measured bitstring therefore covers the shortfall exactly, and the
    cost Hamiltonian only has to encode the soft objectives (social cost and
    fairness). This needs all feeder blocks to have the same load.

Shared pipeline
---------------
1. QUBO -> Ising:  x_i = (1 - Z_i) / 2  gives
       H_C = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j + const.
2. p layers of  exp(-i gamma_k H_C / s)  followed by the mixer with angle beta_k
   (s is a fixed energy scale that keeps gamma of order one).
3. COBYLA tunes the 2p angles to minimise <H_C>, evaluated with Qiskit's
   StatevectorEstimator primitive.
4. The tuned circuit is sampled on Qiskit Aer (optionally with noise). Every
   sampled bitstring is a candidate schedule; the lowest-energy one is returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import comb

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import StatePreparation, XXPlusYYGate
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from scipy.optimize import minimize

from .instance import Instance, Weights
from .qubo import build_qubo

BASIS_GATES = ["rz", "sx", "x", "cx"]
# Gate set the simulators evaluate natively; used only to speed up the loop.
SIMULATOR_GATES = ["rz", "rx", "ry", "rzz", "rxx", "ryy", "cx", "h", "sx", "x"]


# --------------------------------------------------------------------------
# 1. QUBO -> Ising
# --------------------------------------------------------------------------
def qubo_to_ising(Q: np.ndarray, offset: float) -> tuple[np.ndarray, dict, float]:
    """Return (h, J, const) with E(x) = sum h_i z_i + sum J_ij z_i z_j + const,
    where z_i = 1 - 2 x_i is the eigenvalue of Pauli Z on qubit i."""
    n = Q.shape[0]
    h = np.zeros(n)
    J: dict[tuple[int, int], float] = {}
    const = float(offset)
    for i in range(n):
        # q_ii * x_i = q_ii/2 - q_ii/2 * z_i
        h[i] -= Q[i, i] / 2
        const += Q[i, i] / 2
        for j in range(i + 1, n):
            q = Q[i, j]
            if q == 0:
                continue
            # q * x_i x_j = q/4 * (1 - z_i - z_j + z_i z_j)
            J[(i, j)] = q / 4
            h[i] -= q / 4
            h[j] -= q / 4
            const += q / 4
    return h, J, const


def ising_to_operator(h: np.ndarray, J: dict, n: int) -> SparsePauliOp:
    """Cost Hamiltonian as a Qiskit operator (constant term kept separately)."""
    terms = [("Z", [i], float(h[i])) for i in range(n) if h[i] != 0]
    terms += [("ZZ", [i, j], float(v)) for (i, j), v in J.items()]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n)


def energy_scale(h: np.ndarray, J: dict) -> float:
    """Root-mean-square coefficient size, used to normalise the cost layer."""
    coeffs = np.concatenate([h, np.array(list(J.values()))])
    coeffs = coeffs[coeffs != 0]
    return float(np.sqrt(np.mean(coeffs**2)))


# --------------------------------------------------------------------------
# 2. Circuit building blocks
# --------------------------------------------------------------------------
def dicke_state(n: int, k: int) -> np.ndarray:
    """Amplitudes of the Dicke state: equal superposition of all n-bit strings
    with exactly k ones."""
    amp = np.zeros(2**n)
    for idx in range(2**n):
        if bin(idx).count("1") == k:
            amp[idx] = 1.0
    return amp / np.sqrt(comb(n, k))


def add_cost_layer(qc: QuantumCircuit, h, J, gamma, scale: float) -> None:
    """exp(-i gamma H_C / scale).  RZ(t) = exp(-i t Z / 2), RZZ likewise."""
    for i in range(len(h)):
        if h[i] != 0:
            qc.rz(2 * gamma * h[i] / scale, i)
    for (i, j), v in J.items():
        qc.rzz(2 * gamma * v / scale, i, j)


@dataclass
class QAOAResult:
    mode: str
    p: int
    params: np.ndarray  # [gamma_1..gamma_p, beta_1..beta_p]
    expected_energy: float
    history: list[float] = field(default_factory=list)  # best start, per evaluation
    n_evaluations: int = 0


def _ramp(p: int) -> np.ndarray:
    """Annealing-style start: gamma ramps up, beta ramps down."""
    k = (np.arange(p) + 0.5) / p
    return np.concatenate([0.6 * k, 0.6 * (1 - k)])


def _extend(prev: np.ndarray, p: int) -> np.ndarray:
    """Start for depth p from the optimum at depth p-1 (linear interpolation)."""
    q = len(prev) // 2
    old = (np.arange(q) + 0.5) / q
    new = (np.arange(p) + 0.5) / p
    return np.concatenate([np.interp(new, old, prev[:q]), np.interp(new, old, prev[q:])])


# --------------------------------------------------------------------------
# 3. Solver
# --------------------------------------------------------------------------
class QAOASolver:
    def __init__(
        self,
        inst: Instance,
        weights: Weights,
        mode: str = "constrained",
        estimator: str = "reference",
    ):
        """estimator: "reference" = Qiskit's StatevectorEstimator (default, used
        for the reported experiments and the app); "aer" = Qiskit Aer's
        EstimatorV2 in exact statevector mode, about twice as fast at 15 qubits.
        Do not use "aer" inside Streamlit: on Windows / Python 3.14 it crashed
        the server process when called from Streamlit's script thread."""
        if mode not in ("constrained", "penalty"):
            raise ValueError("mode must be 'constrained' or 'penalty'")
        if estimator not in ("reference", "aer"):
            raise ValueError("estimator must be 'reference' or 'aer'")
        self.inst, self.weights, self.mode = inst, weights, mode
        self.estimator_kind = estimator
        self.n = inst.n_vars

        if mode == "constrained":
            loads = {fd.load for fd in inst.feeders}
            if len(loads) != 1:
                raise ValueError("constrained mode needs equal-size feeder blocks")
            block = loads.pop()
            self.cuts_per_slot = []
            for d in inst.deficits:
                if d % block or not 0 <= d // block <= inst.n_feeders:
                    raise ValueError("each deficit must be a whole number of blocks")
                self.cuts_per_slot.append(d // block)
            # The shortfall rule is enforced by the circuit, so the cost
            # Hamiltonian drops the deficit penalty (it is zero on every state
            # the circuit can reach).
            soft = Weights(0.0, weights.consecutive, weights.concentration)
            Q, offset = build_qubo(inst, soft)
        else:
            Q, offset = build_qubo(inst, weights)

        self.h, self.J, self.const = qubo_to_ising(Q, offset)
        self.operator = ising_to_operator(self.h, self.J, self.n)
        self.scale = energy_scale(self.h, self.J)
        if estimator == "aer":
            self.estimator = AerEstimator(options={
                "backend_options": {"method": "statevector"}, "default_precision": 0.0})
        else:
            self.estimator = StatevectorEstimator()

    # -- circuit ------------------------------------------------------------
    def slot_qubits(self, t: int) -> list[int]:
        return [self.inst.index(f, t) for f in range(self.inst.n_feeders)]

    def circuit(self, p: int) -> QuantumCircuit:
        """Parameterised circuit. theta = [gamma_1..gamma_p, beta_1..beta_p]."""
        theta = ParameterVector("theta", 2 * p)
        gammas, betas = theta[:p], theta[p:]
        qc = QuantumCircuit(self.n)

        if self.mode == "penalty":
            qc.h(range(self.n))
        else:
            for t, k in enumerate(self.cuts_per_slot):
                qubits = self.slot_qubits(t)
                qc.append(StatePreparation(dicke_state(len(qubits), k)), qubits)

        for layer in range(p):
            add_cost_layer(qc, self.h, self.J, gammas[layer], self.scale)
            if self.mode == "penalty":
                # exp(+i beta X): evolution under H_M = -sum X, ground state |+>^n.
                for i in range(self.n):
                    qc.rx(-2 * betas[layer], i)
            else:
                # XY mixer inside each slot: exp(+i beta (XX+YY)/2) on every
                # pair. It swaps a 1 and a 0 between two feeders, so the number
                # of feeders switched off in the slot never changes.
                for t in range(self.inst.n_slots):
                    for a, b in combinations(self.slot_qubits(t), 2):
                        qc.append(XXPlusYYGate(-2 * betas[layer]), [a, b])
        return qc

    def expectation(self, qc: QuantumCircuit, theta: np.ndarray) -> float:
        """<H_C> + const for the given angles, via the Estimator primitive."""
        result = self.estimator.run([(qc, self.operator, theta)]).result()
        return float(result[0].data.evs) + self.const

    # -- classical optimisation loop ----------------------------------------
    def optimize(
        self,
        p: int,
        warm_start: np.ndarray | None = None,
        n_random_starts: int = 4,
        maxiter: int = 300,
        seed: int = 7,
        callback=None,
    ) -> QAOAResult:
        """callback(n_evaluations, best_value_so_far) is called after every
        circuit evaluation; the app uses it to draw a progress bar."""
        qc = self.circuit(p)
        if self.estimator_kind == "aer":
            # Aer needs the Dicke preparation broken into gates it knows.
            qc = transpile(qc, basis_gates=SIMULATOR_GATES, optimization_level=1,
                           seed_transpiler=seed)
        rng = np.random.default_rng(seed + p)
        starts = [_ramp(p)]
        if warm_start is not None and len(warm_start) == 2 * (p - 1):
            starts.append(_extend(warm_start, p))
        starts += [rng.uniform(-0.8, 0.8, size=2 * p) for _ in range(n_random_starts)]

        best: QAOAResult | None = None
        total = 0
        seen = {"n": 0, "best": np.inf}
        for x0 in starts:
            history: list[float] = []

            def cost(theta: np.ndarray) -> float:
                value = self.expectation(qc, theta)
                history.append(value)
                seen["n"] += 1
                seen["best"] = min(seen["best"], value)
                if callback is not None:
                    callback(seen["n"], seen["best"])
                return value

            res = minimize(cost, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.15})
            total += len(history)
            if best is None or res.fun < best.expected_energy:
                best = QAOAResult(self.mode, p, np.array(res.x), float(res.fun), history)
        assert best is not None
        best.n_evaluations = total
        return best

    # -- sampling -----------------------------------------------------------
    def compiled(self, p: int, measure: bool = False) -> QuantumCircuit:
        qc = self.circuit(p)
        if measure:
            qc.measure_all()
        return transpile(qc, basis_gates=BASIS_GATES, optimization_level=1, seed_transpiler=7)

    def sample(
        self,
        result: QAOAResult,
        shots: int = 1024,
        seed: int = 7,
        noise_model: NoiseModel | None = None,
    ) -> dict[int, int]:
        """Measure the tuned circuit. Returns {basis-state index: count}.

        Index k has bit i equal to the measured value of qubit i, the same
        convention as classical.brute_force_energies, so energies[k] is the
        energy of the sampled schedule.
        """
        tqc = self.compiled(result.p, measure=True).assign_parameters(result.params)
        backend = AerSimulator(noise_model=noise_model)
        counts = backend.run(tqc, shots=shots, seed_simulator=seed).result().get_counts()
        # A counts key is written qubit n-1 ... qubit 0, so int(key, 2) = sum_i bit_i 2^i.
        return {int(key, 2): c for key, c in counts.items()}

    def sample_statevector(self, result: QAOAResult, shots: int = 1024, seed: int = 7) -> dict[int, int]:
        """Noise-free measurement with Qiskit's Statevector simulator (no Aer).
        Same return convention as sample(). Used by the interactive app."""
        state = Statevector(self.circuit(result.p).assign_parameters(result.params))
        state.seed(seed)
        counts = state.sample_counts(shots)
        return {int(key, 2): int(c) for key, c in counts.items()}

    def exact_probabilities(self, result: QAOAResult) -> np.ndarray:
        """Noise-free outcome distribution of the tuned circuit (infinite shots),
        from Qiskit's Statevector simulator. Entry k uses the same index
        convention as sample()."""
        qc = self.circuit(result.p).assign_parameters(result.params)
        return Statevector(qc).probabilities()

    def resources(self, p: int) -> dict:
        """Gate counts after compiling to a hardware-style basis (all-to-all)."""
        tqc = self.compiled(p)
        return {
            "qubits": self.n,
            "two_qubit_gates": int(tqc.count_ops().get("cx", 0)),
            "depth": int(tqc.depth()),
            "zz_terms": len(self.J),
            "parameters": 2 * p,
        }


def depolarizing_noise(p1: float = 0.001, p2: float = 0.01) -> NoiseModel:
    """Simple device-like noise: depolarizing error after every gate."""
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["sx", "x"])
    noise.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    return noise


# --------------------------------------------------------------------------
# Scoring a set of samples against the exact reference
# --------------------------------------------------------------------------
def score_samples(counts: dict[int, int], energies: np.ndarray, valid: np.ndarray) -> dict:
    """Score measured bitstrings against the exact enumeration of all schedules.

    `valid` marks schedules that obey the hard shortfall rule. The optimum is
    the lowest-energy valid schedule, and the returned best sample is the
    lowest-energy valid sample (any sample if none is valid)."""
    shots = sum(counts.values())
    e_min, e_max = float(energies.min()), float(energies.max())
    optimal = valid & np.isclose(energies, energies[valid].min())
    mean_e = sum(energies[k] * c for k, c in counts.items()) / shots
    pool = [k for k in counts if valid[k]] or list(counts)
    best = min(pool, key=lambda k: energies[k])
    most_likely = max(counts, key=lambda k: counts[k])
    return {
        "shots": shots,
        "distinct_bitstrings": len(counts),
        "mean_energy": float(mean_e),
        # 1.0 = always the best schedule, 0.0 = always the worst one.
        "approximation_ratio": float((e_max - mean_e) / (e_max - e_min)),
        "p_optimal": sum(c for k, c in counts.items() if optimal[k]) / shots,
        "p_feasible": sum(c for k, c in counts.items() if valid[k]) / shots,
        "best_index": int(best),
        "best_energy": float(energies[best]),
        "best_is_optimal": bool(optimal[best]),
        "most_likely_index": int(most_likely),
        "most_likely_energy": float(energies[most_likely]),
        "most_likely_is_optimal": bool(optimal[most_likely]),
    }
