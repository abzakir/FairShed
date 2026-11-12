"""Correctness checks for the formulation and the quantum mapping.

Run with:  python -m pytest -q
"""

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from fairshed import Feeder, Instance, Weights, default_instance, default_weights
from fairshed.classical import brute_force_energies, greedy_policy
from fairshed.qubo import bits_to_index, breakdown, build_qubo, index_to_bits, qubo_energy
from fairshed.quantum import QAOASolver, dicke_state, qubo_to_ising


@pytest.fixture(scope="module")
def setup():
    inst, w = default_instance(), default_weights()
    Q, offset = build_qubo(inst, w)
    return inst, w, Q, offset, brute_force_energies(Q, offset)


def test_qubo_matrix_matches_direct_definition(setup):
    """x^T Q x + offset equals the term-by-term objective for all 2^n schedules."""
    inst, w, Q, offset, energies = setup
    for k in range(2**inst.n_vars):
        x = index_to_bits(k, inst.n_vars)
        direct = breakdown(inst, w, x)["total"]
        assert energies[k] == pytest.approx(direct)
        assert qubo_energy(Q, offset, x) == pytest.approx(direct)


def test_qubo_handles_unequal_loads():
    inst = Instance(
        (Feeder("a", 2, 8.0), Feeder("b", 3, 5.0), Feeder("c", 1, 3.0)), (3, 4), ("t0", "t1")
    )
    w = Weights(6.0, 4.0, 1.5)
    Q, offset = build_qubo(inst, w)
    energies = brute_force_energies(Q, offset)
    for k in range(2**inst.n_vars):
        assert energies[k] == pytest.approx(breakdown(inst, w, index_to_bits(k, inst.n_vars))["total"])


def test_index_bit_round_trip():
    for k in (0, 1, 5, 1234, 4095):
        assert bits_to_index(index_to_bits(k, 12)) == k


def test_ising_hamiltonian_diagonal_equals_qubo_energies(setup):
    """The Qiskit cost operator has exactly the QUBO energies on its diagonal."""
    inst, w, Q, offset, energies = setup
    solver = QAOASolver(inst, w, mode="penalty")
    diagonal = np.real(solver.operator.to_matrix(sparse=True).diagonal()) + solver.const
    assert np.allclose(diagonal, energies)


def test_ising_coefficients_small_example():
    # E = 3 x0 + 2 x0 x1  ->  with x = (1 - z)/2
    Q = np.array([[3.0, 2.0], [0.0, 0.0]])
    h, J, const = qubo_to_ising(Q, 0.0)
    for x0 in (0, 1):
        for x1 in (0, 1):
            z0, z1 = 1 - 2 * x0, 1 - 2 * x1
            assert h[0] * z0 + h[1] * z1 + J[(0, 1)] * z0 * z1 + const == pytest.approx(
                3 * x0 + 2 * x0 * x1
            )


def test_optimum_is_unique_and_covers_the_shortfall(setup):
    inst, w, _, _, energies = setup
    order = np.argsort(energies)
    assert energies[order[1]] - energies[order[0]] > 1e-6
    best = breakdown(inst, w, index_to_bits(int(order[0]), inst.n_vars))
    assert best["shed_per_slot"] == list(inst.deficits)
    assert best["consecutive_pairs"] == 0


def test_greedy_covers_shortfall_but_is_not_optimal(setup):
    inst, w, _, _, energies = setup
    g = breakdown(inst, w, greedy_policy(inst))
    assert g["feasible"]
    assert g["total"] > energies.min()


def test_dicke_state_is_normalised_with_fixed_weight():
    amp = dicke_state(4, 2)
    assert np.sum(amp**2) == pytest.approx(1.0)
    assert all(bin(i).count("1") == 2 for i in np.flatnonzero(amp))


@pytest.mark.parametrize("p", [1, 2])
def test_constrained_circuit_only_produces_valid_schedules(setup, p):
    """For any angles, every outcome switches off exactly D_t feeders in slot t."""
    inst, w, _, _, _ = setup
    solver = QAOASolver(inst, w, mode="constrained")
    rng = np.random.default_rng(0)
    theta = rng.uniform(-1, 1, size=2 * p)
    probs = Statevector(solver.circuit(p).assign_parameters(theta)).probabilities()
    for k in np.flatnonzero(probs > 1e-12):
        shed = breakdown(inst, w, index_to_bits(int(k), inst.n_vars))["shed_per_slot"]
        assert shed == list(inst.deficits)


def test_sampling_uses_same_bit_convention_as_brute_force(setup):
    """Aer counts and Statevector probabilities agree on which index is which."""
    inst, w, _, _, _ = setup
    solver = QAOASolver(inst, w, mode="constrained")
    result = solver.optimize(1, n_random_starts=0, maxiter=30)
    probs = solver.exact_probabilities(result)
    counts = solver.sample(result, shots=2000)
    assert all(probs[k] > 1e-9 for k in counts)
    top = max(counts, key=counts.get)
    assert probs[top] > 0.5 * probs.max()


def test_prior_outages_are_counted_in_fairness():
    """A feeder that was already off earlier today is spared by the optimum."""
    base = default_instance()
    w = default_weights()
    feeders = list(base.feeders)
    cheapest = min(range(len(feeders)), key=lambda f: feeders[f].social_cost)
    feeders[cheapest] = Feeder(feeders[cheapest].name, 1, feeders[cheapest].social_cost, prior_outages=3)
    inst = Instance(tuple(feeders), base.deficits, base.slot_labels)
    Q, offset = build_qubo(inst, w)
    energies = brute_force_energies(Q, offset)
    for k in range(0, 2**inst.n_vars, 37):
        x = index_to_bits(k, inst.n_vars)
        assert energies[k] == pytest.approx(breakdown(inst, w, x)["total"])

    def outages_on_cheapest(instance):
        q, off = build_qubo(instance, w)
        best = int(np.argmin(brute_force_energies(q, off)))
        bits = index_to_bits(best, instance.n_vars)
        return breakdown(instance, w, bits)["outages_per_feeder"][cheapest]

    assert outages_on_cheapest(inst) < outages_on_cheapest(base)


def test_aer_estimator_matches_reference_estimator(setup):
    from qiskit import transpile

    from fairshed.quantum import SIMULATOR_GATES

    inst, w, _, _, _ = setup
    theta = np.array([0.4, 0.2])
    ref = QAOASolver(inst, w, estimator="reference")
    fast = QAOASolver(inst, w, estimator="aer")
    fast_circuit = transpile(fast.circuit(1), basis_gates=SIMULATOR_GATES)
    assert fast.expectation(fast_circuit, theta) == pytest.approx(
        ref.expectation(ref.circuit(1), theta), abs=1e-6
    )
