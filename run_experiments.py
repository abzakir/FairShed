"""Reproduce every number and figure in the report.

    python run_experiments.py            # full run (a few minutes)
    python run_experiments.py --quick    # p = 1 only, skips the 15-qubit check

Outputs: figures/*.png and results/results.json. All random seeds are fixed.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from fairshed import default_instance, default_weights, larger_instance
from fairshed import plots
from fairshed.analysis import (
    describe,
    distribution_metrics,
    exact_reference,
    greedy_index,
    run_depths,
    sample_metrics,
)
from fairshed.quantum import QAOASolver, depolarizing_noise

ROOT = Path(__file__).parent
FIG, RES = ROOT / "figures", ROOT / "results"
SHOTS = 1024


def brief(d: dict) -> str:
    return (f"objective {d['total']:.1f} | social cost {d['social_cost']:.1f} | "
            f"worst feeder off {d['max_outages']} slots | back-to-back {d['consecutive_pairs']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    depths = (1,) if args.quick else (1, 2, 3)
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    t_start = time.time()
    out: dict = {"shots": SHOTS, "depths": list(depths)}

    # ---------------------------------------------------------------- 1. baselines
    inst, w = default_instance(), default_weights()
    ref = exact_reference(inst, w)
    exact = describe(ref, ref.best_valid_index)
    greedy = describe(ref, greedy_index(inst))
    n_valid = int(ref.exact_cover.sum())
    out["instance"] = {
        "feeders": [vars(fd) for fd in inst.feeders], "deficits": list(inst.deficits),
        "slots": list(inst.slot_labels), "weights": vars(w), "qubits": inst.n_vars,
        "schedules_total": int(len(ref.energies)), "schedules_valid": n_valid,
    }
    out["exact"], out["greedy"] = exact, greedy
    print(f"Instance: {inst.n_feeders} feeders x {inst.n_slots} slots = {inst.n_vars} qubits, "
          f"{len(ref.energies)} schedules, {n_valid} cover the shortfall exactly")
    print(f"  exact optimum : {brief(exact)}")
    print(f"  greedy policy : {brief(greedy)}  (rank {greedy['rank']} of {len(ref.energies)})")

    # ---------------------------------------------------------------- 2. QAOA, both encodings
    noise = depolarizing_noise()
    rows, results = [], {}
    for mode in ("constrained", "penalty"):
        solver = QAOASolver(inst, w, mode)
        t0 = time.time()
        results[mode] = run_depths(solver, depths)
        for res in results[mode]:
            ideal = distribution_metrics(solver.exact_probabilities(res), ref)
            shots = sample_metrics(solver, res, ref, shots=SHOTS)
            noisy = sample_metrics(solver, res, ref, shots=4 * SHOTS, noise_model=noise)
            row = {
                "mode": mode, "p": res.p, "params": res.params.tolist(),
                "evaluations": res.n_evaluations, **ideal,
                "sampled_best_energy": shots["best_energy"],
                "sampled_best_is_optimal": shots["best_is_optimal"],
                "sampled_best_index": shots["best_index"],
                "sampled_distinct": shots["distinct_bitstrings"],
                "noisy_p_optimal": noisy["p_optimal"],
                "noisy_p_exact_cover": noisy["p_exact_cover"],
                "noisy_best_is_optimal": noisy["best_is_optimal"],
                **solver.resources(res.p),
            }
            rows.append(row)
            print(f"  {mode:11s} p={res.p}: P(opt) {100 * row['p_optimal']:5.2f}%  "
                  f"valid {100 * row['p_exact_cover']:5.1f}%  ratio {row['approximation_ratio']:.4f}  "
                  f"best-of-{SHOTS} optimal: {row['sampled_best_is_optimal']}  "
                  f"| noisy P(opt) {100 * row['noisy_p_optimal']:.2f}% valid "
                  f"{100 * row['noisy_p_exact_cover']:.1f}%  | CX {row['two_qubit_gates']}")
        print(f"    ({mode}: {time.time() - t0:.0f} s)")
    out["qaoa"] = rows
    out["uniform_valid_p_optimal"] = 1 / n_valid
    out["uniform_all_p_optimal"] = 1 / len(ref.energies)

    main_solver = QAOASolver(inst, w, "constrained")
    main_res = results["constrained"][-1]
    main_row = [r for r in rows if r["mode"] == "constrained"][-1]
    qaoa = describe(ref, main_row["sampled_best_index"])
    out["qaoa_best_schedule"] = qaoa

    plots.plot_schedules(inst, [
        ("Greedy priority list", brief(greedy).split(" | ", 1)[1], greedy["index"]),
        ("Exact optimum (brute force)", brief(exact).split(" | ", 1)[1], exact["index"]),
        (f"QAOA best sample (p = {main_res.p})", brief(qaoa).split(" | ", 1)[1], qaoa["index"]),
    ], FIG / "schedules.png")
    plots.plot_distribution(
        main_solver.exact_probabilities(main_res), ref, FIG / "distribution.png",
        title=f"Constraint-preserving QAOA, p = {main_res.p}: most likely outcomes")
    plots.plot_depth_sweep(rows, FIG / "depth_sweep.png")
    plots.plot_convergence(results["constrained"], float(ref.energies.min()), FIG / "convergence.png")
    plots.plot_architecture(FIG / "architecture.png")
    labelled = main_solver.circuit(1).assign_parameters([0.6, 0.2])
    for gate in labelled.data:
        if gate.operation.name == "state_preparation":
            gate.operation.label = "Dicke"
    fig_c = labelled.draw("mpl", fold=60, style="clifford", scale=0.7)
    fig_c.savefig(FIG / "circuit_p1.png", dpi=120, bbox_inches="tight")

    # ---------------------------------------------------------------- 3. p = 1 landscape
    qc1 = main_solver.circuit(1)
    gammas = np.linspace(-1.6, 1.6, 41)
    betas = np.linspace(-0.8, 0.8, 41)
    grid = np.stack(np.meshgrid(gammas, betas, indexing="ij"), axis=-1)  # (G, B, 2)
    evs = main_solver.estimator.run([(qc1, main_solver.operator, grid)]).result()[0].data.evs
    plots.plot_landscape(gammas, betas, np.asarray(evs) + main_solver.const,
                         best=results["constrained"][0].params, path=FIG / "landscape.png")

    # ---------------------------------------------------------------- 4. cost vs fairness
    sweep = []
    p_sweep = min(2, depths[-1])
    for scale in (0.0, 0.1, 0.25, 1.0, 4.0, 8.0):
        w_s = replace(w, consecutive=w.consecutive * scale, concentration=w.concentration * scale)
        ref_s = exact_reference(inst, w_s)
        solver_s = QAOASolver(inst, w_s, "constrained")
        res_s = run_depths(solver_s, range(1, p_sweep + 1), n_random_starts=2)[-1]
        best = sample_metrics(solver_s, res_s, ref_s, shots=SHOTS)
        # Reference = best schedule that obeys the hard shortfall rule.
        e, q = describe(ref_s, ref_s.best_valid_index), describe(ref_s, best["best_index"])
        sweep.append({
            "fairness_scale": scale, "exact_social": e["social_cost"], "exact_worst": e["max_outages"],
            "exact_consecutive": e["consecutive_pairs"], "qaoa_social": q["social_cost"],
            "qaoa_worst": q["max_outages"], "qaoa_consecutive": q["consecutive_pairs"],
            "qaoa_matches_exact": bool(abs(q["total"] - e["total"]) < 1e-9),
            "exact_outages": e["outages_per_feeder"],
        })
        print(f"  fairness x{scale:g}: exact social {e['social_cost']:.1f}, worst feeder "
              f"{e['max_outages']} slots, outages {e['outages_per_feeder']} | "
              f"QAOA matches exact: {sweep[-1]['qaoa_matches_exact']}")
    out["tradeoff"] = sweep
    plots.plot_tradeoff(sweep, FIG / "tradeoff.png")

    # ---------------------------------------------------------------- 5. 15-qubit check
    if not args.quick:
        big = larger_instance()
        ref_b = exact_reference(big, w)
        solver_b = QAOASolver(big, w, "constrained")
        t0 = time.time()
        res_b = run_depths(solver_b, (1, 2), n_random_starts=1, maxiter=150)
        large_rows = []
        for res in res_b:
            ideal = distribution_metrics(solver_b.exact_probabilities(res), ref_b)
            shots = sample_metrics(solver_b, res, ref_b, shots=SHOTS)
            large_rows.append({"p": res.p, **ideal,
                               "sampled_best_is_optimal": shots["best_is_optimal"],
                               "sampled_distinct": shots["distinct_bitstrings"],
                               **solver_b.resources(res.p)})
        out["larger_instance"] = {
            "qubits": big.n_vars, "schedules_total": int(len(ref_b.energies)),
            "schedules_valid": int(ref_b.exact_cover.sum()), "rows": large_rows,
            "seconds": round(time.time() - t0, 1),
        }
        for r in large_rows:
            print(f"  15 qubits p={r['p']}: P(opt) {100 * r['p_optimal']:.2f}% "
                  f"(random valid {100 / ref_b.exact_cover.sum():.2f}%)  "
                  f"best-of-{SHOTS} optimal: {r['sampled_best_is_optimal']}")

    out["runtime_seconds"] = round(time.time() - t_start, 1)
    (RES / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Done in {out['runtime_seconds']} s. Figures in {FIG}, numbers in {RES / 'results.json'}")


if __name__ == "__main__":
    main()
