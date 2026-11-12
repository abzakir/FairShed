"""Solver worker process used by the Streamlit app.

    python -m fairshed.worker     (reads one JSON job on stdin)

Why a separate process: calling Qiskit's estimator repeatedly from Streamlit's
script thread crashed the whole server with a native access violation (Windows,
Python 3.14, Qiskit 2.6). Run from the main thread of its own process, exactly
like run_experiments.py, the same code is stable. A crash here can also no
longer take the web app down with it.

Protocol (stdout, one line each):
    PROGRESS <evaluations> <best value so far>
    RESULT <json>
"""

from __future__ import annotations

import base64
import io
import json
import sys

import numpy as np

from .analysis import distribution_metrics, exact_reference, run_depths
from .instance import Feeder, Instance, Weights
from .quantum import QAOASolver, depolarizing_noise, score_samples


def instance_from_job(job: dict) -> Instance:
    return Instance(
        feeders=tuple(Feeder(f["name"], int(f["load"]), float(f["social_cost"]),
                             int(f.get("prior_outages", 0))) for f in job["feeders"]),
        deficits=tuple(int(d) for d in job["deficits"]),
        slot_labels=tuple(job["slot_labels"]),
        unit_mw=float(job.get("unit_mw", 5.0)),
    )


def solve(job: dict, progress=None) -> dict:
    inst = instance_from_job(job)
    weights = Weights(**job["weights"])
    p, shots, noisy = int(job["p"]), int(job["shots"]), bool(job["noisy"])

    ref = exact_reference(inst, weights)
    solver = QAOASolver(inst, weights, "constrained")
    results = run_depths(solver, range(1, p + 1), n_random_starts=1, maxiter=150, callback=progress)
    result = results[-1]

    probs = solver.exact_probabilities(result)
    counts = solver.sample(result, shots=shots, noise_model=depolarizing_noise() if noisy else None)
    scored = score_samples(counts, ref.energies, ref.exact_cover)
    top = np.argsort(probs)[::-1][:12]
    return {
        "ideal": distribution_metrics(probs, ref),
        "sampled": scored,
        "top_outcomes": [{"index": int(k), "probability": float(probs[k])} for k in top],
        "resources": solver.resources(p),
        "evaluations": int(sum(r.n_evaluations for r in results)),
        "history": np.minimum.accumulate(result.history).tolist(),
        "params": result.params.tolist(),
    }


def draw_circuit(job: dict) -> dict:
    """PNG of one QAOA layer for the given problem, base64-encoded."""
    import matplotlib

    matplotlib.use("Agg")
    inst = instance_from_job(job)
    solver = QAOASolver(inst, Weights(**job["weights"]), "constrained")
    qc = solver.circuit(1).assign_parameters(job.get("angles", [0.6, 0.2]))
    for gate in qc.data:
        if gate.operation.name == "state_preparation":
            gate.operation.label = "Dicke"
    fig = qc.draw("mpl", fold=60, style="clifford", scale=0.7)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    return {"png_base64": base64.b64encode(buf.getvalue()).decode("ascii")}


def main() -> None:
    job = json.loads(sys.stdin.read())

    def progress(n: int, best: float) -> None:
        if n % 5 == 0:
            print(f"PROGRESS {n} {best:.4f}", flush=True)

    if job.get("task") == "circuit":
        out = draw_circuit(job)
    else:
        out = solve(job, progress)
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
