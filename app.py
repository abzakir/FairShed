"""FairShed: interactive load-shedding planner.

    streamlit run app.py

Enter the feeders, tonight's shortfall and how much fairness matters. The app
builds the optimisation problem, solves it with QAOA on a quantum simulator
(Qiskit), verifies the answer against an exact solver and prints a schedule
the control room can publish.
"""

from __future__ import annotations

import base64
import html
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from fairshed import Feeder, Instance, default_weights
from fairshed.analysis import describe, exact_reference, greedy_index
from fairshed.schedule import (
    notice_lines,
    schedule_frame,
    schedule_matrix,
    slot_range,
    slot_summary,
)

ROOT = Path(__file__).parent
MAX_QUBITS = 15
BLUE, AQUA, YELLOW, GREY = "#2a78d6", "#1baf7a", "#eda100", "#c3c2b7"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

st.set_page_config(page_title="FairShed", page_icon="⚡", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 3.4rem; padding-bottom: 3rem; max-width: 1280px;}
section[data-testid="stSidebar"] {width: 420px !important; min-width: 420px !important;}
table.fs-grid, table.fs-grid tr, table.fs-grid th, table.fs-grid td {border: none !important;}
.fs-title {font-size: 2.3rem !important; font-weight: 800 !important; color: #0b0b0b !important;
           margin: 0 !important; line-height: 1.1 !important;}
.fs-tag {color: #52514e !important; font-size: 1.05rem !important; margin: .35rem 0 1rem !important;}
.fs-chips {display: flex; flex-wrap: wrap; gap: .5rem; margin-bottom: 1rem;}
.fs-chip {background: #eef4fc; color: #184f95; border-radius: 999px; padding: .22rem .75rem;
          font-size: .82rem; font-weight: 600;}
.fs-kpis {display: grid; grid-template-columns: repeat(5, minmax(0, 1fr));
          gap: .75rem; margin: .25rem 0 1.25rem;}
@media (max-width: 1100px) {.fs-kpis {grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));}}
.fs-kpi {background: #ffffff; border: 1px solid #e1e0d9; border-radius: 12px; padding: .85rem 1rem;}
.fs-kpi .l {font-size: .78rem; color: #52514e; margin-bottom: .15rem;}
.fs-kpi .v {font-size: 1.45rem; font-weight: 750; color: #0b0b0b; line-height: 1.15;}
.fs-kpi .s {font-size: .78rem; color: #52514e; margin-top: .2rem;}
.fs-kpi.accent {border-top: 4px solid #2a78d6;}
table.fs-grid {border-collapse: separate; border-spacing: 5px; width: 100%; margin-bottom: .35rem;}
table.fs-grid th {font-size: .8rem; font-weight: 600; color: #52514e; padding: .2rem .4rem; text-align: center;}
table.fs-grid th.fs-name, table.fs-grid td.fs-name {text-align: left; white-space: nowrap;
          font-size: .9rem; color: #0b0b0b; font-weight: 500; padding-right: .8rem;}
table.fs-grid td.fs-cell {text-align: center; border-radius: 8px; padding: .55rem .3rem;
          font-size: .82rem; min-width: 72px;}
td.fs-off {background: #184f95; color: #ffffff; font-weight: 700; letter-spacing: .03em;}
td.fs-on {background: #f0efec; color: #898781;}
td.fs-lock {background: #ffffff; color: #898781; border: 1px dashed #c3c2b7;}
table.fs-grid td.fs-total {text-align: center; font-size: .85rem; color: #0b0b0b; font-weight: 600;}
table.fs-grid tr.fs-foot td {font-size: .78rem; color: #52514e; text-align: center; padding-top: .3rem;}
table.fs-grid.compact {border-spacing: 4px; table-layout: fixed;}
table.fs-grid.compact td.fs-cell {min-width: 0; padding: .5rem .1rem; font-size: .74rem;}
table.fs-grid.compact th {font-size: .74rem; padding: .15rem .1rem;}
table.fs-grid.compact th.fs-name, table.fs-grid.compact td.fs-name {white-space: normal;
          font-size: .8rem; width: 34%; padding-right: .3rem; line-height: 1.2;}
table.fs-grid.compact tr.fs-foot td {font-size: .72rem;}
.fs-cap {font-size: .84rem !important; color: #52514e !important; margin: -.1rem 0 .8rem !important;}
.fs-h {font-size: 1.08rem !important; font-weight: 700 !important; color: #0b0b0b !important;
       margin: .2rem 0 .1rem !important;}
.fs-note {background: #f6f6f3; border-radius: 10px; padding: .8rem 1rem; font-size: .9rem; color: #0b0b0b;}
.fs-step {display: flex; gap: .8rem; margin-bottom: .7rem;}
.fs-step .n {flex: 0 0 1.8rem; height: 1.8rem; border-radius: 50%; background: #2a78d6; color: #fff;
          font-weight: 700; display: flex; align-items: center; justify-content: center; font-size: .9rem;}
.fs-step .t {font-size: .93rem; color: #0b0b0b;}
.fs-step .t b {display: block;}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Scenario presets
# --------------------------------------------------------------------------
BASE_FEEDERS = [
    ("Hospital & water", 9.0, 0, False),
    ("Industrial estate", 5.0, 0, False),
    ("Commercial market", 3.5, 0, False),
    ("Residential block", 2.0, 0, False),
]
PRESETS = {
    "Evening peak (default)": {"feeders": BASE_FEEDERS, "shortfall": [1, 2, 1], "start": 18},
    "Severe shortfall": {"feeders": BASE_FEEDERS, "shortfall": [2, 3, 2], "start": 18},
    "Residential already off 2 h today": {
        "feeders": BASE_FEEDERS[:3] + [("Residential block", 2.0, 2, False)],
        "shortfall": [1, 2, 1], "start": 18,
    },
    "Hospital protected (9 qubits)": {
        "feeders": [("Hospital & water", 9.0, 0, True)] + BASE_FEEDERS[1:],
        "shortfall": [1, 2, 1], "start": 18,
    },
    "Five feeders (15 qubits, slower)": {
        "feeders": BASE_FEEDERS[:3] + [("Residential A", 2.0, 0, False), ("Residential B", 2.5, 0, False)],
        "shortfall": [2, 3, 2], "start": 18,
    },
}

# --------------------------------------------------------------------------
# Sidebar: inputs
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Scenario")
    preset_name = st.selectbox("Start from", list(PRESETS), label_visibility="collapsed")
    preset = PRESETS[preset_name]

    st.markdown("### 1 · Feeders")
    feeder_df = st.data_editor(
        pd.DataFrame(preset["feeders"], columns=["Feeder", "Social cost", "Off today (h)", "Protect"]),
        key=f"feeders::{preset_name}",
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            "Feeder": st.column_config.TextColumn(required=True, width=150),
            "Social cost": st.column_config.NumberColumn(
                "Cost", width=60, min_value=0.5, max_value=10.0, step=0.5, required=True,
                help="Harm of one hour of outage on this feeder. 10 = most critical."),
            "Off today (h)": st.column_config.NumberColumn(
                "Off today", width=75, min_value=0, max_value=12, step=1, default=0, format="%d h",
                help="Hours this feeder was already off earlier today. Fairness takes it into account."),
            "Protect": st.column_config.CheckboxColumn(
                width=65, default=False, help="Never switch this feeder off."),
        },
    )
    st.caption("Cost = harm of one hour without power, 1 (low) to 10 (critical). "
               "Add or delete rows to change the feeders.")
    unit_mw = st.number_input("Size of one feeder block (MW)", 1.0, 50.0, 5.0, 1.0)

    feeder_df = feeder_df.dropna(subset=["Feeder", "Social cost"])
    feeder_df["Feeder"] = feeder_df["Feeder"].astype(str).str.strip()
    feeder_df = feeder_df[feeder_df["Feeder"] != ""]
    feeder_df["Off today (h)"] = feeder_df["Off today (h)"].fillna(0).astype(int)
    feeder_df["Protect"] = feeder_df["Protect"].fillna(False).astype(bool)
    active_df = feeder_df[~feeder_df["Protect"]]
    protected = tuple(feeder_df[feeder_df["Protect"]]["Feeder"])
    n_active = len(active_df)

    st.markdown("### 2 · Shortfall")
    c_a, c_b = st.columns(2)
    start_hour = c_a.selectbox("First hour", list(range(24)), index=preset["start"],
                               format_func=lambda h: f"{h:02d}:00")
    n_hours = c_b.selectbox("Hours", [2, 3, 4], index=1)
    slot_labels = tuple(f"{(start_hour + k) % 24:02d}:00" for k in range(n_hours))
    deficits = []
    for k, label in enumerate(slot_labels):
        default = preset["shortfall"][k] if k < len(preset["shortfall"]) else 1
        blocks = st.slider(
            f"{slot_range(label)}", 0, max(n_active, 1), min(default, max(n_active, 1)),
            key=f"short::{preset_name}::{k}", format="%d blocks",
        )
        deficits.append(blocks)
    st.caption(f"Total to shed: {sum(deficits) * unit_mw:g} MW·h over {n_hours} hours.")

    st.markdown("### 3 · Policy")
    fairness = st.select_slider(
        "How much fairness matters",
        options=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0], value=1.0,
        format_func=lambda v: "off" if v == 0 else f"{v:g}×",
        help="Off = only minimise social cost (the priority-list behaviour). "
             "Higher = spread outages across feeders and avoid back-to-back hours.",
    )

    with st.expander("Quantum settings"):
        p_layers = st.slider("QAOA layers (p)", 1, 3, 2,
                             help="More layers = better answers, longer circuit, more run time.")
        shots = st.select_slider("Measurements (shots)", options=[256, 512, 1024, 2048], value=1024)
        noisy = st.toggle("Simulate hardware noise", value=False,
                          help="Depolarizing error: 0.1% per one-qubit gate, 1% per CX gate.")

    run = st.button("⚡ Plan the schedule", type="primary", width="stretch")

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown('<p class="fs-title">⚡ FairShed</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="fs-tag">Quantum-optimised load shedding: cover the shortfall, protect critical '
    "feeders, share the burden fairly.</p>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Validate inputs
# --------------------------------------------------------------------------
problems = []
if n_active < 2:
    problems.append("Add at least two feeders that are not protected.")
if feeder_df["Feeder"].duplicated().any():
    problems.append("Feeder names must be unique.")
n_qubits = n_active * n_hours
if n_qubits > MAX_QUBITS:
    problems.append(
        f"This plan needs {n_qubits} qubits ({n_active} feeders × {n_hours} hours). The simulator "
        f"in this demo is limited to {MAX_QUBITS}. Protect a feeder, remove one, or plan fewer hours."
    )
if sum(deficits) == 0:
    problems.append("There is no shortfall to cover. Raise the shortfall for at least one hour.")
if problems:
    for msg in problems:
        st.error(msg)
    st.stop()

inst = Instance(
    feeders=tuple(
        Feeder(r["Feeder"], 1, float(r["Social cost"]), int(r["Off today (h)"]))
        for _, r in active_df.iterrows()
    ),
    deficits=tuple(deficits),
    slot_labels=slot_labels,
    unit_mw=float(unit_mw),
)
base_w = default_weights()
weights = replace(base_w, consecutive=base_w.consecutive * fairness,
                  concentration=base_w.concentration * fairness)
spec = (inst, weights, p_layers, shots, noisy)


@st.cache_data(show_spinner=False)
def exact_solution(inst: Instance, weights):
    """Exact reference: enumerate every schedule (the ground truth)."""
    ref = exact_reference(inst, weights)
    return ref, describe(ref, ref.best_valid_index), describe(ref, greedy_index(inst))


def run_worker(job: dict, on_progress=None) -> dict:
    """Run one job in the solver worker process (see fairshed/worker.py).

    The quantum computation lives in its own process on purpose: Qiskit's
    estimator crashed the Streamlit server when called from its script thread.
    """
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(
        [sys.executable, "-m", "fairshed.worker"], cwd=ROOT, env=env, creationflags=flags,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8",
    )
    proc.stdin.write(json.dumps(job))
    proc.stdin.close()
    result = None
    for line in proc.stdout:
        if line.startswith("PROGRESS ") and on_progress is not None:
            _, n, best = line.split()
            on_progress(int(n), float(best))
        elif line.startswith("RESULT "):
            result = json.loads(line[len("RESULT "):])
    errors = proc.stderr.read()
    if proc.wait() != 0 or result is None:
        raise RuntimeError(errors.strip().splitlines()[-1] if errors.strip() else "solver stopped")
    return result


def job_for(inst: Instance, weights, p: int, shots: int, noisy: bool) -> dict:
    return {
        "feeders": [asdict(fd) for fd in inst.feeders],
        "deficits": list(inst.deficits),
        "slot_labels": list(inst.slot_labels),
        "unit_mw": inst.unit_mw,
        "weights": asdict(weights),
        "p": p, "shots": shots, "noisy": noisy,
    }


def quantum_solution(inst: Instance, weights, p: int, shots: int, noisy: bool, progress=None):
    """Solve in the worker process, remembering answers for this browser session.

    A plain dictionary is used instead of st.cache_data because the progress
    bar is updated while the worker runs, and Streamlit cannot replay element
    updates made inside a cached function.
    """
    job = job_for(inst, weights, p, shots, noisy)
    key = json.dumps(job, sort_keys=True)
    solved = st.session_state.setdefault("solved", {})
    if key not in solved:
        budget = 150 * (2 + 3 * (p - 1))  # rough number of circuit evaluations

        def tick(n, best):
            if progress is not None:
                progress.progress(
                    min(n / budget, 0.97),
                    text=f"Tuning the quantum circuit · evaluation {n} · best ⟨H⟩ {best:.2f}")

        solved[key] = run_worker(job, tick)
    return solved[key]


@st.cache_data(show_spinner=False)
def circuit_png(inst: Instance, weights, angles: tuple) -> bytes:
    job = job_for(inst, weights, 1, 1, False) | {"task": "circuit", "angles": list(angles)}
    return base64.b64decode(run_worker(job)["png_base64"])


ref, exact, greedy = exact_solution(inst, weights)
n_valid = int(ref.exact_cover.sum())

chips = [
    f"{inst.n_feeders} feeders × {inst.n_slots} hours",
    f"{inst.n_vars} qubits",
    f"{len(ref.energies):,} possible schedules",
    f"{n_valid:,} cover the shortfall exactly",
]
st.markdown('<div class="fs-chips">' + "".join(f'<span class="fs-chip">{c}</span>' for c in chips)
            + "</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------
def grid_html(index: int, compact: bool = False) -> str:
    """Schedule as an HTML grid: rows = feeders, columns = hours.

    compact=True is the narrow variant used when three schedules sit side by side."""
    X = schedule_matrix(inst, index)
    head = "".join(f"<th>{lbl if compact else slot_range(lbl)}</th>" for lbl in inst.slot_labels)
    rows = []
    for f, feeder in enumerate(inst.feeders):
        cells = "".join(
            f'<td class="fs-cell {"fs-off" if X[f, t] else "fs-on"}">{"OFF" if X[f, t] else "on"}</td>'
            for t in range(inst.n_slots)
        )
        total = "" if compact else f'<td class="fs-total">{int(X[f].sum())} h</td>'
        rows.append(f'<tr><td class="fs-name">{html.escape(feeder.name)}</td>{cells}{total}</tr>')
    for name in protected:
        cells = f'<td class="fs-cell fs-lock">{"🔒" if compact else "protected"}</td>' * inst.n_slots
        total = "" if compact else '<td class="fs-total">0 h</td>'
        rows.append(f'<tr><td class="fs-name">{html.escape(name)}</td>{cells}{total}</tr>')
    shed = X.sum(axis=0) * inst.unit_mw
    if compact:
        foot = "".join(f"<td>{shed[t]:g}/{inst.deficits[t] * inst.unit_mw:g} MW</td>"
                       for t in range(inst.n_slots))
        return (f'<table class="fs-grid compact"><tr><th class="fs-name"></th>{head}</tr>'
                f'{"".join(rows)}<tr class="fs-foot"><td class="fs-name">Shed</td>{foot}</tr></table>')
    foot = "".join(f"<td>{shed[t]:g} of {inst.deficits[t] * inst.unit_mw:g} MW</td>"
                   for t in range(inst.n_slots))
    return (f'<table class="fs-grid"><tr><th class="fs-name">Feeder</th>{head}<th>Off</th></tr>'
            f'{"".join(rows)}<tr class="fs-foot"><td class="fs-name">Load shed</td>{foot}<td></td></tr></table>')


def kpi(label: str, value: str, sub: str = "", accent: bool = False) -> str:
    return (f'<div class="fs-kpi{" accent" if accent else ""}"><div class="l">{label}</div>'
            f'<div class="v">{value}</div><div class="s">{sub}</div></div>')


def caption_for(d: dict) -> str:
    return (f"Social cost {d['social_cost']:.1f} · worst-hit feeder off {d['max_outages']} h · "
            f"{d['consecutive_pairs']} back-to-back outage{'s' if d['consecutive_pairs'] != 1 else ''}")


# --------------------------------------------------------------------------
# Before the first run: show today's practice and what the button will do
# --------------------------------------------------------------------------
if run:
    st.session_state["planned"] = spec
if st.session_state.get("planned") != spec:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown('<p class="fs-h">What a fixed priority list would do</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="fs-cap">{caption_for(greedy)}</p>', unsafe_allow_html=True)
        st.markdown(grid_html(greedy["index"]), unsafe_allow_html=True)
    with right:
        st.markdown('<p class="fs-h">Ready to plan</p>', unsafe_allow_html=True)
        st.markdown(
            '<div class="fs-note">Set the feeders, the shortfall and the fairness policy in the '
            "sidebar, then press <b>Plan the schedule</b>. FairShed will tune a "
            f"{inst.n_vars}-qubit QAOA circuit, measure it {shots} times and check the result "
            "against the exact answer.</div>",
            unsafe_allow_html=True,
        )
    st.stop()

# --------------------------------------------------------------------------
# Solve
# --------------------------------------------------------------------------
bar = st.progress(0.0, text="Starting the quantum solver…")
try:
    out = quantum_solution(inst, weights, p_layers, shots, noisy, progress=bar)
except RuntimeError as err:
    bar.empty()
    st.error(f"The quantum solver stopped unexpectedly: {err}. Press Plan the schedule to try again.")
    st.stop()
bar.empty()

qaoa = describe(ref, out["sampled"]["best_index"])
matches = qaoa["valid"] and abs(qaoa["total"] - exact["total"]) < 1e-9

tab_plan, tab_compare, tab_quantum, tab_how = st.tabs(
    ["📋 Schedule", "⚖️ Compare", "⚛️ Quantum details", "ℹ️ How it works"]
)

# ----------------------------------------------------------------- Schedule tab
with tab_plan:
    if not qaoa["valid"]:
        st.error("No measured schedule covered the shortfall exactly (noise). "
                 "Increase the shots or turn noise off and plan again.")
    saved = greedy["max_outages"] - qaoa["max_outages"]
    cards = [
        kpi("Verified against exact solver",
            "Optimal ✓" if matches else f"Rank {qaoa['rank_valid']} of {n_valid}",
            "identical to the best possible schedule" if matches else "close to the best schedule",
            accent=True),
        kpi("Shortfall covered", "Every hour" if qaoa["valid"] else "Not fully",
            f"{sum(inst.deficits) * inst.unit_mw:g} MW·h shed, none wasted"),
        kpi("Worst-hit feeder", f"{qaoa['max_outages']} h off",
            f"priority list: {greedy['max_outages']} h"
            + (f" ({saved} h less)" if saved > 0 else "")),
        kpi("Back-to-back outages", f"{qaoa['consecutive_pairs']}",
            f"priority list: {greedy['consecutive_pairs']}"),
        kpi("Social cost", f"{qaoa['social_cost']:.1f}",
            f"priority list: {greedy['social_cost']:.1f}"),
    ]
    st.markdown('<div class="fs-kpis">' + "".join(cards) + "</div>", unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown('<p class="fs-h">Schedule to publish</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="fs-cap">Best of {shots} measurements of the quantum circuit '
                    f"(p = {p_layers}).</p>", unsafe_allow_html=True)
        st.markdown(grid_html(qaoa["index"]), unsafe_allow_html=True)
    with right:
        st.markdown('<p class="fs-h">Outage notice</p>', unsafe_allow_html=True)
        lines = notice_lines(inst, qaoa["index"], protected)
        st.markdown('<div class="fs-note">' + "<br>".join(html.escape(l) for l in lines) + "</div>",
                    unsafe_allow_html=True)
        st.write("")
        frame = schedule_frame(inst, qaoa["index"], protected)
        d1, d2 = st.columns(2)
        d1.download_button("⬇ Schedule (CSV)", frame.to_csv().encode("utf-8"),
                           "fairshed_schedule.csv", "text/csv", width="stretch")
        d2.download_button("⬇ Notice (TXT)", "\n".join(lines).encode("utf-8"),
                           "fairshed_notice.txt", "text/plain", width="stretch")

    if fairness == 0:
        st.info("Fairness is off, so the best schedule is simply the cheapest one, the same as the priority list.")
    elif qaoa["social_cost"] > greedy["social_cost"] + 1e-9:
        st.caption(
            f"Trade-off: this schedule has a higher social cost than the priority list "
            f"({qaoa['social_cost']:.1f} against {greedy['social_cost']:.1f}) because it moves some "
            "outages onto feeders that matter more, in exchange for sharing the burden."
        )

# ----------------------------------------------------------------- Compare tab
with tab_compare:
    cols = st.columns(3, gap="medium")
    panels = [
        ("Priority list (greedy)", greedy, "Always cuts the cheapest feeders first."),
        ("Exact optimum", exact, f"Best of all {n_valid:,} valid schedules, by brute force."),
        (f"FairShed (QAOA, p = {p_layers})", qaoa, f"Best of {shots} quantum measurements."),
    ]
    for col, (title, d, note) in zip(cols, panels):
        with col:
            st.markdown(f'<p class="fs-h">{title}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="fs-cap">{note}</p>', unsafe_allow_html=True)
            st.markdown(grid_html(d["index"], compact=True), unsafe_allow_html=True)

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.markdown('<p class="fs-h">The numbers</p>', unsafe_allow_html=True)
        table = pd.DataFrame(
            {
                "Priority list": [greedy["total"], greedy["social_cost"], greedy["max_outages"],
                                  greedy["consecutive_pairs"], greedy["rank_valid"]],
                "Exact": [exact["total"], exact["social_cost"], exact["max_outages"],
                          exact["consecutive_pairs"], exact["rank_valid"]],
                "FairShed": [qaoa["total"], qaoa["social_cost"], qaoa["max_outages"],
                             qaoa["consecutive_pairs"], qaoa["rank_valid"]],
            },
            index=["Objective (lower is better)", "Social cost", "Worst-hit feeder: hours off",
                   "Back-to-back outages", f"Rank among {n_valid:,} valid schedules"],
        )
        st.dataframe(table.astype(float).round(1), width="stretch")
    with right:
        st.markdown('<p class="fs-h">Hours without power, per feeder</p>', unsafe_allow_html=True)
        long = pd.DataFrame(
            [
                {"Feeder": fd.name, "Method": method, "Hours off": d["outages_per_feeder"][f]}
                for method, d in (("Priority list", greedy), ("Exact", exact), ("FairShed", qaoa))
                for f, fd in enumerate(inst.feeders)
            ]
        )
        chart = (
            alt.Chart(long)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Feeder:N", sort=[fd.name for fd in inst.feeders], title=None,
                        axis=alt.Axis(labelAngle=0, labelLimit=140)),
                xOffset=alt.XOffset("Method:N", sort=["Priority list", "Exact", "FairShed"]),
                y=alt.Y("Hours off:Q", title="hours off", axis=alt.Axis(tickMinStep=1)),
                color=alt.Color("Method:N", sort=["Priority list", "Exact", "FairShed"],
                                scale=alt.Scale(domain=["Priority list", "Exact", "FairShed"],
                                                range=[YELLOW, AQUA, BLUE]),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=["Feeder", "Method", "Hours off"],
            )
            .properties(height=260)
            .configure_axis(gridColor=GRID, labelColor=INK2, titleColor=INK2, domainColor=GREY)
            .configure_view(stroke=None)
        )
        st.altair_chart(chart, width="stretch")

# ----------------------------------------------------------------- Quantum tab
with tab_quantum:
    res = out["resources"]
    cards = [
        kpi("Optimal measurements", f"{100 * out['sampled']['p_optimal']:.1f}%",
            f"random valid guess: {100 / n_valid:.1f}%", accent=True),
        kpi("Valid measurements", f"{100 * out['sampled']['p_feasible']:.0f}%",
            "cover the shortfall exactly"),
        kpi("Qubits", f"{res['qubits']}", "one per feeder per hour"),
        kpi("Two-qubit gates", f"{res['two_qubit_gates']}", f"circuit depth {res['depth']}"),
        kpi("Circuit evaluations", f"{out['evaluations']}", "used to tune the angles"),
    ]
    st.markdown('<div class="fs-kpis">' + "".join(cards) + "</div>", unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown('<p class="fs-h">What the circuit measures</p>', unsafe_allow_html=True)
        st.markdown('<p class="fs-cap">The 12 most likely schedules (noise-free distribution). '
                    "Hover a bar to see the schedule.</p>", unsafe_allow_html=True)
        order = np.array([o["index"] for o in out["top_outcomes"]])
        top_probs = np.array([o["probability"] for o in out["top_outcomes"]])
        opt_mask = ref.optimal_mask
        dist = pd.DataFrame(
            {
                "Rank": [f"#{k + 1}" for k in range(len(order))],
                "Probability (%)": np.round(top_probs * 100, 2),
                "Kind": ["Exact optimum" if opt_mask[k] else "Other valid schedule" for k in order],
                "Objective": np.round(ref.energies[order], 1),
                "Schedule": [slot_summary(inst, int(k)) for k in order],
            }
        )
        bars = (
            alt.Chart(dist)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Rank:N", sort=None, title="most likely outcomes", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Probability (%):Q", title="probability (%)"),
                color=alt.Color("Kind:N",
                                scale=alt.Scale(domain=["Exact optimum", "Other valid schedule"],
                                                range=[AQUA, BLUE]),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=["Schedule", "Objective", "Probability (%)", "Kind"],
            )
        )
        rule = (
            alt.Chart(pd.DataFrame({"y": [100 / n_valid], "label": ["random valid guess"]}))
            .mark_rule(color=INK2, strokeDash=[4, 3])
            .encode(y="y:Q", tooltip=["label", alt.Tooltip("y:Q", title="probability (%)", format=".2f")])
        )
        st.altair_chart(
            (bars + rule).properties(height=280)
            .configure_axis(gridColor=GRID, labelColor=INK2, titleColor=INK2, domainColor=GREY)
            .configure_view(stroke=None),
            width="stretch",
        )
    with right:
        st.markdown('<p class="fs-h">Tuning the circuit</p>', unsafe_allow_html=True)
        st.markdown('<p class="fs-cap">Best expected objective found so far by the classical '
                    "optimiser. Dashed line: the exact optimum.</p>", unsafe_allow_html=True)
        hist = pd.DataFrame({"Evaluation": range(1, len(out["history"]) + 1), "Best ⟨H⟩": out["history"]})
        line = alt.Chart(hist).mark_line(color=BLUE, strokeWidth=2).encode(
            x=alt.X("Evaluation:Q", title="circuit evaluations"),
            y=alt.Y("Best ⟨H⟩:Q", title="expected objective", scale=alt.Scale(zero=False)),
            tooltip=["Evaluation", alt.Tooltip("Best ⟨H⟩:Q", format=".2f")],
        )
        target = alt.Chart(pd.DataFrame({"y": [exact["total"]]})).mark_rule(
            color=AQUA, strokeDash=[4, 3]).encode(y="y:Q")
        st.altair_chart(
            (line + target).properties(height=280)
            .configure_axis(gridColor=GRID, labelColor=INK2, titleColor=INK2, domainColor=GREY)
            .configure_view(stroke=None),
            width="stretch",
        )

    with st.expander("Show the quantum circuit (one layer)"):
        if st.toggle("Draw the circuit", value=False, key="draw_circuit"):
            angles = (out["params"][0], out["params"][p_layers])
            try:
                st.image(circuit_png(inst, weights, angles), width="stretch")
            except RuntimeError as err:
                st.warning(f"Could not draw the circuit: {err}")
            st.caption("Each group of qubits is one hour. Dicke: start in every valid choice at once. "
                       "RZ and ZZ: encode cost and fairness. XX+YY: move an outage between two feeders "
                       "without changing how many are off.")
    if noisy:
        st.caption("Noise is on: some measurements break the shortfall rule and are discarded "
                   "before the best schedule is chosen.")

# ----------------------------------------------------------------- How it works tab
with tab_how:
    left, right = st.columns([2, 3], gap="large")
    with left:
        steps = [
            ("Describe the grid", "Feeders, how harmful an outage is on each, and the shortfall per hour."),
            ("Build the optimisation problem", "One yes/no variable per feeder per hour: off or on. "
             "The score adds social cost and fairness penalties."),
            ("Map it to qubits", "Each variable becomes one qubit. The score becomes an energy "
             "(an Ising Hamiltonian) whose lowest state is the best schedule."),
            ("Run QAOA", "The circuit starts in every valid schedule at once, then layers of gates "
             "make good schedules more likely. A classical optimiser tunes the gate angles."),
            ("Measure and verify", "Each measurement is a schedule. The best one is checked against "
             "an exact solver and shown as the plan."),
        ]
        for n, (title, text) in enumerate(steps, 1):
            st.markdown(f'<div class="fs-step"><div class="n">{n}</div><div class="t"><b>{title}</b>'
                        f"{text}</div></div>", unsafe_allow_html=True)
        st.markdown(
            '<div class="fs-note"><b>Honest note.</b> At this size a laptop can check every schedule '
            "instantly, which is how the answer is verified. FairShed shows a correct quantum "
            "formulation of a real problem, not a quantum speedup. Everything runs on a simulator.</div>",
            unsafe_allow_html=True,
        )
    with right:
        arch = ROOT / "figures" / "architecture.png"
        if arch.exists():
            st.image(str(arch), width="stretch")
