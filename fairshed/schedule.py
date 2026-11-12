"""Turn a solved bitstring into things an operator can read and publish."""

from __future__ import annotations

import pandas as pd

from .instance import Instance
from .qubo import index_to_bits, to_matrix


def slot_start_hour(label: str) -> int:
    return int(label.split(":")[0])


def slot_range(label: str) -> str:
    """'18:00' -> '18:00–19:00' (each slot is one hour)."""
    h = slot_start_hour(label)
    return f"{h:02d}:00–{(h + 1) % 24:02d}:00"


def schedule_matrix(inst: Instance, index: int):
    """(feeders x slots) array of 0/1, 1 = feeder off."""
    return to_matrix(inst, index_to_bits(index, inst.n_vars))


def schedule_frame(inst: Instance, index: int, protected: tuple[str, ...] = ()) -> pd.DataFrame:
    """Operator table: one row per feeder, one column per hour, ON/OFF cells."""
    X = schedule_matrix(inst, index)
    columns = [slot_range(label) for label in inst.slot_labels]
    rows = {}
    for f, feeder in enumerate(inst.feeders):
        rows[feeder.name] = ["OFF" if X[f, t] else "ON" for t in range(inst.n_slots)]
    for name in protected:
        rows[f"{name} (protected)"] = ["ON"] * inst.n_slots
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=columns)
    frame.index.name = "Feeder"
    frame["Hours off"] = [row.count("OFF") for row in rows.values()]
    return frame


def outage_intervals(inst: Instance, index: int) -> dict[str, list[str]]:
    """Per feeder, its outages as merged time ranges, e.g. ['18:00–20:00']."""
    X = schedule_matrix(inst, index)
    out: dict[str, list[str]] = {}
    for f, feeder in enumerate(inst.feeders):
        ranges, start = [], None
        for t in range(inst.n_slots + 1):
            off = t < inst.n_slots and bool(X[f, t])
            if off and start is None:
                start = t
            if not off and start is not None:
                begin = slot_start_hour(inst.slot_labels[start])
                end = (slot_start_hour(inst.slot_labels[t - 1]) + 1) % 24
                ranges.append(f"{begin:02d}:00–{end:02d}:00")
                start = None
        out[feeder.name] = ranges
    return out


def notice_lines(inst: Instance, index: int, protected: tuple[str, ...] = ()) -> list[str]:
    """Plain-language outage notice, one line per feeder."""
    lines = []
    for name, ranges in outage_intervals(inst, index).items():
        if ranges:
            lines.append(f"{name}: power OFF {', '.join(ranges)}")
        else:
            lines.append(f"{name}: no outage")
    lines += [f"{name}: no outage (protected)" for name in protected]
    return lines


def slot_summary(inst: Instance, index: int) -> str:
    """One line describing a schedule, used in chart tooltips."""
    X = schedule_matrix(inst, index)
    parts = []
    for t, label in enumerate(inst.slot_labels):
        names = [inst.feeders[f].name for f in range(inst.n_feeders) if X[f, t]]
        parts.append(f"{label}: {' + '.join(names) if names else 'nothing off'}")
    return "; ".join(parts)
