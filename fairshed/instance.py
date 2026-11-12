"""Problem data for the load-shedding scheduling problem.

A distribution company (DISCO) has a forecast supply shortfall in each time
slot. It must choose which feeder blocks to switch off in which slot.

Decision variable:  x[f, t] = 1  if feeder block f is switched OFF in slot t.

Loads and deficits are expressed in integer "blocks" (1 block = `unit_mw`
megawatts). Rotational load shedding is planned over feeder groups of similar
size, so the default instance uses equal one-block feeders. The QUBO in qubo.py
also accepts unequal loads; the constraint-preserving quantum circuit does not
(see quantum.py).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Feeder:
    name: str
    load: int  # blocks removed from the grid when this feeder is off
    social_cost: float  # harm of one slot of outage (higher = more critical)
    prior_outages: int = 0  # slots this feeder was already off earlier today


@dataclass(frozen=True)
class Weights:
    """Penalty weights of the objective. See qubo.py for the exact terms."""

    deficit: float  # A: mismatch between shed load and required shortfall
    consecutive: float  # B: same feeder off in two back-to-back slots
    concentration: float  # C: outages piled onto the same feeder


@dataclass(frozen=True)
class Instance:
    feeders: tuple[Feeder, ...]
    deficits: tuple[int, ...]  # required load reduction per slot, in blocks
    slot_labels: tuple[str, ...]
    unit_mw: float = 5.0

    def __post_init__(self) -> None:
        if len(self.deficits) != len(self.slot_labels):
            raise ValueError("deficits and slot_labels must have the same length")

    @property
    def n_feeders(self) -> int:
        return len(self.feeders)

    @property
    def n_slots(self) -> int:
        return len(self.deficits)

    @property
    def n_vars(self) -> int:
        return self.n_feeders * self.n_slots

    def index(self, f: int, t: int) -> int:
        """Flat variable (qubit) index of x[f, t].

        Slot-major order: the qubits of one time slot sit next to each other,
        which keeps the per-slot state preparation and mixer gates local."""
        return t * self.n_feeders + f


def default_instance() -> Instance:
    """4 feeder blocks x 3 evening-peak slots = 12 binary variables (12 qubits).

    The shortfall is 5 MW at 18:00, 10 MW at 19:00 and 5 MW at 20:00.
    """
    return Instance(
        feeders=(
            Feeder("Hospital & water", load=1, social_cost=9.0),
            Feeder("Industrial estate", load=1, social_cost=5.0),
            Feeder("Commercial market", load=1, social_cost=3.5),
            Feeder("Residential block", load=1, social_cost=2.0),
        ),
        deficits=(1, 2, 1),
        slot_labels=("18:00", "19:00", "20:00"),
        unit_mw=5.0,
    )


def larger_instance() -> Instance:
    """5 feeder blocks x 3 slots = 15 qubits, used for the scaling check."""
    return Instance(
        feeders=(
            Feeder("Hospital & water", load=1, social_cost=9.0),
            Feeder("Industrial estate", load=1, social_cost=5.0),
            Feeder("Commercial market", load=1, social_cost=3.5),
            Feeder("Residential A", load=1, social_cost=2.0),
            Feeder("Residential B", load=1, social_cost=2.5),
        ),
        deficits=(2, 3, 2),
        slot_labels=("18:00", "19:00", "20:00"),
        unit_mw=5.0,
    )


def default_weights() -> Weights:
    """A must exceed the largest social-cost saving from under-shedding one
    block, otherwise the penalty formulation prefers to ignore the shortfall."""
    return Weights(deficit=12.0, consecutive=4.0, concentration=1.5)
