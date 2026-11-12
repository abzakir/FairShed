"""FairShed: a quantum (QAOA) scheduler for fair load shedding."""

from .instance import (
    Feeder,
    Instance,
    Weights,
    default_instance,
    default_weights,
    larger_instance,
)

__all__ = [
    "Feeder",
    "Instance",
    "Weights",
    "default_instance",
    "default_weights",
    "larger_instance",
]
