"""Observation-vector layout — the single source of truth for an env's obs.

Each env declares its observation as an ``ObsSpec``: an ordered list of named
terms with their widths, in the exact order ``_observe`` concatenates them.
Everything that needs to know the obs shape derives it from the spec rather
than re-stating it: rejax reads ``.dim``, and the export bakes ``.layout`` (a
flat string) into policy.npz so the flight-side loop can sanity-check it. The
string only exists at that serialization boundary; in code it stays structured.
"""

from collections.abc import Iterable
from typing import NamedTuple


class ObsTerm(NamedTuple):
    """One contiguous slice of the observation vector."""

    name: str
    dim: int


_TermLike = ObsTerm | tuple[str, int]


class ObsSpec(tuple[ObsTerm, ...]):
    """An ordered, immutable sequence of ``ObsTerm`` describing a full obs.

    Iterable/indexable like a tuple; construct from ``(name, dim)`` pairs:

        ObsSpec([("pos_err", 3), ("self_vel", 3)])
    """

    def __new__(cls, terms: Iterable[_TermLike]) -> "ObsSpec":
        return super().__new__(cls, (ObsTerm(*t) for t in terms))

    @property
    def dim(self) -> int:
        """Total width of the observation vector."""
        return sum(t.dim for t in self)

    @property
    def layout(self) -> str:
        """Flat layout string for the export, e.g. ``pos_err(3) self_vel(3)``."""
        return " ".join(f"{t.name}({t.dim})" for t in self)
