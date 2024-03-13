from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    def __str__(self) -> str:
        """Used when dumping enum fields in a schema."""
        ret: str = self.value
        return ret

    @classmethod
    def list(cls) -> list[str]:
        return list(map(lambda c: c.value, cls))  # type: ignore


class Protocols(StrEnum):
    """The available protocols for running experiments."""

    MITIGATIONS = "mitigations"
    """The Mitigations protocol."""
    # TODO: Placeholder for the measurements protocol.
    # MEASUREMENTS = "measurements"
    # """The Measurements protocol."""


qadence_available_protocols = Protocols.list()


class ReadOutOptimization(StrEnum):
    # Basic inversion and maximum likelihood estimate
    MLE = "mle"
    # Constrained inverse optimization
    CONSTRAINED = "constrained"
    # Matrix free measurement mitigation
    MTHREE = "mthree"
    # Majority voting
    MAJ_VOTE = "majority_vote"
