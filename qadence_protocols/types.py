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
    # basic inversion and maximum likelihood estimate
    MLE = "mle"

    # constrained inverse optimization
    CONSTRAINED = "constrained"

    # matrix free measurement mitigation
    MTHREE = "mthree"

    # majority voting
    MV = "majority_vote"
