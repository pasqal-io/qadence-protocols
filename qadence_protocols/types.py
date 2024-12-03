from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import Enum

import torch
from torch import Tensor


class StrEnum(str, Enum):
    def __str__(self) -> str:
        """Used when dumping enum fields in a schema."""
        ret: str = self.value
        return ret

    @classmethod
    def list(cls) -> list[str]:
        return list(map(lambda c: c.value, cls))  # type: ignore


@dataclass
class MeasurementData:
    samples: Tensor | list[Counter] = dataclass_field(default_factory=list)
    """Samples from protocol."""
    unitaries: Tensor = torch.empty(0)
    """Random unitaries used in shadows."""


class Protocols(StrEnum):
    """The available protocols for running experiments."""

    MITIGATIONS = "mitigations"
    """The Mitigations protocol."""
    MEASUREMENTS = "measurements"
    """The Measurements protocol."""


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


class MeasurementProtocol(StrEnum):
    TOMOGRAPHY = "tomography"
    """Tomography of a quantum state."""
    SHADOW = "shadow"
    """Snapshots of a state via shadows."""
    ROBUST_SHADOW = "robust_shadow"
    """Snapshots of a state via shadows for noisy settings."""
