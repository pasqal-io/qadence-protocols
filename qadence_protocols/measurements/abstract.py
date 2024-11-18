from __future__ import annotations

from abc import ABC, abstractmethod

from qadence import QuantumModel
from qadence.blocks.abstract import AbstractBlock
from torch import Tensor

from qadence_protocols.types import MeasurementData


class MeasurementManager(ABC):
    """The abstract class that defines the interface for the managing measurements."""

    def __init__(self, measurement_data: MeasurementData = None, options: dict = dict()):
        self.measurement_data = measurement_data
        self.options = options

    @abstractmethod
    def measure(
        self,
        model: QuantumModel,
        observables: list[AbstractBlock] = list(),
        param_values: dict[str, Tensor] = dict(),
        state: Tensor | None = None,
    ) -> MeasurementData:
        raise NotImplementedError

    @abstractmethod
    def expectation(
        self,
        model: QuantumModel,
        observables: list[AbstractBlock] = list(),
        param_values: dict[str, Tensor] = dict(),
        state: Tensor | None = None,
    ) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def verify_options(self) -> dict:
        raise NotImplementedError
