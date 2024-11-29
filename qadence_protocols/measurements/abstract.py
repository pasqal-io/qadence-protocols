from __future__ import annotations

from abc import ABC, abstractmethod

from qadence import QuantumModel
from qadence.blocks.abstract import AbstractBlock
from torch import Tensor

from qadence_protocols.types import MeasurementData


class MeasurementManager(ABC):
    """The abstract class that defines the interface for managing measurements.

    Attributes:
        options (dict, optional): Dictionary of options specific to protocol.
        model (QuantumModel): Quantum model instance.
        observables (list[AbstractBlock], optional): List of observables. Defaults to list().
        param_values (dict[str, Tensor], optional): Parameter values. Defaults to dict().
        state (Tensor | None, optional): Input state. Defaults to None.
        data (MeasurementData, optional): Measurement data if already obtained.
    """

    def __init__(
        self,
        options: dict,
        model: QuantumModel,
        observables: list[AbstractBlock] = list(),
        param_values: dict[str, Tensor] = dict(),
        state: Tensor | None = None,
        data: MeasurementData = MeasurementData(),
    ):
        self.options = options
        self.model = model

        self.observables = observables
        self.param_values = param_values
        self.state = state
        self.data = data

    @abstractmethod
    def validate_data(self, data: MeasurementData) -> MeasurementData:
        """Validate input data for a protocol.

        Args:
            data (MeasurementData): Input data

        Returns:
            MeasurementData: Validated data
        """
        raise NotImplementedError

    @abstractmethod
    def validate_options(self, options: dict) -> dict:
        """Return a dict of validated options.

        To be used in init.

        Args:
            options (dict): Input options.

        Returns:
            dict: Validated options.
        """
        raise NotImplementedError

    @abstractmethod
    def measure(
        self,
    ) -> MeasurementData:
        """Obtain measurement data from a quantum program for measurement protocol.

        Returns:
            MeasurementData: Measurement data.
        """
        raise NotImplementedError

    @abstractmethod
    def expectation(
        self,
        observables: list[AbstractBlock],
    ) -> Tensor:
        """Compute expectation values from protocol.

        Args:
            observables (list[AbstractBlock], optional): List of observables. Defaults to list().
                Can be different from

        Returns:
            Tensor: Expectation values.
        """
        raise NotImplementedError

    @abstractmethod
    def reconstruct_state(
        self,
    ) -> Tensor:
        """Reconstruct the state from the snapshots.

        Args:
            model (QuantumModel): Quantum model instance.
            param_values (dict[str, Tensor], optional): Parameter values. Defaults to dict().
            state (Tensor | None, optional): Input state. Defaults to None.

        Returns:
            Tensor: Reconstructed state.
        """
        raise NotImplementedError
