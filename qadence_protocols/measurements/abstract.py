from __future__ import annotations

from abc import ABC, abstractmethod

from qadence import QuantumModel
from qadence.blocks.abstract import AbstractBlock
from torch import Tensor

from qadence_protocols.types import MeasurementData


class MeasurementManager(ABC):
    """The abstract class that defines the interface for managing measurements.

    Attributes:
        data (MeasurementData, optional): Measurement data if already obtained.
        options (dict, optional): Dictionary of options specific to protocol.
    """

    def __init__(self, data: MeasurementData | None = None, options: dict = dict()):
        self.data = data
        self.options = options

    @abstractmethod
    def measure(
        self,
        model: QuantumModel,
        observables: list[AbstractBlock] = list(),
        param_values: dict[str, Tensor] = dict(),
        state: Tensor | None = None,
    ) -> MeasurementData:
        """Obtain measurement data from a quantum program for measurement protocol.

        Args:
            model (QuantumModel): Quantum model instance.
            observables (list[AbstractBlock], optional): List of observables. Defaults to list().
            param_values (dict[str, Tensor], optional): Parameter values. Defaults to dict().
            state (Tensor | None, optional): Input state. Defaults to None.

        Returns:
            MeasurementData: Measurement data.
        """
        raise NotImplementedError

    @abstractmethod
    def expectation(
        self,
        model: QuantumModel,
        observables: list[AbstractBlock] = list(),
        param_values: dict[str, Tensor] = dict(),
        state: Tensor | None = None,
    ) -> Tensor:
        """Compute expectation values from protocol.

        Args:
            model (QuantumModel): Quantum model instance.
            observables (list[AbstractBlock], optional): List of observables. Defaults to list().
            param_values (dict[str, Tensor], optional): Parameter values. Defaults to dict().
            state (Tensor | None, optional): Input state. Defaults to None.

        Returns:
            Tensor: Expectation values.
        """
        raise NotImplementedError

    @abstractmethod
    def validate_options(self) -> dict:
        """Validate options passed to procotol and return a dict with default validated values."""
        raise NotImplementedError


class ShadowManagerAbstract(MeasurementManager, ABC):
    """The abstract class that defines the interface for managing shadows.

    Attributes:
        data (MeasurementData, optional): Measurement data if already obtained.
        options (dict, optional): Dictionary of options specific to protocol.
    """

    def __init__(self, data: MeasurementData | None = None, options: dict = dict()):
        super().__init__(data, options)

    @abstractmethod
    def reconstruct_state(
        self,
        model: QuantumModel,
        param_values: dict[str, Tensor] = dict(),
        state: Tensor | None = None,
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

    @abstractmethod
    def get_snapshots(
        self,
        model: QuantumModel,
        param_values: dict[str, Tensor] = dict(),
        state: Tensor | None = None,
    ) -> Tensor:
        """Obtain snapshots from the measurement data (only for shadows).

        Args:
            model (QuantumModel): Quantum model instance.
            param_values (dict[str, Tensor], optional): Parameter values. Defaults to dict().
            state (Tensor | None, optional): Input state. Defaults to None.

        Returns:
            Tensor: Snapshots for a input circuit model and state.
                The shape is (batch_size, shadow_size, 2**n, 2**n).
        """
        raise NotImplementedError
