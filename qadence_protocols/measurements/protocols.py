from __future__ import annotations

import importlib

from qadence import QuantumModel
from qadence.blocks.abstract import AbstractBlock
from torch import Tensor

from qadence_protocols.measurements.abstract import MeasurementManager
from qadence_protocols.protocols import Protocol
from qadence_protocols.types import MeasurementData

PROTOCOL_TO_MODULE = {
    "tomography": ("qadence_protocols.measurements.tomography", "Tomography"),
    "shadow": ("qadence_protocols.measurements.shadow", "ShadowManager"),
    "robust_shadow": ("qadence_protocols.measurements.robust_shadow", "RobustShadowManager"),
}


class Measurements(Protocol):
    """Define a measurement protocol.

    Possible options are available via the `MeasurementProtocols` type.

    Attributes:
        protocol (str): Protocol name.
        options (dict, optional): Options to run protocol.
    """

    def __init__(
        self,
        protocol: str,
        model: QuantumModel,
        options: dict = dict(),
        observables: list[AbstractBlock] = list(),
        param_values: dict[str, Tensor] = dict(),
        state: Tensor | None = None,
        data: MeasurementData = MeasurementData(),
    ) -> None:
        try:
            module = importlib.import_module(PROTOCOL_TO_MODULE[protocol][0])
            proto_class = getattr(module, PROTOCOL_TO_MODULE[protocol][1])
        except (KeyError, ModuleNotFoundError, ImportError) as e:
            raise type(e)(f"Failed to import Measurements due to {e}.")

        super().__init__(protocol, options)

        # Note that options and data are validated inside the manager
        self._manager: MeasurementManager = proto_class(
            options, model, observables, param_values, state, data
        )

    @property
    def data(self) -> MeasurementData:
        return self._manager.data

    @data.setter
    def data(self, newdata: MeasurementData) -> None:
        self._manager.data = self._manager.validate_data(newdata)

    def expectation(
        self,
        observables: list[AbstractBlock] = list(),
    ) -> Tensor:
        """Compute expectation values from the model by sampling.

        Args:
            observables (list[AbstractBlock], optional): A list of observables
                to estimate the expectation values from. Defaults to list().

        Returns:
            Tensor: Expectation values
        """
        return self._manager.expectation(observables)

    def reconstruct_state(
        self,
    ) -> Tensor:
        """Reconstruct the state.

        Used only in shadow protocols.

         Returns:
             Tensor: Reconstructed state.
        """
        return self._manager.reconstruct_state()

    def measure(
        self,
    ) -> MeasurementData:
        """Obtain measurements by sampling using the protocol.

        Returns:
            MeasurementData: Measurements collected by protocol.
        """
        return self._manager.measure()

    def __call__(
        self,
        observables: list[AbstractBlock] = list(),
    ) -> Tensor:
        """Shortcut for obtaining expectation values.

        Args:
            observables (list[AbstractBlock], optional): List of observables to evaluate on.
                Defaults to the model observables if an empty list is provided.
            data (MeasurementData, optional): Previously obtained measurement data if any.

        Returns:
            Tensor: Expectation values.
        """
        return self.expectation(observables)
