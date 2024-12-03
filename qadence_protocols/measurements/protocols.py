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
    Note if you already have experimental data, this can be passed in the options with a `data` key.

    Attributes:
        protocol (str): Protocol name.
        options (dict, optional): Options to run protocol.
    """

    def __init__(
        self,
        protocol: str,
        options: dict = dict(),
    ) -> None:
        try:
            module = importlib.import_module(PROTOCOL_TO_MODULE[protocol][0])
            proto_class = getattr(module, PROTOCOL_TO_MODULE[protocol][1])
        except (KeyError, ModuleNotFoundError, ImportError) as e:
            raise type(e)(f"Failed to import Measurements due to {e}.")

        super().__init__(protocol, options)

        # Note that options and data are validated inside the manager
        self._manager: MeasurementManager = proto_class(
            options, data=options.get("data", MeasurementData())
        )

    @property
    def data(self) -> MeasurementData:
        return self._manager.data

    @data.setter
    def data(self, new_data: MeasurementData) -> None:
        self._manager.data = self._manager.validate_data(new_data)

    def _reset_manager(
        self,
        model: QuantumModel,
        observables: list[AbstractBlock] = list(),
        param_values: dict[str, Tensor] = dict(),
        state: Tensor | None = None,
    ) -> None:
        """Reset attributes of manager for a given model, observables, param_values and state.

        Note we do not reset the data.

        Args:
            model (QuantumModel): Quantum model instance.
            observables (list[AbstractBlock], optional): List of observables. Defaults to list().
            param_values (dict[str, Tensor], optional): Parameter values. Defaults to dict().
            state (Tensor | None, optional): Input state. Defaults to None.
        """

        # assume the data is already obtained or default MeasurementData
        self._manager.__init__(  # type: ignore[misc]
            self.options, model, observables, param_values, state, self._manager.data
        )

    def expectation(
        self,
        model: QuantumModel,
        observables: list[AbstractBlock] = list(),
        param_values: dict[str, Tensor] = dict(),
        state: Tensor | None = None,
    ) -> Tensor:
        """Compute expectation values from the model by sampling.

        Args:
            model (QuantumModel): Quantum model instance.
            observables (list[AbstractBlock], optional): List of observables. Defaults to list().
            param_values (dict[str, Tensor], optional): Parameter values. Defaults to dict().
            state (Tensor | None, optional): Input state. Defaults to None.

        Returns:
            Tensor: Expectation values
        """

        self._reset_manager(model, observables, param_values, state)
        return self._manager.expectation(observables)

    def reconstruct_state(
        self,
    ) -> Tensor:
        """Reconstruct the state from the data.

        Used only in shadow protocols.

        Args:
            model (QuantumModel): Quantum model instance.
            observables (list[AbstractBlock], optional): List of observables. Defaults to list().
            param_values (dict[str, Tensor], optional): Parameter values. Defaults to dict().
            state (Tensor | None, optional): Input state. Defaults to None.

        Returns:
            Tensor: Reconstructed state.
        """
        if self._manager.model is None:
            raise ValueError(
                "Cannot call `reconstruct_state` without "
                "defining a model in `__call__` or `measure`."
            )
        return self._manager.reconstruct_state()

    def measure(
        self,
        model: QuantumModel,
        observables: list[AbstractBlock] = list(),
        param_values: dict[str, Tensor] = dict(),
        state: Tensor | None = None,
    ) -> MeasurementData:
        """Obtain measurements by sampling using the protocol.

        Returns:
            MeasurementData: Measurements collected by protocol.
        """
        self._reset_manager(model, observables, param_values, state)
        return self._manager.measure()

    def __call__(
        self,
        model: QuantumModel,
        observables: list[AbstractBlock] = list(),
        param_values: dict[str, Tensor] = dict(),
        state: Tensor | None = None,
    ) -> Tensor:
        """Shortcut for obtaining expectation values.

        Args:
            model (QuantumModel): Quantum model instance.
            observables (list[AbstractBlock], optional): List of observables. Defaults to list().
            param_values (dict[str, Tensor], optional): Parameter values. Defaults to dict().
            state (Tensor | None, optional): Input state. Defaults to None.

        Returns:
            Tensor: Expectation values.
        """
        return self.expectation(model, observables, param_values, state)
