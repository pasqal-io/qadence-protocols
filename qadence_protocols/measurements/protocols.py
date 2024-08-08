from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import partial

from qadence import QuantumModel
from torch import Tensor

from qadence_protocols.protocols import Protocol

PROTOCOL_TO_MODULE = {
    "tomography": "qadence_protocols.measurements.tomography",
    "shadow": "qadence_protocols.measurements.shadow",
}


@dataclass
class Measurements(Protocol):
    TOMOGRAPHY = "tomography"
    SHADOW = "shadow"

    def __init__(self, protocol: str, options: dict = dict()) -> None:
        super().__init__(protocol, options)

    def __call__(
        self,
        model: QuantumModel,
        param_values: dict[str, Tensor] = dict(),
    ) -> Tensor:
        """Compute expectation values via measurements.

        Args:
            model (QuantumModel): Model to evaluate.
            param_values (dict[str, Tensor], optional): Parameter values. Defaults to dict().

        Returns:
            Tensor: Expectation values.
        """
        try:
            module = importlib.import_module(PROTOCOL_TO_MODULE[self.protocol])
        except (KeyError, ModuleNotFoundError, ImportError) as e:
            raise type(e)(f"Failed to import Mitigations due to {e}.")

        conv_observables = model._observable
        observables = [obs.abstract for obs in conv_observables]

        # Partially pass the options and observable.
        expectation = partial(
            getattr(module, "compute_expectation"), observables=observables, options=self.options
        )

        return expectation(model, param_values=param_values)
