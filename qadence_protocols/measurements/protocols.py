from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import partial

from qadence import QuantumModel
from torch import Tensor

from qadence_protocols.protocols import Protocol

PROTOCOL_TO_MODULE = {
    "tomography": "qadence_protocols.measurements.tomography",
}


@dataclass
class Measurements(Protocol):
    TOMOGRAPHY = "tomography"

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
        except KeyError:
            ImportError(f"The module for the protocol {self.protocol} is not implemented.")
        # Partially pass the options.
        expectation = partial(getattr(module, "compute_expectation"), options=self.options)
        return expectation(model, param_values=param_values)
