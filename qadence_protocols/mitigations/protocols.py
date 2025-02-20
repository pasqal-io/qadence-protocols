from __future__ import annotations

import importlib
from collections import Counter
from dataclasses import dataclass

from qadence import NoiseHandler, QuantumModel
from torch import Tensor

from qadence_protocols.protocols import Protocol

PROTOCOL_TO_MODULE = {
    "twirl": "qadence_protocols.mitigations.twirl",
    "readout": "qadence_protocols.mitigations.readout",
    "zne": "qadence_protocols.mitigations.analog_zne",
}


@dataclass
class Mitigations(Protocol):
    TWIRL = "twirl"
    READOUT = "readout"
    ANALOG_ZNE = "zne"

    def __init__(self, protocol: str, options: dict = dict()) -> None:
        super().__init__(protocol, options)

    def __call__(
        self,
        noise: NoiseHandler,
        model: QuantumModel | None = None,
        param_values: dict[str, Tensor] = dict(),
    ) -> list[Counter]:
        if noise is None:
            raise ValueError(
                "A noise model must be provided to .mitigate()"
            )
        try:
            module = importlib.import_module(PROTOCOL_TO_MODULE[self.protocol])
        except (KeyError, ModuleNotFoundError, ImportError) as e:
            raise type(e)(f"Failed to import Mitigations due to {e}.")
        migitation_fn = getattr(module, "mitigate")
        mitigated_counters: list[Counter] = migitation_fn(
            model=model, options=self.options, noise=noise, param_values=param_values
        )
        return mitigated_counters
