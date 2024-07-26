from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import partial
from typing import Callable

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

    def mitigation(self) -> Callable:
        try:
            module = importlib.import_module(PROTOCOL_TO_MODULE[self.protocol])
        except KeyError:
            ImportError(f"The module for the protocol {self.protocol} is not implemented.")
        # Partially pass the options.
        return partial(getattr(module, "mitigate"), options=self.options)
