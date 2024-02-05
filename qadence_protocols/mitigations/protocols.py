from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import partial
from typing import Callable

PROTOCOL_TO_MODULE = {
    "twirl": "qadence_protocols.mitigations.twirl",
    "readout": "qadence_protocols.mitigations.readout",
    "zne": "qadence_protocols.mitigations.analog_zne",
}


@dataclass
class Mitigations:
    TWIRL = "twirl"
    READOUT = "readout"
    ANALOG_ZNE = "zne"

    def __init__(self, protocol: str, options: dict = dict()) -> None:
        self.protocol: str = protocol
        self.options: dict = options

    def mitigation(self) -> Callable:
        try:
            module = importlib.import_module(PROTOCOL_TO_MODULE[self.protocol])
        except KeyError:
            ImportError(f"The module for the protocol {self.protocol} is not implemented.")
        # Partially pass the options.
        return partial(getattr(module, "mitigate"), options=self.options)

    def _to_dict(self) -> dict:
        return {"protocol": self.protocol, "options": self.options}

    @classmethod
    def _from_dict(cls, d: dict) -> Mitigations | None:
        if d:
            return cls(d["protocol"], **d["options"])
        return None

    @classmethod
    def list(cls) -> list:
        return list(filter(lambda el: not el.startswith("__"), dir(cls)))
