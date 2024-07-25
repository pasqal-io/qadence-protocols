from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import partial
from typing import Callable

PROTOCOL_TO_MODULE = {
    "tomography": "qadence_protocols.measurements.tomography",
}


@dataclass
class Measurements:
    TOMOGRAPHY = "tomography"

    def __init__(self, protocol: str, options: dict = dict()) -> None:
        self.protocol: str = protocol
        self.options: dict = options

    def compute_expectation(self) -> Callable:
        try:
            module = importlib.import_module(PROTOCOL_TO_MODULE[self.protocol])
        except KeyError:
            ImportError(f"The module for the protocol {self.protocol} is not implemented.")
        # Partially pass the options.
        return partial(getattr(module, "compute_expectation"), options=self.options)

    def _to_dict(self) -> dict:
        return {"protocol": self.protocol, "options": self.options}

    @classmethod
    def _from_dict(cls, d: dict) -> Measurements | None:
        if d:
            return cls(d["protocol"], **d["options"])
        return None

    @classmethod
    def list(cls) -> list:
        return list(filter(lambda el: not el.startswith("__"), dir(cls)))
