from __future__ import annotations

import importlib
from string import Template
from typing import Protocol, TypeVar

from qadence.backend import Backend
from qadence.blocks.abstract import AbstractBlock
from qadence.extensions import _available_backends as qd_backends
from qadence.extensions import _available_engines as qd_engines
from qadence.extensions import _set_backend_config as qd_backend_config
from qadence.extensions import (
    _supported_gates as qd_gates,
)

from qadence_protocols.types import (
    qadence_protocols
)

TAbstractBlock = TypeVar("TAbstractBlock", bound=AbstractBlock)


ext_backends_namespace = Template("qadence_extensions.backends.$name")
qd_backends_namespace = Template("qadence.backends.$name")

qd_ext_backends = ["emu_c"]



def available_protocols() -> dict:
    """Return the available protocols"""

    qadence_protocols: dict = dict()

    for protocol in qadence_protocols:
        module = importlib.import_module(
            f"qadence_protocols.{protocol}.protocols"
        )
        ProtocolCls = getattr(module, protocol.capitalize())
        qadence_protocols[protocol] = ProtocolCls

    return qadence_backends
