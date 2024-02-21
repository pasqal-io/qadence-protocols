from __future__ import annotations

import importlib

from qadence_protocols.types import qadence_available_protocols


def available_protocols() -> dict:
    """Return the available protocols."""

    qadence_protocols: dict = dict()

    for protocol in qadence_available_protocols:
        module = importlib.import_module(f"qadence_protocols.{protocol}.protocols")
        ProtocolCls = getattr(module, protocol.capitalize())
        qadence_protocols[protocol] = ProtocolCls

    return qadence_protocols
