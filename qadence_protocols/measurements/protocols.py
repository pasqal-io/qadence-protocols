from __future__ import annotations

import importlib
from functools import partial

from qadence import QuantumModel
from torch import Tensor

from qadence_protocols.measurements.abstract import MeasurementManager
from qadence_protocols.protocols import Protocol

PROTOCOL_TO_MODULE = {
    "tomography": ("qadence_protocols.measurements.tomography", "Tomography"),
    "shadow": ("qadence_protocols.measurements.shadow", "ShadowManager"),
    "robust_shadow": ("qadence_protocols.measurements.robust_shadow", "RobustShadowManager"),
}


class Measurements(Protocol):
    """Define a measurement protocol using a `MeasurementManager`.

    Possible options are available via the `MeasurementProtocols` type.
    The `MeasurementManager` instance has a verify_options

    Attributes:
        protocol (str): Protocol name.
        options (dict): Options to run protocol.
    """

    def __init__(self, protocol: str, options: dict = dict()) -> None:
        try:
            module = importlib.import_module(PROTOCOL_TO_MODULE[protocol][0])
            proto_class = getattr(module, PROTOCOL_TO_MODULE[protocol][1])
        except (KeyError, ModuleNotFoundError, ImportError) as e:
            raise type(e)(f"Failed to import Mitigations due to {e}.")
        self.manager: MeasurementManager = proto_class(data=None, options=options)
        verified_options = self.manager.validate_options()
        super().__init__(protocol, verified_options)

    def __call__(
        self,
        model: QuantumModel,
        param_values: dict[str, Tensor] = dict(),
        state: Tensor | None = None,
        return_expectations: bool = True,
    ) -> Tensor:
        """Compute measurements or expectation values via measurements.

        Args:
            model (QuantumModel): Model to evaluate.
            param_values (dict[str, Tensor], optional): Parameter values. Defaults to dict().
            state (Tensor | None, optional): Input state. Defaults to None.

        Returns:
            Tensor: Expectation values.
        """

        conv_observables = model._observable
        observables = [obs.abstract for obs in conv_observables]

        # Partially pass the options and observable.
        compute_fn = "expectation" if return_expectations else "measure"
        output_fn = partial(
            getattr(self.manager, compute_fn),
            observables=observables,
            state=state,
        )

        return output_fn(model, param_values=param_values)
