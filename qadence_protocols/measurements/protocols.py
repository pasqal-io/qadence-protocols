from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import partial

from qadence import QuantumModel
from torch import Tensor

from qadence_protocols.measurements.abstract import MeasurementManager
from qadence_protocols.protocols import Protocol

PROTOCOL_TO_MODULE = {
    "tomography": "qadence_protocols.measurements.tomography.Tomography",
    "shadow": "qadence_protocols.measurements.shadow.Shadows",
    "robust_shadow": "qadence_protocols.measurements.robust_shadow.RobustShadows",
}


@dataclass
class MeasurementOptions:
    def __init__(self, protocol: str, options: dict = dict()) -> None:
        self.protocol = protocol
        self.options = options

        self.OPTIONS_TO_VERIFY = {
            "shadow": self.verify_classical_shadow,
            "robust_shadow": self.verify_robust_shadow,
            "tomography": self.verify_tomography,
        }

    def verify_options(self) -> dict:
        return self.OPTIONS_TO_VERIFY[self.protocol]()

    def verify_classical_shadow(self) -> dict:
        """Extract shadow_size, accuracy and confidence from options."""

        shadow_size = self.options.get("shadow_size", None)
        accuracy = self.options.get("accuracy", None)
        if shadow_size is None and accuracy is None:
            raise KeyError(
                "Shadow protocol requires either an option"
                " 'shadow_size' of type 'int' or 'accuracy' of type 'float'."
            )
        confidence = self.options.get("confidence", None)
        if confidence is None:
            raise KeyError("Shadow protocol requires an option 'confidence' of type 'float'.")

        return {"shadow_size": shadow_size, "accuracy": accuracy, "confidence": confidence}

    def verify_tomography(self) -> dict:
        n_shots = self.options.get("n_shots")
        if n_shots is None:
            raise KeyError("Tomography protocol requires a 'n_shots' kwarg of type 'int').")

        return {"n_shots": n_shots}

    def verify_robust_shadow(self) -> dict:
        """Extract shadow_size, accuracy and confidence from options."""

        shadow_size = self.options.get("shadow_size", None)
        if shadow_size is None:
            raise KeyError("Robust Shadow protocol requires an option 'shadow_size' of type 'int'.")
        shadow_groups = self.options.get("shadow_groups", None)
        if shadow_groups is None:
            raise KeyError("Shadow protocol requires an option 'shadow_groups' of type 'int'.")

        calibration = self.options.get("calibration", None)

        return {
            "shadow_size": shadow_size,
            "shadow_groups": shadow_groups,
            "calibration": calibration,
        }


class Measurements(Protocol):
    def __init__(self, protocol: str, options: dict = dict()) -> None:
        try:
            module = importlib.import_module(PROTOCOL_TO_MODULE[self.protocol])
        except (KeyError, ModuleNotFoundError, ImportError) as e:
            raise type(e)(f"Failed to import Mitigations due to {e}.")
        self.measurement_manager: MeasurementManager = module(measurement_data=None, **options)
        verified_options = self.measurement_manager.verify_options()
        super().__init__(protocol, verified_options)

    def __call__(
        self,
        model: QuantumModel,
        param_values: dict[str, Tensor] = dict(),
        return_expectations: bool = True,
    ) -> Tensor:
        """Compute measurements or expectation values via measurements.

        Args:
            model (QuantumModel): Model to evaluate.
            param_values (dict[str, Tensor], optional): Parameter values. Defaults to dict().

        Returns:
            Tensor: Expectation values.
        """

        conv_observables = model._observable
        observables = [obs.abstract for obs in conv_observables]

        # Partially pass the options and observable.
        compute_fn = "expectation" if return_expectations else "measure"
        output_fn = partial(
            getattr(self.measurement_manager, compute_fn),
            observables=observables,
            options=self.options,
        )

        return output_fn(model, param_values=param_values)
