from __future__ import annotations

from functools import partial

import torch
from qadence import QuantumModel
from qadence.blocks.abstract import AbstractBlock
from torch import Tensor

from qadence_protocols.measurements.shadow import ShadowManager
from qadence_protocols.measurements.utils_shadow import (
    compute_snapshots,
    expectation_estimations,
    robust_local_shadow,
    shadow_samples,
)
from qadence_protocols.types import MeasurementData


class RobustShadowManager(ShadowManager):
    """The class for managing randomized robust shadow."""

    def __init__(
        self,
        options: dict,
        model: QuantumModel | None = None,
        observables: list[AbstractBlock] = list(),
        param_values: dict[str, Tensor] = dict(),
        state: Tensor | None = None,
        data: MeasurementData = MeasurementData(),
    ):
        super().__init__(options, model, observables, param_values, state, data)

    def validate_options(self, options: dict) -> dict:
        """Extract shadow_size, shadow_medians and calibration from options.

        Args:
            options (dict): Input options for tomography

        Raises:
            KeyError: If `shadow_size` or `shadow_medians` absent from options.

        Returns:
            dict: Validated options.
        """

        shadow_size = options.get("shadow_size", None)
        if shadow_size is None:
            raise KeyError("Robust Shadow protocol requires an option 'shadow_size' of type 'int'.")
        n_shots = options.get("n_shots", 1)

        shadow_medians = options.get("shadow_medians", None)
        if shadow_medians is None:
            raise KeyError("Shadow protocol requires an option 'shadow_medians' of type 'int'.")

        calibration = options.get("calibration", None)

        validated_options = {
            "shadow_size": shadow_size,
            "n_shots": n_shots,
            "shadow_medians": shadow_medians,
            "calibration": calibration,
        }
        return validated_options

    def measure(
        self,
    ) -> MeasurementData:
        """Obtain measurement data from a quantum program for classical shadows.

        Note the observables are not used here.

        Returns:
            MeasurementData: Measurement data as locally sampled pauli unitaries and
                samples from the circuit
                rotated according to the locally sampled pauli unitaries.
        """
        if self.model is None:
            raise ValueError("Please provide a model to run protocol.")

        circuit = self.model._circuit.original
        shadow_size = self.options["shadow_size"]

        self.data = shadow_samples(
            shadow_size=shadow_size,
            circuit=circuit,
            param_values=self.model.embedding_fn(self.model._params, self.param_values),
            state=self.state,
            backend=self.model.backend,
            noise=self.model._noise,
            n_shots=self.options["n_shots"],
        )
        return self.data

    def snapshots(
        self,
    ) -> Tensor:
        """Obtain snapshots from the measurement data.

        Returns:
            list[Tensor]: Snapshots for a input circuit model and state.
                The shape is (batch_size, shadow_size, 2**n, 2**n).
        """
        if self.data.samples is None:
            self.measure()

        calibration = self.options["calibration"]
        if calibration is None:
            calibration = torch.tensor([1.0 / 3.0] * self.data.unitaries.shape[1])

        caller = partial(robust_local_shadow, calibration=calibration)

        return compute_snapshots(self.data.samples, self.data.unitaries, caller)

    def expectation(
        self,
        observables: list[AbstractBlock] = list(),
    ) -> Tensor:
        """Compute expectation values by medians of means from the measurement data.

        Args:
            observables (list[AbstractBlock], optional): List of observables.
            Defaults to the model observables if an empty list is provided.
            Can be different from the observables passed at initialization.

        Returns:
            Tensor: Expectation values.
        """
        if self.model is None:
            raise ValueError("Please provide a model to run protocol.")

        K = int(self.options["shadow_medians"])
        calibration = self.options["calibration"]
        if calibration is None:
            calibration = torch.tensor([1.0 / 3.0] * self.model._circuit.original.n_qubits)

        if self.data.samples.numel() == 0:  # type: ignore[union-attr]
            self.measure()

        observables = (
            observables
            if len(observables) > 0
            else [obs.abstract for obs in self.model._observable]
        )
        return expectation_estimations(
            observables,
            self.data.unitaries,
            self.data.samples,
            K,
            calibration=calibration,
        )
