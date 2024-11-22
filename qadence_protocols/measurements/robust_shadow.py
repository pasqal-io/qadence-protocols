from __future__ import annotations

from functools import partial

import torch
from qadence import QuantumModel
from qadence.blocks.abstract import AbstractBlock
from torch import Tensor

from qadence_protocols.measurements.abstract import MeasurementManager
from qadence_protocols.measurements.utils_shadow import (
    compute_snapshots,
    expectation_estimations,
    robust_local_shadow,
    shadow_samples,
)
from qadence_protocols.types import MeasurementData


class RobustShadowManager(MeasurementManager):
    """The class for managing randomized robust shadow."""

    def __init__(self, measurement_data: MeasurementData = None, options: dict = dict()):
        self.measurement_data = measurement_data
        self.options = options

    def verify_options(self) -> dict:
        """Extract shadow_size, accuracy and confidence from options."""

        shadow_size = self.options.get("shadow_size", None)
        if shadow_size is None:
            raise KeyError("Robust Shadow protocol requires an option 'shadow_size' of type 'int'.")
        shadow_groups = self.options.get("shadow_groups", None)
        if shadow_groups is None:
            raise KeyError("Shadow protocol requires an option 'shadow_groups' of type 'int'.")

        calibration = self.options.get("calibration", None)

        self.options = {
            "shadow_size": shadow_size,
            "shadow_groups": shadow_groups,
            "calibration": calibration,
        }
        return self.options

    def reconstruct_state(self, snapshots: Tensor) -> Tensor:
        """Reconstruct the state from the snapshots.

        Args:
            snapshots (Tensor): Snapshots of size
                (batch_size, shadow_size, 2**n, 2**n).

        Returns:
            Tensor: Reconstructed state.
        """
        N = snapshots.shape[1]
        return snapshots.sum(axis=1) / N

    def get_snapshots(
        self,
        model: QuantumModel,
        param_values: dict[str, Tensor] = dict(),
        state: Tensor | None = None,
    ) -> Tensor:
        """Obtain snapshots from the measurement data.

        Args:
            model (QuantumModel): Quantum model instance.
            param_values (dict[str, Tensor], optional): Parameter values. Defaults to dict().
            state (Tensor | None, optional): Input state. Defaults to None.

        Returns:
            list[Tensor]: Snapshots for a input circuit model and state.
                The shape is (batch_size, shadow_size, 2**n, 2**n).
        """
        if self.measurement_data is None:
            self.measure(model, list(), param_values, state)

        calibration = self.options["calibration"]
        if calibration is None:
            calibration = torch.tensor([1.0 / 3.0] * model._circuit.original.n_qubits)

        caller = partial(robust_local_shadow, calibration=calibration)

        unitaries_ids, bitstrings = self.measurement_data  # type: ignore[misc]
        return compute_snapshots(bitstrings, unitaries_ids, caller)

    def measure(
        self,
        model: QuantumModel,
        observables: list[AbstractBlock] = list(),
        param_values: dict[str, Tensor] = dict(),
        state: Tensor | None = None,
    ) -> MeasurementData:
        """Obtain measurement data from a quantum program for classical shadows.

        Args:
            model (QuantumModel): Quantum model instance.
            observables (list[AbstractBlock], optional): List of observables. Defaults to list().
            param_values (dict[str, Tensor], optional): Parameter values. Defaults to dict().
            state (Tensor | None, optional): Input state. Defaults to None.

        Returns:
            MeasurementData: Measurement data as locally sampled pauli unitaries and
                samples from the circuit
                rotated according to the locally sampled pauli unitaries.
        """

        circuit = model._circuit.original
        shadow_size = self.options["shadow_size"]

        self.measurement_data = shadow_samples(
            shadow_size=shadow_size,
            circuit=circuit,
            param_values=model.embedding_fn(model._params, param_values),
            state=state,
            backend=model.backend,
            noise=model._noise,
        )
        return self.measurement_data

    def expectation(
        self,
        model: QuantumModel,
        observables: list[AbstractBlock] = list(),
        param_values: dict[str, Tensor] = dict(),
        state: Tensor | None = None,
    ) -> Tensor:
        """Compute expectation values by medians of means from the measurement data.

        Args:
            model (QuantumModel): Quantum model instance.
            observables (list[AbstractBlock], optional): List of observables. Defaults to list().
            param_values (dict[str, Tensor], optional): Parameter values. Defaults to dict().
            state (Tensor | None, optional): Input state. Defaults to None.

        Returns:
            Tensor: Expectation values.
        """
        K = int(self.options["shadow_groups"])
        calibration = self.options["calibration"]
        if calibration is None:
            calibration = torch.tensor([1.0 / 3.0] * model._circuit.original.n_qubits)

        if self.measurement_data is None:
            self.measure(model, observables, param_values, state)

        unitaries_ids, batch_shadow_samples = self.measurement_data  # type: ignore[misc]
        return expectation_estimations(
            observables, unitaries_ids, batch_shadow_samples, K, calibration=calibration
        )
