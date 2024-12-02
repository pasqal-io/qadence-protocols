from __future__ import annotations

from qadence import QuantumModel
from qadence.blocks.abstract import AbstractBlock
from torch import Tensor

from qadence_protocols.measurements.abstract import MeasurementManager
from qadence_protocols.measurements.utils_shadow import (
    compute_snapshots,
    expectation_estimations,
    local_shadow,
    number_of_samples,
    shadow_samples,
)
from qadence_protocols.types import MeasurementData


class ShadowManager(MeasurementManager):
    """The class for managing randomized classical shadow."""

    def __init__(
        self,
        options: dict,
        model: QuantumModel,
        observables: list[AbstractBlock] = list(),
        param_values: dict[str, Tensor] = dict(),
        state: Tensor | None = None,
        data: MeasurementData = MeasurementData(),
    ):
        self.options = self.validate_options(options)
        self.model = model

        self.observables = (
            observables if len(observables) > 0 else [obs.abstract for obs in model._observable]
        )
        self.param_values = param_values
        self.state = state
        self.data = self.validate_data(data)

    def validate_options(self, options: dict) -> dict:
        """Extract shadow_size, accuracy and confidence from options.

        Args:
            options (dict): Input options for tomography

        Raises:
            KeyError: If `n_shots` absent from options.

        Returns:
            dict: Validated options.
        """

        shadow_size = options.get("shadow_size", None)
        accuracy = options.get("accuracy", None)
        if shadow_size is None and accuracy is None:
            raise KeyError(
                "Shadow protocol requires either an option"
                " 'shadow_size' of type 'int' or 'accuracy' of type 'float'."
            )
        confidence = options.get("confidence", None)
        if confidence is None:
            raise KeyError("Shadow protocol requires an option 'confidence' of type 'float'.")
        validated_options = {
            "shadow_size": shadow_size,
            "accuracy": accuracy,
            "confidence": confidence,
        }
        return validated_options

    def validate_data(self, data: MeasurementData) -> MeasurementData:
        """Validate passed data.

        Raises:
            ValueError: If data passed does not correspond to the typical shadow data.
        """
        if data.samples is None:
            return data

        if data.unitaries is None:
            raise ValueError("Shadow data must have `unitaries` filled.")

        if not isinstance(data.samples, Tensor):
            raise ValueError("`measurements` must be a Tensor.")

        if len(data.unitaries.size()) != 2:
            raise ValueError("Provide correctly the unitaries as a 2D Tensor.")

        if len(data.samples.size()) != 3:
            raise ValueError("Provide correctly the measurements as a 3D Tensor.")

        shadow_size = self.options["shadow_size"]
        if not (data.unitaries.shape[0] == data.samples.shape[1] == shadow_size):
            raise ValueError(
                f"Provide correctly data as Tensors with {shadow_size} `shadow_size` elements."
            )

        n_qubits = self.model._circuit.original.n_qubits
        if not (data.unitaries.shape[1] == data.samples.shape[2] == n_qubits):
            raise ValueError(
                f"Provide correctly data as Tensors with {n_qubits} `qubits` in the last dimension."
            )
        return data

    def reconstruct_state(
        self,
    ) -> Tensor:
        """Reconstruct the state from the snapshots.

        Returns:
            Tensor: Reconstructed state.
        """
        snapshots = self.snapshots()

        N = snapshots.shape[1]
        return snapshots.sum(axis=1) / N

    def snapshots(
        self,
    ) -> Tensor:
        """Obtain snapshots from the measurement data.

        Args:
            model (QuantumModel): Quantum model instance.
            param_values (dict[str, Tensor], optional): Parameter values. Defaults to dict().
            state (Tensor | None, optional): Input state. Defaults to None.

        Returns:
            Tensor: Snapshots for a input circuit model and state.
                The shape is (batch_size, shadow_size, 2**n, 2**n).
        """
        if self.data.samples is None:
            self.measure()

        return compute_snapshots(self.data.samples, self.data.unitaries, local_shadow)

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

        circuit = self.model._circuit.original
        shadow_size = self.options["shadow_size"]
        accuracy = self.options["accuracy"]
        confidence = self.options["confidence"]

        if shadow_size is None:
            shadow_size = number_of_samples(
                observables=self.observables, accuracy=accuracy, confidence=confidence
            )[0]

        self.data = shadow_samples(
            shadow_size=shadow_size,
            circuit=circuit,
            param_values=self.model.embedding_fn(self.model._params, self.param_values),
            state=self.state,
            backend=self.model.backend,
            noise=self.model._noise,
        )
        return self.data

    def expectation(
        self,
        observables: list[AbstractBlock] = list(),
    ) -> Tensor:
        """Compute expectation values by medians of means from the measurement data.

        Args:
            observables (list[AbstractBlock], optional): List of observables.
            Can be different from the observables passed at initialization.
            Defaults to the model observables if an empty list is provided.

        Returns:
            Tensor: Expectation values.
        """
        accuracy = self.options["accuracy"]
        confidence = self.options["confidence"]
        observables = (
            observables
            if len(observables) > 0
            else [obs.abstract for obs in self.model._observable]
        )
        _, K = number_of_samples(observables=observables, accuracy=accuracy, confidence=confidence)

        if self.data.samples is None:
            self.measure()

        return expectation_estimations(observables, self.data.unitaries, self.data.samples, K)
