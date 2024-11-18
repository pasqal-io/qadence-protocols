from __future__ import annotations

import torch
from qadence import QuantumModel
from qadence.blocks.abstract import AbstractBlock
from qadence.blocks.utils import unroll_block_with_scaling
from torch import Tensor

from qadence_protocols.measurements.abstract import MeasurementData, MeasurementManager
from qadence_protocols.measurements.utils_tomography import (
    convert_samples_to_pauli_expectation,
    iterate_pauli_decomposition,
)


class Tomography(MeasurementManager):
    """The abstract class that defines the interface for the managing measurements."""

    def __init__(self, measurement_data: MeasurementData = None, options: dict = dict()):
        self.measurement_data = measurement_data
        self.options = options

    def verify_options(self) -> dict:
        """Verify options contain `n_shots`.

        Raises:
            KeyError: If `n_shots` absent from options.

        Returns:
            dict: Options if correct.
        """
        n_shots = self.options.get("n_shots")
        if n_shots is None:
            raise KeyError("Tomography protocol requires a 'n_shots' kwarg of type 'int').")
        return self.options

    def measure(
        self,
        model: QuantumModel,
        observables: list[AbstractBlock] = list(),
        param_values: dict[str, Tensor] = dict(),
        state: Tensor | None = None,
    ) -> MeasurementData:
        """Obtain measurements by sampling via rotated circuits in Z-basis.

        Args:
            model (QuantumModel): Model to evaluate.
            observables (list[AbstractBlock], optional): A list of observables
                to estimate the expectation values from. Defaults to list().
            param_values (dict[str, Tensor], optional): Parameter values. Defaults to dict().
            state (Tensor | None, optional): Input state. Defaults to None.

        Returns:
            MeasurementData: Measurements collected by tomography.
        """
        n_shots = self.options["n_shots"]
        circuit = model._circuit.original

        samples = []
        for observable in observables:
            pauli_decomposition = unroll_block_with_scaling(observable)
            samples.append(
                iterate_pauli_decomposition(
                    circuit=circuit,
                    param_values=model.embedding_fn(model._params, param_values),
                    pauli_decomposition=pauli_decomposition,
                    n_shots=n_shots,
                    state=state,
                    backend=model.backend,
                    noise=model._noise,
                )
            )
        self.measurement_data = samples
        return samples

    def expectation(
        self,
        model: QuantumModel,
        observables: list[AbstractBlock] = list(),
        param_values: dict[str, Tensor] = dict(),
        state: Tensor | None = None,
    ) -> Tensor:
        """Compute expectation values from the model by sampling via rotated circuits in Z-basis.

        Args:
            model (QuantumModel): Model to evaluate.
            observables (list[AbstractBlock], optional): A list of observables
                to estimate the expectation values from. Defaults to list().
            param_values (dict[str, Tensor], optional): Parameter values. Defaults to dict().
            state (Tensor | None, optional): Input state. Defaults to None.

        Returns:
            Tensor: Expectation values
        """
        if self.measurement_data is None:
            self.measure(model, observables, param_values, state)

        estimated_values = []
        for samples, observable in zip(self.measurement_data, observables):
            estimated_values.append(
                convert_samples_to_pauli_expectation(samples, unroll_block_with_scaling(observable))
            )
        return torch.transpose(torch.vstack(estimated_values), 1, 0)
