from __future__ import annotations

from typing import Any, Iterable

import torch
from qadence import QuantumModel
from qadence.blocks.abstract import AbstractBlock
from qadence.blocks.utils import unroll_block_with_scaling
from torch import Tensor

from qadence_protocols.measurements.abstract import MeasurementManager
from qadence_protocols.measurements.protocols import MeasurementData
from qadence_protocols.measurements.utils_tomography import (
    convert_samples_to_pauli_expectation,
    iterate_pauli_decomposition,
)


def flatten_recursive(lst: list[Any]) -> Iterable[Any]:
    """Flatten a list using recursion."""
    for item in lst:
        if isinstance(item, list):
            yield from flatten_recursive(item)
        else:
            yield item


class Tomography(MeasurementManager):
    """The abstract class that defines the interface for the managing measurements."""

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
        """Verify options contain `n_shots`.

        Args:
            options (dict): Input options for tomography

        Raises:
            KeyError: If `n_shots` absent from options.

        Returns:
            dict: Options if correct.
        """
        n_shots = options.get("n_shots")
        if n_shots is None:
            raise KeyError("Tomography protocol requires a 'n_shots' kwarg of type 'int').")
        return options

    def validate_data(self, data: MeasurementData) -> MeasurementData:
        """Validate passed data.

        Raises:
            ValueError: If data passed does not correspond to the typical tomography data.
        """
        if data.unitaries is not None:
            raise ValueError("Tomography data cannot have `unitaries` filled.")

        if data.measurements is not None:
            if len(data.measurements) != len(self.observables):
                raise ValueError(
                    "Provide data as a list of Counters matching the number of observables."
                )
            n_shots = self.options["n_shots"]
            for obs_measurements in data.measurements:
                for iter_pauli_meas in obs_measurements:
                    for counter in iter_pauli_meas:
                        if sum(counter.values()) != n_shots:
                            raise ValueError(
                                f"The frequencies in each counter must sum up to {n_shots}"
                            )
        return data

    def reconstruct_state(self) -> Tensor:
        raise NotImplementedError

    def measure(
        self,
    ) -> MeasurementData:
        """Obtain measurements by sampling via rotated circuits in Z-basis.

        Note that one needs the observable.

        Returns:
            MeasurementData: Measurements collected by tomography.
        """
        n_shots = self.options["n_shots"]
        circuit = self.model._circuit.original

        samples = []
        for observable in self.observables:
            pauli_decomposition = unroll_block_with_scaling(observable)
            samples.append(
                iterate_pauli_decomposition(
                    circuit=circuit,
                    param_values=self.model.embedding_fn(self.model._params, self.param_values),
                    pauli_decomposition=pauli_decomposition,
                    n_shots=n_shots,
                    state=self.state,
                    backend=self.model.backend,
                    noise=self.model._noise,
                )
            )
        self.data = MeasurementData(samples)
        return self.data

    def expectation(
        self,
        observables: list[AbstractBlock] = list(),
    ) -> Tensor:
        """Set new observables and compute expectation values from the model .

        Sampling is performed via rotated circuits in Z-basis.

        Returns:
            Tensor: Expectation values
        """
        observables = observables if len(observables) > 0 else self.observables
        if self.data.measurements is None:
            self.observables = observables
            self.measure()

        estimated_values = []
        for samples, observable in zip(self.data.measurements, observables):  # type: ignore[arg-type]
            estimated_values.append(
                convert_samples_to_pauli_expectation(samples, unroll_block_with_scaling(observable))
            )
        return torch.transpose(torch.vstack(estimated_values), 1, 0)
