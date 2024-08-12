from __future__ import annotations

import torch
from qadence import QuantumModel
from qadence.blocks.abstract import AbstractBlock
from qadence.blocks.utils import unroll_block_with_scaling
from torch import Tensor

from qadence_protocols.measurements.utils_tomography import iterate_pauli_decomposition


def compute_expectation(
    model: QuantumModel,
    observables: list[AbstractBlock],
    options: dict,
    param_values: dict[str, Tensor] = dict(),
    state: Tensor | None = None,
) -> Tensor:
    """Compute expectation values from the model by sampling via rotated circuits in Z-basis.

    Args:
        model (QuantumModel): Model to evaluate.
        observables (list[ConvertedObservable]): a list of observables
            to estimate the expectation values from.
        options (dict): Tomography options.
        Must be a dict with a 'n_shots' key and integer value.
        param_values (dict[str, Tensor], optional): Parameter values. Defaults to dict().
        state (Tensor, optional): Input state. Defaults to dict().

    Raises:
        TypeError: If observables are not given as list
        KeyError: If a 'n_shots' kwarg of type 'int' is missing in options.

    Returns:
        Tensor: Expectation values
    """

    n_shots = options.get("n_shots")
    if n_shots is None:
        raise KeyError("Tomography protocol requires a 'n_shots' kwarg of type 'int').")

    circuit = model._circuit.original

    estimated_values = []
    for observable in observables:
        pauli_decomposition = unroll_block_with_scaling(observable)
        estimated_values.append(
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
    return torch.transpose(torch.vstack(estimated_values), 1, 0)
