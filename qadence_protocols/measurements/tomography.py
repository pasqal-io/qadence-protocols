from __future__ import annotations

import torch
from qadence import QuantumModel
from qadence.blocks.utils import unroll_block_with_scaling
from torch import Tensor

from qadence_protocols.measurements.utils import iterate_pauli_decomposition


def compute_expectation(
    model: QuantumModel,
    options: dict,
    param_values: dict[str, Tensor] = dict(),
) -> Tensor:
    """Compute expectation values from the model by sampling via rotated circuits in Z-basis.

    Args:
        model (QuantumModel): Model to evaluate.
        options (dict): _description_
        param_values (dict[str, Tensor], optional): _description_. Defaults to dict().

    Raises:
        TypeError: If observables are not given as list
        KeyError: If a 'n_shots' kwarg of type 'int' is missing in options.

    Returns:
        Tensor: Expectation values
    """

    observables = model._observable
    if not isinstance(observables, list):
        raise TypeError(
            "Observables must be of type <class 'List[AbstractBlock]'>. Got {}.".format(
                type(observables)
            )
        )

    n_shots = options.get("n_shots")
    if n_shots is None:
        raise KeyError("Tomography protocol requires a 'n_shots' kwarg of type 'int').")

    circuit = model._circuit.original

    estimated_values = []
    for observable in observables:
        pauli_decomposition = unroll_block_with_scaling(observable.abstract)
        estimated_values.append(
            iterate_pauli_decomposition(
                circuit=circuit,
                param_values=param_values,
                pauli_decomposition=pauli_decomposition,
                n_shots=n_shots,
                backend=model.backend,
                noise=model._noise,
            )
        )
    return torch.transpose(torch.vstack(estimated_values), 1, 0)
