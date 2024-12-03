from __future__ import annotations

from collections import Counter
from functools import reduce

import numpy as np
import torch
from qadence.backend import Backend
from qadence.backends.pyqtorch import Backend as PyQBackend
from qadence.blocks import AbstractBlock, PrimitiveBlock, chain
from qadence.circuit import QuantumCircuit
from qadence.engines.differentiable_backend import DifferentiableBackend
from qadence.noise import NoiseHandler
from qadence.operations import H, I, SDagger, X, Y
from qadence.parameters import evaluate
from qadence.utils import Endianness
from sympy import Basic
from torch import Tensor


def get_qubit_indices_for_op(
    pauli_term: tuple[AbstractBlock, Basic], op: PrimitiveBlock | None = None
) -> list[int]:
    """Get qubit indices for the given op in the Pauli term if any.

    Args:
        pauli_term: Tuple of a Pauli block and a parameter.
        op: Tuple of Primitive blocks or None.

    Returns: A list of integers representing qubit indices.
    """
    blocks = getattr(pauli_term[0], "blocks", None) or [pauli_term[0]]
    indices = [block.qubit_support[0] for block in blocks if (op is None) or (type(block) is op)]
    return indices


def get_counts(samples: list, support: list[int]) -> list[Counter]:
    """Marginalise the probability mass function to the support.

    Args:
        samples: list of samples against which expectation value is to be computed.
        support: A list of integers representing qubit indices.

    Returns: A list[Counter] of bit strings.
    """
    return [
        reduce(
            lambda x, y: x + y,
            [Counter({"".join([k[i] for i in support]): sample[k]}) for k, v in sample.items()],
        )
        for sample in samples
    ]


def empirical_average(samples: list, support: list[int]) -> Tensor:
    """Compute the empirical average.

    Args:
        samples: list of samples against which expectation value is to be computed.
        support: A list of integers representing qubit indices.

    Returns: A torch.Tensor of the empirical average.
    """
    PARITY = -1
    counters = get_counts(samples, support)
    n_shots = np.sum(list(counters[0].values()))
    expectations = []
    for counter in counters:
        counter_exps = []
        for bitstring, count in counter.items():
            counter_exps.append(count * PARITY ** (np.sum([int(bit) for bit in bitstring])))
        expectations.append(np.sum(counter_exps) / n_shots)
    return torch.tensor(expectations)


def rotate(circuit: QuantumCircuit, pauli_term: tuple[AbstractBlock, Basic]) -> QuantumCircuit:
    """Rotate circuit to measurement basis and return the qubit support.

    Args:
        circuit: The circuit that is executed.
        pauli_term: Tuple of a Pauli term and a parameter.

    Returns: Rotated QuantumCircuit.
    """

    rotations = []

    for op, gate in [(X, I), (Y, SDagger)]:
        qubit_indices = get_qubit_indices_for_op(pauli_term, op=op)
        for index in qubit_indices:
            rotations += [gate(index), H(index)]
    rotated_block = chain(circuit.block, *rotations)
    return QuantumCircuit(circuit.register, rotated_block)


def iterate_pauli_decomposition(
    circuit: QuantumCircuit,
    param_values: dict[str, Tensor],
    pauli_decomposition: list[tuple[AbstractBlock, Basic]],
    n_shots: int,
    state: Tensor | None = None,
    backend: Backend | DifferentiableBackend = PyQBackend(),
    noise: NoiseHandler | None = None,
    endianness: Endianness = Endianness.BIG,
) -> list:
    """Sample circuits given all Pauli terms.

    Args:
        circuit: The circuit that is executed.
        param_values: Parameters of the circuit.
        pauli_decomposition: A list of Pauli decomposed terms.
        n_shots: Number of shots to sample.
        state: Initial state.
        backend: A backend for circuit execution.
        noise: A noise model to use.
        endianness: Endianness of the resulting bit strings.

    Returns: A torch.Tensor of bit strings n_shots x n_qubits.
    """

    sample_values = list()  # type: ignore [var-annotated]

    for pauli_term in pauli_decomposition:
        if pauli_term[0].is_identity:
            sample_values.append(list())
        else:
            # Rotate the circuit according to the given observable term.
            rotated_circuit = rotate(circuit=circuit, pauli_term=pauli_term)
            # Use the low-level backend API to avoid embedding of parameters
            # already performed at the higher QuantumModel level.
            # Therefore, parameters passed here have already been embedded.
            conv_circ = backend.circuit(rotated_circuit)
            sample_values.append(
                backend.sample(
                    circuit=conv_circ,
                    param_values=param_values,
                    n_shots=n_shots,
                    state=state,
                    noise=noise,
                    endianness=endianness,
                )
            )
    return sample_values


def convert_samples_to_pauli_expectation(
    samples_iteration: list,
    pauli_decomposition: list[tuple[AbstractBlock, Basic]],
) -> Tensor:
    """Estimate total expectation value by averaging samples from all Pauli terms.

    Args:
        circuit: The circuit that is executed.
        param_values: Parameters of the circuit.
        pauli_decomposition: A list of Pauli decomposed terms.
        n_shots: Number of shots to sample.
        state: Initial state.
        backend: A backend for circuit execution.
        noise: A noise model to use.
        endianness: Endianness of the resulting bit strings.

    Returns: A torch.Tensor of bit strings n_shots x n_qubits.
    """

    estimated_values = []

    for pauli_term, samples in zip(pauli_decomposition, samples_iteration):
        if pauli_term[0].is_identity:
            estimated_values.append(evaluate(pauli_term[1], as_torch=True))
        else:
            # Get the full qubit support for the Pauli term.
            # Note: duplicates must be kept here to allow for
            # observables chaining multiple operations on the same qubit
            # such as `b = chain(Z(0), Z(0))`
            support = get_qubit_indices_for_op(pauli_term)
            estim_values = empirical_average(samples=samples, support=support)
            # TODO: support for parametric observables to be tested
            estimated_values.append(estim_values * evaluate(pauli_term[1]))
    res = torch.sum(torch.stack(estimated_values), axis=0)
    # Allow for automatic differentiation.
    res.requires_grad = True
    return res
