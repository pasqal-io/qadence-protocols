from __future__ import annotations

from collections import Counter
from functools import reduce

import numpy as np
import torch
from qadence.backend import Backend
from qadence.backends.pyqtorch import Backend as PyQBackend
from qadence.blocks import AbstractBlock, KronBlock, kron
from qadence.blocks.primitive import PrimitiveBlock
from qadence.blocks.utils import get_pauli_blocks, unroll_block_with_scaling
from qadence.circuit import QuantumCircuit
from qadence.engines.differentiable_backend import DifferentiableBackend
from qadence.noise import NoiseHandler
from qadence.operations import X, Y, Z
from qadence.transpile.noise import set_noise
from qadence.types import BackendName, Endianness, NoiseProtocol
from torch import Tensor

from qadence_protocols.measurements.utils_shadow.unitaries import UNITARY_TENSOR, pauli_rotations
from qadence_protocols.types import MeasurementData

batch_kron = torch.func.vmap(lambda x: reduce(torch.kron, x))


def _max_observable_weight(observable: AbstractBlock) -> int:
    """
    Get the maximal weight for the given observable.

    The weight is a measure of the locality of the observable,
    a count of the number of qubits on which the observable acts
    non-trivially.

    See https://arxiv.org/pdf/2002.08953.pdf
    Supplementary Material 1 and Eq. (S17).
    """
    pauli_decomposition = unroll_block_with_scaling(observable)
    weights = []
    for pauli_term in pauli_decomposition:
        weight = 0
        block = pauli_term[0]
        if isinstance(block, PrimitiveBlock):
            if isinstance(block, (X, Y, Z)):
                weight += 1
            weights.append(weight)
        else:
            pauli_blocks = get_pauli_blocks(block=block)
            weight = 0
            for block in pauli_blocks:
                if isinstance(block, (X, Y, Z)):
                    weight += 1
            weights.append(weight)
    return max(weights)


def maximal_weight(observables: list[AbstractBlock]) -> int:
    """Return the maximal weight if a list of observables is provided."""
    return max([_max_observable_weight(observable=observable) for observable in observables])


def number_of_samples(
    observables: list[AbstractBlock], accuracy: float | None = None, confidence: float | None = None
) -> tuple[int, ...]:
    """
    Estimate an optimal shot budget and a shadow partition size.

    to guarantee given accuracy on all observables expectation values
    within 1 - confidence range.

    See https://arxiv.org/pdf/2002.08953.pdf
    Supplementary Material 1 and Eqs. (S23)-(S24).

    If accuracy is None, we return the shot budget 0.
    If confidence is None, we return the shadow partition size 1.
    """
    max_k = maximal_weight(observables=observables)
    N = 0
    K = 1
    if accuracy and accuracy > 0:
        N = round(3**max_k * 34.0 / accuracy**2)
    if confidence and confidence > 0:
        K = round(2.0 * np.log(2.0 * len(observables) / confidence))
    return N, K


def nested_operator_indexing(
    idx_array: np.ndarray,
) -> list:
    """Obtain the list of rotation operators from indices.

    Args:
        idx_array (np.ndarray): Indices for obtaining the operators.

    Returns:
        list: Map of rotations.
    """
    if idx_array.ndim == 1:
        return [pauli_rotations[int(ind_pauli)](i) for i, ind_pauli in enumerate(idx_array)]  # type: ignore[abstract]
    return [nested_operator_indexing(sub_array) for sub_array in idx_array]


def rotations_unitary_map(
    idx_array: Tensor, rotation_unitaries_choice: list[Tensor] = UNITARY_TENSOR
) -> Tensor:
    """Obtain the list of unitaries rotation operators from indices.

    Args:
        idx_array (Tensor): Indices for obtaining the unitaries.
        rotation_unitaries_choice (list[Tensor]): Map of indices to unitaries.

    Returns:
        list: Map of local unitaries.
    """
    result = torch.empty(idx_array.size() + (2, 2), dtype=rotation_unitaries_choice[0].dtype)
    for n in range(3):
        mask = idx_array == n
        result[mask] = rotation_unitaries_choice[n]
    return result


def kron_if_non_empty(list_operations: list) -> KronBlock | None:
    """Apply kron to a list of operations."""
    filtered_op: list = list(filter(None, list_operations))
    return kron(*filtered_op) if len(filtered_op) > 0 else None


def extract_operators(unitary_ids: np.ndarray, n_qubits: int) -> list:
    """Sample `shadow_size` rotations of `n_qubits`.

    Args:
        unitary_ids (np.ndarray): Indices for obtaining the operators.
        n_qubits (int): Number of qubits
    Returns:
        list: Pauli strings.
    """
    operations = nested_operator_indexing(unitary_ids)
    if n_qubits > 1:
        operations = [kron_if_non_empty(ops) for ops in operations]
    return operations


def counter_to_freq_vector(
    counter: Counter,
    endianness: Endianness = Endianness.BIG,
) -> Tensor:
    vector_length = 2 ** len(list(counter.keys())[0])
    freq_vector = torch.zeros(vector_length, dtype=torch.float32)
    # Populate frequency vector
    if endianness == Endianness.BIG:
        for bitstring, count in counter.items():
            freq_vector[int("".join(bitstring), 2)] = count
    else:
        for bitstring, count in counter.items():
            freq_vector[int("".join(reversed(bitstring)), 2)] = count
    return freq_vector


def shadow_samples(
    shadow_size: int,
    circuit: QuantumCircuit,
    param_values: dict,
    state: Tensor | None = None,
    backend: Backend | DifferentiableBackend = PyQBackend(),
    noise: NoiseHandler | None = None,
    n_shots: int = 1,
    endianness: Endianness = Endianness.BIG,
) -> MeasurementData:
    """Sample the circuit rotated according to locally sampled pauli unitaries.

    Args:
        shadow_size (int): Number of shadow snapshots.
        circuit (QuantumCircuit): Circuit to rotate.
        param_values (dict): Circuit parameters.
        state (Tensor | None, optional): Input state. Defaults to None.
        backend (Backend | DifferentiableBackend, optional): Backend to run program.
            Defaults to PyQBackend().
        noise (NoiseHandler | None, optional): Noise description. Defaults to None.
        n_shots (int, optional): number of shots per circuit. Defaults to 1.
        endianness (Endianness, optional): Endianness use within program.
            Defaults to Endianness.BIG.

    Returns:
        MeasurementData: A MeasurementData containing
            the pauli indices of local unitaries and measurements.
            0, 1, 2 correspond to X, Y, Z.
    """

    unitary_ids = np.random.randint(0, 3, size=(shadow_size, circuit.n_qubits))
    shadow: list = list()
    all_rotations = extract_operators(unitary_ids, circuit.n_qubits)

    initial_state = state
    if isinstance(backend, PyQBackend) or backend.backend.name == BackendName.PYQTORCH:
        # run the initial circuit without rotations
        # to save computation time
        conv_circ = backend.circuit(circuit)
        initial_state = backend.run(
            circuit=conv_circ,
            param_values=param_values,
            state=state,
            endianness=endianness,
        )
        all_rotations = [
            QuantumCircuit(circuit.n_qubits, rots) if rots else QuantumCircuit(circuit.n_qubits)
            for rots in all_rotations
        ]
    else:
        all_rotations = [
            QuantumCircuit(circuit.n_qubits, circuit.block, rots)
            if rots
            else QuantumCircuit(circuit.n_qubits, circuit.block)
            for rots in all_rotations
        ]

    if noise is not None:
        digital_part = noise.filter(NoiseProtocol.DIGITAL)
        if digital_part is not None:
            all_rotations = [set_noise(rots, digital_part) for rots in all_rotations]

    for i in range(shadow_size):
        # Reverse endianness to get sample bitstrings in ILO.
        conv_circ = backend.circuit(all_rotations[i])
        batch_samples = backend.sample(
            circuit=conv_circ,
            param_values=param_values,
            n_shots=n_shots,
            state=initial_state,
            noise=noise.filter(NoiseProtocol.READOUT) if noise is not None else None,
            endianness=endianness,
        )
        shadow.append(batch_samples)

    bitstrings = list()
    batchsize = len(batch_samples)

    if n_shots == 1:
        for b in range(batchsize):
            bitstrings.append([list(batch[b].keys())[0] for batch in shadow])
        bitstrings_torch = torch.stack(
            [
                torch.stack([torch.tensor([int(b_i) for b_i in sample]) for sample in batch])
                for batch in bitstrings
            ]
        )
    else:
        # return probabilities as data
        for b in range(batchsize):
            bitstrings.append([counter_to_freq_vector(batch[b], endianness) for batch in shadow])
        bitstrings_torch = torch.stack([torch.stack(batch) for batch in bitstrings]) / n_shots
    return MeasurementData(samples=bitstrings_torch, unitaries=torch.tensor(unitary_ids))
