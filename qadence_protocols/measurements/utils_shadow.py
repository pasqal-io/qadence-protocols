from __future__ import annotations

from collections import Counter
from functools import reduce

import numpy as np
import torch
from qadence.backend import Backend
from qadence.backends.pyqtorch import Backend as PyQBackend
from qadence.blocks import AbstractBlock, KronBlock, kron
from qadence.blocks.block_to_tensor import HMAT, IMAT, SDAGMAT
from qadence.blocks.composite import CompositeBlock
from qadence.blocks.primitive import PrimitiveBlock
from qadence.blocks.utils import get_pauli_blocks, unroll_block_with_scaling
from qadence.circuit import QuantumCircuit
from qadence.engines.differentiable_backend import DifferentiableBackend
from qadence.noise import NoiseHandler
from qadence.operations import H, I, SDagger, X, Y, Z
from qadence.transpile.noise import set_noise
from qadence.types import Endianness, NoiseProtocol
from qadence.utils import P0_MATRIX, P1_MATRIX
from torch import Tensor

from qadence_protocols.measurements.utils_tomography import get_qubit_indices_for_op

pauli_gates = [X, Y, Z]
pauli_rotations = [
    lambda index: H(index),
    lambda index: SDagger(index) * H(index),
    lambda index: None,
]

UNITARY_TENSOR = [
    HMAT,
    HMAT @ SDAGMAT,
    IMAT,
]


def identity(n_qubits: int) -> Tensor:
    return torch.eye(2**n_qubits, dtype=torch.complex128)


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
    observables: list[AbstractBlock], accuracy: float, confidence: float
) -> tuple[int, ...]:
    """
    Estimate an optimal shot budget and a shadow partition size.

    to guarantee given accuracy on all observables expectation values
    within 1 - confidence range.

    See https://arxiv.org/pdf/2002.08953.pdf
    Supplementary Material 1 and Eqs. (S23)-(S24).
    """
    max_k = maximal_weight(observables=observables)
    N = round(3**max_k * 34.0 / accuracy**2)
    K = round(2.0 * np.log(2.0 * len(observables) / confidence))
    return N, K


def local_shadow(sample: Counter, unitary_ids: list) -> Tensor:
    """
    Compute local shadow by inverting the quantum channel for each projector state.

    See https://arxiv.org/pdf/2002.08953.pdf
    Supplementary Material 1 and Eqs. (S17,S44).

    Expects a sample bitstring in ILO.
    """
    bitstring = list(sample.keys())[0]
    local_density_matrices = []
    idmat = identity(1)
    for bit, unitary_id in zip(bitstring, unitary_ids):
        proj_mat = P0_MATRIX if bit == "0" else P1_MATRIX
        unitary_tensor = UNITARY_TENSOR[unitary_id].squeeze(dim=0)
        local_density_matrices.append(
            3 * (unitary_tensor.adjoint() @ proj_mat @ unitary_tensor) - idmat
        )
    if len(local_density_matrices) == 1:
        return local_density_matrices[0]
    else:
        return reduce(torch.kron, local_density_matrices)


def robust_local_shadow(
    sample: Counter, unitary_ids: list, calibration: list[float] | Tensor
) -> Tensor:
    """Compute robust shadow by inverting the quantum channel for each projector state."""
    bitstring = list(sample.keys())[0]
    local_density_matrices = []
    idmat = identity(1)
    for bit, unitary_id, corr_coeff in zip(bitstring, unitary_ids, calibration):
        proj_mat = P0_MATRIX if bit == "0" else P1_MATRIX
        unitary_tensor = UNITARY_TENSOR[unitary_id].squeeze(dim=0)
        local_density_matrices.append(
            (1.0 / corr_coeff) * (unitary_tensor.adjoint() @ proj_mat @ unitary_tensor)
            - 0.5 * (1.0 / corr_coeff - 1.0) * idmat
        )
    if len(local_density_matrices) == 1:
        return local_density_matrices[0]
    else:
        return reduce(torch.kron, local_density_matrices)


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


def shadow_samples(
    shadow_size: int,
    circuit: QuantumCircuit,
    param_values: dict,
    state: Tensor | None = None,
    backend: Backend | DifferentiableBackend = PyQBackend(),
    noise: NoiseHandler | None = None,
    endianness: Endianness = Endianness.BIG,
) -> tuple[np.ndarray, Tensor]:
    """Sample the circuit rotated according to locally sampled pauli unitaries.

    Args:
        shadow_size (int): Number of shadow snapshots.
        circuit (QuantumCircuit): Circuit to rotate.
        param_values (dict): Circuit parameters.
        state (Tensor | None, optional): Input state. Defaults to None.
        backend (Backend | DifferentiableBackend, optional): Backend to run program.
            Defaults to PyQBackend().
        noise (NoiseHandler | None, optional): Noise description. Defaults to None.
        endianness (Endianness, optional): Endianness use within program.
            Defaults to Endianness.BIG.

    Returns:
        tuple[np.ndarray, Tensor]: The pauli indices of local unitaries and sampled bitstrings.
            0, 1, 2 correspond to X, Y, Z.
    """

    unitary_ids = np.random.randint(0, 3, size=(shadow_size, circuit.n_qubits))
    shadow: list = list()
    all_rotations = extract_operators(unitary_ids, circuit.n_qubits)
    all_rotations = [
        QuantumCircuit(circuit.n_qubits, rots) if rots else QuantumCircuit(circuit.n_qubits)
        for rots in all_rotations
    ]

    if noise is not None:
        # temporary fix before qadence bump
        if sum([isinstance(proto, NoiseProtocol.DIGITAL) for proto in noise.protocol]) > 0:
            digital_part = noise.filter(NoiseProtocol.DIGITAL)
            all_rotations = [set_noise(rots, digital_part) for rots in all_rotations]
            circuit = set_noise(circuit, digital_part)

    # run the initial circuit without rotations
    conv_circ = backend.circuit(circuit)
    circ_output = backend.run(
        circuit=conv_circ,
        param_values=param_values,
        state=state,
        endianness=endianness,
    )

    for i in range(shadow_size):
        # Reverse endianness to get sample bitstrings in ILO.
        conv_circ = backend.circuit(all_rotations[i])
        batch_samples = backend.sample(
            circuit=conv_circ,
            param_values=param_values,
            n_shots=1,
            state=circ_output,
            noise=noise,
            endianness=endianness,
        )
        shadow.append(batch_samples)

    bitstrings = list()
    batchsize = len(batch_samples)
    for b in range(batchsize):
        bitstrings.append([list(batch[b].keys())[0] for batch in shadow])
    bitstrings_torch = [
        torch.stack([torch.tensor([int(b_i) for b_i in sample]) for sample in batch])
        for batch in bitstrings
    ]
    return unitary_ids, bitstrings_torch


def reconstruct_state(shadow: list) -> Tensor:
    """Reconstruct the state density matrix for the given shadow."""
    return reduce(torch.add, shadow) / len(shadow)


def estimators(
    N: int,
    K: int,
    unitary_shadow_ids: np.ndarray,
    shadow_samples: Tensor,
    observable: AbstractBlock,
    calibration: Tensor | None = None,
) -> Tensor:
    """
    Return trace estimators from the samples for K equally-sized shadow partitions.

    See https://arxiv.org/pdf/2002.08953.pdf
    Algorithm 1.
    """

    obs_qubit_support = list(observable.qubit_support)
    if isinstance(observable, PrimitiveBlock):
        if isinstance(observable, I):
            return torch.tensor(1.0, dtype=torch.get_default_dtype())
        obs_to_pauli_index = [pauli_gates.index(type(observable))]

    elif isinstance(observable, CompositeBlock):
        obs_to_pauli_index = [
            pauli_gates.index(type(p)) for p in observable.blocks if not isinstance(p, I)  # type: ignore[arg-type]
        ]
        ind_I = set(get_qubit_indices_for_op((observable, 1.0), I(0)))
        obs_qubit_support = [ind for ind in observable.qubit_support if ind not in ind_I]

    floor = int(np.floor(N / K))
    traces = []

    if calibration is not None:
        calibration_match = calibration[obs_qubit_support]
    for k in range(K):
        indices_match = np.all(
            unitary_shadow_ids[k * floor : (k + 1) * floor, obs_qubit_support]
            == obs_to_pauli_index,
            axis=1,
        )
        if indices_match.sum() > 0:
            matching_bits = shadow_samples[k * floor : (k + 1) * floor][indices_match][
                :, obs_qubit_support
            ]
            matching_bits = 1.0 - 2.0 * matching_bits
            if calibration is not None:
                matching_bits *= 3.0 * calibration_match

            # recalibrate for robust shadow mainly
            trace = torch.prod(
                matching_bits,
                axis=-1,
            )
            trace = trace.sum() / sum(indices_match)
            traces.append(trace)
        else:
            traces.append(torch.tensor(0.0))
    return torch.tensor(traces, dtype=torch.get_default_dtype())


def expectation_estimations(
    observables: list[AbstractBlock],
    unitaries_ids: np.ndarray,
    batch_shadow_samples: Tensor,
    K: int,
    calibration: Tensor | None = None,
) -> Tensor:
    estimations = []
    N = unitaries_ids.shape[0]

    for observable in observables:
        pauli_decomposition = unroll_block_with_scaling(observable)
        batch_estimations = []
        for batch in batch_shadow_samples:
            pauli_term_estimations = []
            for pauli_term in pauli_decomposition:
                # Get the estimators for the current Pauli term.
                # This is a tensor<float> of size K.
                estimation = estimators(
                    N=N,
                    K=K,
                    unitary_shadow_ids=unitaries_ids,
                    shadow_samples=batch,
                    observable=pauli_term[0],
                    calibration=calibration,
                )
                # Compute the median of means for the current Pauli term.
                # Weigh the median by the Pauli term scaling.
                pauli_term_estimations.append(torch.median(estimation) * pauli_term[1])
            # Sum the expectations for each Pauli term to get the expectation for the
            # current batch.
            batch_estimations.append(sum(pauli_term_estimations))
        estimations.append(batch_estimations)
    return torch.transpose(torch.tensor(estimations, dtype=torch.get_default_dtype()), 1, 0)
