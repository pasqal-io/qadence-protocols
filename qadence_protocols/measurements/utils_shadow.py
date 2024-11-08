from __future__ import annotations

from collections import Counter
from functools import partial, reduce

import numpy as np
import torch
from qadence.backend import Backend
from qadence.backends.pyqtorch import Backend as PyQBackend
from qadence.blocks import AbstractBlock, chain, kron
from qadence.blocks.block_to_tensor import HMAT, IMAT, SDAGMAT, ZMAT, block_to_tensor
from qadence.blocks.composite import CompositeBlock
from qadence.blocks.primitive import PrimitiveBlock
from qadence.blocks.utils import get_pauli_blocks, unroll_block_with_scaling
from qadence.circuit import QuantumCircuit
from qadence.engines.differentiable_backend import DifferentiableBackend
from qadence.noise import Noise
from qadence.operations import X, Y, Z
from qadence.types import Endianness
from qadence.utils import P0_MATRIX, P1_MATRIX
from qadence_protocols.measurements.utils_trace import apply_operator_dm
from qadence_protocols.measurements.utils_tomography import rotate
from torch import Tensor

pauli_gates = [X, Y, Z]


UNITARY_TENSOR = [
    HMAT,
    SDAGMAT.squeeze(dim=0) @ HMAT,
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

    This is to guarantee given accuracy on all observables expectation values
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

def nested_operator_indexing(idx_array: np.ndarray) -> list[list[AbstractBlock]]:
    """Obtain the list of operators from indices.

    Args:
        idx_array (np.ndarray): Indices for obtaining the operators.

    Returns:
        list[list[AbstractBlock]]: Map of pauli operators.
    """
    if idx_array.ndim == 1:
        return [pauli_gates[int(ind_pauli)](i) for i, ind_pauli in enumerate(idx_array)]
    return [nested_operator_indexing(sub_array) for sub_array in idx_array]

def extract_unitaries(unitary_ids: np.ndarray, n_qubits: int) -> list[AbstractBlock]:
    """Sample `shadow_size` pauli strings of `n_qubits`.

    Args:
        unitary_ids (np.ndarray): Indices for obtaining the operators.
        n_qubits (int): Number of qubits

    Returns:
        list[AbstractBlock]: Pauli strings.
    """
    unitaries = nested_operator_indexing(unitary_ids)
    if n_qubits > 1:
        unitaries = [kron(*l) for l in unitaries]
    return unitaries

def classical_shadow(
    shadow_size: int,
    circuit: QuantumCircuit,
    param_values: dict,
    state: Tensor | None = None,
    backend: Backend | DifferentiableBackend = PyQBackend(),
    noise: Noise | None = None,
    endianness: Endianness = Endianness.BIG,
    robust_shadow: bool = False,
    calibration: list[float] | Tensor | None = None,
) -> list:
    shadow: list = []
    shadow_caller = local_shadow
    if robust_shadow:
        shadow_caller = partial(robust_local_shadow, calibration=calibration)
    
    unitary_ids = np.random.randint(0, 3, size=(shadow_size, circuit.n_qubits))
    all_unitaries = extract_unitaries(unitary_ids, circuit.n_qubits)
    # TODO: Parallelize embarrassingly parallel loop.
    for i in range(shadow_size):
        random_unitary_block = all_unitaries[i]
        rotated_circuit = rotate(circuit, random_unitary_block)
        # Reverse endianness to get sample bitstrings in ILO.
        conv_circ = backend.circuit(rotated_circuit)
        samples = backend.sample(
            circuit=conv_circ,
            param_values=param_values,
            n_shots=1,
            state=state,
            noise=noise,
            endianness=endianness,
        )
        batched_shadow = [shadow_caller(sample=batch, unitary_ids=unitary_ids[i]) for batch in samples]
        shadow.append(batched_shadow)

    return torch.stack([torch.stack(s) for s in zip(*shadow)]), unitary_ids


def reconstruct_state(shadow: list) -> Tensor:
    """Reconstruct the state density matrix for the given shadow."""
    return reduce(torch.add, shadow) / len(shadow)


def compute_traces(
    N: int,
    K: int,
    shadow: list,
    unitary_shadow_ids: list,
    observable: AbstractBlock,
    endianness: Endianness = Endianness.BIG,
) -> list:
    floor = int(np.floor(N / K))
    traces = []

    # if isinstance(observable, PrimitiveBlock):
    #     obs_to_pauli_index = [pauli_gates.index(type(observable))]
    # else:
    #     obs_to_pauli_index = [pauli_gates.index(type(p)) for p in observable.blocks]
    obs_qubit_support = observable.qubit_support
    obs_matrix = block_to_tensor(observable, endianness=endianness, qubit_support=obs_qubit_support).squeeze(dim=0)
    # TODO: Parallelize embarrassingly parallel loop.
    for k in range(K):
        # indices_match = np.all(unitary_shadow_ids[k * floor : (k + 1) * floor, obs_qubit_support] == obs_to_pauli_index, axis=1)
        # if indices_match.sum() > 0:
        #     reconstructed_state = reconstruct_state(shadow=shadow[k * floor : (k + 1) * floor][indices_match])
        #     # Please note the endianness is also flipped to get results in LE.
        #     trace = apply_operator_dm(reconstructed_state, obs_matrix, qubit_support=obs_qubit_support).trace().real
        #     traces.append(trace)
        # else:
        #     traces.append(torch.tensor(0.0))
        
        
        reconstructed_state = reconstruct_state(shadow=shadow[k * floor : (k + 1) * floor])
        # Please note the endianness is also flipped to get results in LE.
        # trace = apply_operator_dm(reconstructed_state, obs_matrix, qubit_support=obs_qubit_support).trace().real
        trace = apply_operator_dm(reconstructed_state, obs_matrix, qubit_support=obs_qubit_support).trace().real
        traces.append(trace)
    return traces


def estimators(
    N: int,
    K: int,
    shadow: list,
    unitary_shadow_ids: list,
    observable: AbstractBlock,
    endianness: Endianness = Endianness.BIG,
) -> Tensor:
    """
    Return estimators (traces of observable times mean density matrix).

    Estimators are computed for K equally-sized shadow partitions.

    See https://arxiv.org/pdf/2002.08953.pdf
    Algorithm 1.
    """
    
    traces = compute_traces(
        N=N,
        K=K,
        shadow=shadow,
        unitary_shadow_ids=unitary_shadow_ids,
        observable=observable,
        endianness=endianness,
    )
    return torch.tensor(traces, dtype=torch.get_default_dtype())


def estimations(
    circuit: QuantumCircuit,
    observables: list[AbstractBlock],
    param_values: dict,
    shadow_size: int | None = None,
    accuracy: float = 0.1,
    confidence_or_groups: float | int = 0.1,
    state: Tensor | None = None,
    backend: Backend | DifferentiableBackend = PyQBackend(),
    noise: Noise | None = None,
    endianness: Endianness = Endianness.BIG,
    return_shadows: bool = False,
    robust_shadow: bool = False,
    calibration: list[float] | Tensor | None = None,
) -> Tensor:
    """Compute expectation values for all local observables using median of means."""
    # N is the estimated shot budget for the classical shadow to
    # achieve desired accuracy for all L = len(observables) within 1 - confidence probablity.
    # K is the size of the shadow partition.
    if robust_shadow:
        K = int(confidence_or_groups)
    else:
        N, K = number_of_samples(
            observables=observables, accuracy=accuracy, confidence=confidence_or_groups
        )
    if shadow_size is not None:
        N = shadow_size
    shadow, unitaries_id = classical_shadow(
        shadow_size=N,
        circuit=circuit,
        param_values=param_values,
        state=state,
        backend=backend,
        noise=noise,
        endianness=endianness,
        robust_shadow=robust_shadow,
        calibration=calibration,
    )
    if return_shadows:
        return shadow, unitaries_id

    estimations = []
    for observable in observables:
        pauli_decomposition = unroll_block_with_scaling(observable)
        batch_estimations = []
        for batch in shadow:
            pauli_term_estimations = []
            for pauli_term in pauli_decomposition:
                # Get the estimators for the current Pauli term.
                # This is a tensor<float> of size K.
                estimation = estimators(
                    N=N,
                    K=K,
                    shadow=batch,
                    unitary_shadow_ids=unitaries_id,
                    observable=pauli_term[0],
                    endianness=endianness,
                )
                # Compute the median of means for the current Pauli term.
                # Weigh the median by the Pauli term scaling.
                pauli_term_estimations.append(torch.median(estimation) * pauli_term[1])
            # Sum the expectations for each Pauli term to get the expectation for the
            # current batch.
            batch_estimations.append(sum(pauli_term_estimations))
        estimations.append(batch_estimations)
    return torch.transpose(torch.tensor(estimations, dtype=torch.get_default_dtype()), 1, 0)
