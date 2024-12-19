from __future__ import annotations

from functools import partial, reduce
from typing import Callable

import numpy as np
import torch
from qadence.blocks import AbstractBlock
from qadence.blocks.composite import CompositeBlock
from qadence.blocks.primitive import PrimitiveBlock
from qadence.blocks.utils import unroll_block_with_scaling
from qadence.operations import I
from qadence.utils import P0_MATRIX, P1_MATRIX
from torch import Tensor

from qadence_protocols.measurements.utils_shadow.data_acquisition import (
    batch_kron,
    rotations_unitary_map,
)
from qadence_protocols.measurements.utils_shadow.unitaries import (
    UNITARY_TENSOR_ADJOINT,
    hamming_one_qubit,
    idmat,
    pauli_gates,
)
from qadence_protocols.measurements.utils_tomography import get_qubit_indices_for_op
from qadence_protocols.utils_trace import expectation_trace

einsum_alphabet = "abcdefghijklmnopqsrtuvwxyz"
einsum_alphabet_cap = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def get_unitaries_and_projectors(bitstrings: Tensor, unitary_ids: Tensor) -> tuple:
    """Obtain unitaries, projector matrices and adjoint unitaries for shadow computations."""
    nested_unitaries = rotations_unitary_map(unitary_ids)
    nested_unitaries_adjoint = rotations_unitary_map(unitary_ids, UNITARY_TENSOR_ADJOINT)
    projmat = torch.empty(nested_unitaries.shape, dtype=nested_unitaries.dtype)
    projmat[..., :, :] = torch.where(
        bitstrings.bool().unsqueeze(-1).unsqueeze(-1), P1_MATRIX, P0_MATRIX
    )
    return (nested_unitaries, projmat, nested_unitaries_adjoint)


def local_shadow(bitstrings: Tensor, unitary_ids: Tensor) -> Tensor:
    """
    Compute local shadow by inverting the quantum channel for each projector state.

    See https://arxiv.org/pdf/2002.08953.pdf
    Supplementary Material 1 and Eqs. (S17,S44).

    Expects a sample bitstring in ILO.
    """

    nested_unitaries, projmat, nested_unitaries_adjoint = get_unitaries_and_projectors(
        bitstrings, unitary_ids
    )
    local_densities = 3.0 * (nested_unitaries_adjoint @ projmat @ nested_unitaries) - idmat
    return local_densities


def robust_local_shadow(bitstrings: Tensor, unitary_ids: Tensor, calibration: Tensor) -> Tensor:
    """Compute robust local shadow by inverting the quantum channel for each projector state."""
    nested_unitaries, projmat, nested_unitaries_adjoint = get_unitaries_and_projectors(
        bitstrings, unitary_ids
    )
    idmatcal = torch.stack([idmat * 0.5 * (1.0 / corr_coeff - 1.0) for corr_coeff in calibration])
    local_densities = (1.0 / calibration.unsqueeze(-1).unsqueeze(-1)) * (
        nested_unitaries_adjoint @ projmat @ nested_unitaries
    ) - idmatcal
    return local_densities


def global_shadow_hamming(probas: Tensor, unitary_ids: Tensor) -> Tensor:
    """Compute global shadow using a Hamming matrix."""

    nested_unitaries = rotations_unitary_map(unitary_ids)
    nested_unitaries_adjoint = rotations_unitary_map(unitary_ids, UNITARY_TENSOR_ADJOINT)

    N = unitary_ids.shape[1]
    if N > 1:
        nested_unitaries = batch_kron(nested_unitaries)
        nested_unitaries_adjoint = batch_kron(nested_unitaries_adjoint)
    hamming_mat = [hamming_one_qubit.to(dtype=probas.dtype) for i in range(N)]
    d = 2**N
    probas = probas.reshape((probas.shape[0],) + (2,) * N)

    ein_command = einsum_alphabet[: N + 1]
    for i in range(1, N + 1):
        ein_command += "," + einsum_alphabet[i] + einsum_alphabet_cap[i]
    ein_command += "->" + einsum_alphabet[0] + einsum_alphabet_cap[1 : N + 1]
    probprime = d * torch.einsum(ein_command, *([probas] + hamming_mat)).to(
        dtype=nested_unitaries.dtype
    )
    probprime = torch.diag_embed(probprime.reshape((-1, d)))
    densities = nested_unitaries_adjoint @ probprime @ nested_unitaries
    return densities


def global_robust_shadow_hamming(
    probas: Tensor, unitary_ids: Tensor, calibration: Tensor
) -> Tensor:
    """Compute robust global shadow using a Hamming matrix."""

    nested_unitaries = rotations_unitary_map(unitary_ids)
    nested_unitaries_adjoint = rotations_unitary_map(unitary_ids, UNITARY_TENSOR_ADJOINT)

    N = unitary_ids.shape[1]
    if N > 1:
        nested_unitaries = batch_kron(nested_unitaries)
        nested_unitaries_adjoint = batch_kron(nested_unitaries_adjoint)

    calibration = 3.0 * calibration
    div = 2.0 * calibration - 1.0
    alpha = 3.0 / div
    beta = (calibration - 2.0) / div

    # shape (N, 2, 2)
    hamming_mat = 0.5 * (
        torch.stack((torch.stack((alpha + beta, beta)), torch.stack((beta, alpha + beta))))
        .permute((-1, 0, 1))
        .to(dtype=probas.dtype)
    )
    hamming_mat = [hamming_mat[i, ...] for i in range(N)]
    d = 2**N
    probas = probas.reshape((probas.shape[0],) + (2,) * N)

    ein_command = einsum_alphabet[: N + 1]
    for i in range(1, N + 1):
        ein_command += "," + einsum_alphabet[i] + einsum_alphabet_cap[i]
    ein_command += "->" + einsum_alphabet[0] + einsum_alphabet_cap[1 : N + 1]
    probprime = d * torch.einsum(ein_command, *([probas] + hamming_mat)).to(
        dtype=nested_unitaries.dtype
    )
    probprime = torch.diag_embed(probprime.reshape((-1, d)))
    densities = nested_unitaries_adjoint @ probprime @ nested_unitaries
    return densities


def compute_snapshots(
    bitstrings: Tensor, unitaries_ids: Tensor, shadow_caller: Callable, local_shadows: bool = True
) -> Tensor:
    snapshots: list = list()

    if local_shadows and unitaries_ids.shape[1] > 1:

        def obtain_global_shadow(bits: Tensor, unit_ids: Tensor) -> Tensor:
            return batch_kron(shadow_caller(bits, unit_ids))

    else:

        def obtain_global_shadow(bits: Tensor, unit_ids: Tensor) -> Tensor:
            return shadow_caller(bits, unit_ids)

    for batch_bitstrings in bitstrings:
        snapshots.append(obtain_global_shadow(batch_bitstrings, unitaries_ids))
    return torch.stack(snapshots)


def reconstruct_state(shadow: list) -> Tensor:
    """Reconstruct the state density matrix for the given shadow."""
    return reduce(torch.add, shadow) / len(shadow)


def estimators_from_bitstrings(
    N: int,
    K: int,
    unitary_shadow_ids: Tensor,
    shadow_samples: Tensor,
    observable: AbstractBlock,
    calibration: Tensor | None = None,
) -> Tensor:
    """
    Return trace estimators from the samples for K equally-sized shadow partitions.

    See https://arxiv.org/pdf/2002.08953.pdf
    Algorithm 1.

    Note that this is for the case where the number of shots per unitary is 1.
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

    obs_to_pauli_index = torch.tensor(obs_to_pauli_index)
    for k in range(K):
        indices_match = torch.all(
            unitary_shadow_ids[k * floor : (k + 1) * floor, obs_qubit_support]
            == obs_to_pauli_index,
            axis=1,
        )
        if indices_match.sum() > 0:
            matching_bits = shadow_samples[k * floor : (k + 1) * floor][indices_match][
                :, obs_qubit_support
            ]
            matching_bits = 1.0 - 2.0 * matching_bits

            # recalibrate for robust shadow mainly
            if calibration is not None:
                matching_bits *= 3.0 * calibration_match

            trace = torch.prod(
                matching_bits,
                axis=-1,
            )
            trace = trace.sum() / indices_match.sum()
            traces.append(trace)
        else:
            traces.append(torch.tensor(0.0))
    return torch.tensor(traces, dtype=torch.get_default_dtype())


def estimators_from_probas(
    N: int,
    K: int,
    unitary_shadow_ids: Tensor,
    shadow_samples: Tensor,
    observable: AbstractBlock,
    calibration: Tensor | None = None,
) -> Tensor:
    """
    Return trace estimators from the samples for K equally-sized shadow partitions.

    See https://arxiv.org/pdf/2002.08953.pdf
    Algorithm 1.

    Note that this is for the case where the number of shots per unitary is more than 1.
    """

    floor = int(np.floor(N / K))
    traces = []

    shadow_caller: Callable = global_shadow_hamming
    if calibration is not None:
        shadow_caller = partial(global_robust_shadow_hamming, calibration=calibration)

    for k in range(K):
        snapshots = shadow_caller(
            shadow_samples[k * floor : (k + 1) * floor],
            unitary_shadow_ids[k * floor : (k + 1) * floor],
        )
        reconstructed_state = snapshots.sum(axis=0) / snapshots.shape[0]

        trace = expectation_trace(reconstructed_state.unsqueeze(0), [observable])[0]
        traces.append(trace)
    return torch.tensor(traces, dtype=torch.get_default_dtype())


def expectation_estimations(
    observables: list[AbstractBlock],
    unitaries_ids: np.ndarray,
    batch_shadow_samples: Tensor,
    K: int,
    calibration: Tensor | None = None,
    n_shots: int = 1,
) -> Tensor:
    estimations = []
    N = unitaries_ids.shape[0]

    estimator_fct = estimators_from_bitstrings if n_shots == 1 else estimators_from_probas

    for observable in observables:
        pauli_decomposition = unroll_block_with_scaling(observable)
        batch_estimations = []
        for batch in batch_shadow_samples:
            pauli_term_estimations = []
            for pauli_term in pauli_decomposition:
                # Get the estimators for the current Pauli term.
                # This is a tensor<float> of size K.
                estimation = estimator_fct(
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
