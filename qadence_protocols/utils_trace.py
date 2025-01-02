from __future__ import annotations

from math import log2
from string import ascii_letters as ABC

import torch
from numpy import argsort, array
from numpy.typing import NDArray
from pyqtorch.utils import dm_partial_trace
from qadence import block_to_tensor
from qadence.blocks.abstract import AbstractBlock
from torch import Tensor, einsum

ABC_ARRAY: NDArray = array(list(ABC))


def permute_basis(operator: Tensor, qubit_support: tuple, inv: bool = False) -> Tensor:
    """Takes an operator tensor and permutes the rows and.

    columns according to the order of the qubit support.

    Args:
        operator (Tensor): Operator to permute over.
        qubit_support (tuple): Qubit support.
        inv (bool): Applies the inverse permutation instead.

    Returns:
        Tensor: Permuted operator.
    """
    ordered_support = argsort(qubit_support)
    ranked_support = argsort(ordered_support)
    n_qubits = len(qubit_support)
    if all(a == b for a, b in zip(ranked_support, list(range(n_qubits)))):
        return operator
    batchsize = operator.size()[0]
    operator = operator.view([batchsize] + [2] * 2 * n_qubits)

    perm = list(tuple(ranked_support) + tuple(ranked_support + n_qubits))

    if inv:
        perm = argsort(perm).tolist()
    perm = [0] + [i + 1 for i in perm]

    return operator.permute(perm).reshape([batchsize, 2**n_qubits, 2**n_qubits])


def apply_operator_dm(
    state: Tensor,
    operator: Tensor,
    qubit_support: tuple[int, ...] | list[int],
) -> Tensor:
    """
    Apply an operator to a density matrix on a given qubit suport, i.e., compute:

    OP.DM.OP.dagger()

    Args:
        state: State to operate on.
        operator: Tensor to contract over 'state'.
        qubit_support: Tuple of qubits on which to apply the 'operator' to.

    Returns:
        DensityMatrix: The resulting density matrix after applying the operator.
    """

    batchsize = state.size()[0]
    n_qubits = int(log2(state.size()[1]))
    n_support = len(qubit_support)
    full_support = tuple(range(n_qubits))
    support_perm = tuple(sorted(qubit_support)) + tuple(set(full_support) - set(qubit_support))
    state = permute_basis(state, support_perm)
    state = state.reshape([batchsize, 2**n_support, (2 ** (2 * n_qubits - n_support))])
    state = einsum("ij,bjk->bik", operator, state).reshape(
        [batchsize, 2**n_qubits, 2**n_qubits]
    )
    return permute_basis(state, support_perm, inv=True)


def expectation_trace(state: Tensor, observables: list[AbstractBlock]) -> Tensor:
    """Calculate the expectation using the trace operator.

    Args:
        state (Tensor): Input states as density matrices.
        observables (list[AbstractBlock]): List of observables to calculate expectations.

    Returns:
        Tensor: The expectations.
    """

    if not isinstance(observables, list):
        observables = [observables]
    tr_obs_rho = [
        apply_operator_dm(
            state,
            block_to_tensor(obs, use_full_support=False).squeeze(0),
            qubit_support=obs.qubit_support,
        )
        for obs in observables
    ]
    vmap_trace = torch.vmap(torch.trace)
    tr_obs_rho = [vmap_trace(res_dm_obs).real for res_dm_obs in tr_obs_rho]
    return torch.stack(tr_obs_rho, axis=1)


def apply_partial_trace(rho: Tensor, keep_indices: list[int]) -> Tensor:
    """
    Computes the partial trace of a density matrix for a system of several qubits with batch size.

    This function also permutes the qubits according to the order specified in keep_indices.

    Args:
        rho (Tensor) : Density matrix of shape [batch_size, 2**n_qubits, 2**n_qubits].
        keep_indices (list[int]): Index of the qubit subsystems to keep.

    Returns:
        Tensor: Reduced density matrix after the partial trace,
        of shape [batch_size, 2**n_keep, 2**n_keep].
    """
    return dm_partial_trace(rho.permute((1, 2, 0)), keep_indices).permute((0, 1, 2))


def compute_purity(rho: Tensor) -> Tensor:
    """Compute the purity of a density matrix.

    Args:
        rho (Tensor): Density matrix.

    Returns:
        Tensor: Tr[rho ** 2]
    """
    return torch.trace(rho**2).real
