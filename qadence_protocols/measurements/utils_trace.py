from __future__ import annotations

from math import log2
from string import ascii_letters as ABC

from numpy import argsort, array
from numpy.typing import NDArray
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
    operator = operator.view([2] * 2 * n_qubits)

    perm = list(tuple(ranked_support) + tuple(ranked_support + n_qubits))

    if inv:
        perm = argsort(perm).tolist()

    return operator.permute(perm).reshape([2**n_qubits, 2**n_qubits])


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

    n_qubits = int(log2(state.size()[0]))
    n_support = len(qubit_support)
    full_support = tuple(range(n_qubits))
    support_perm = tuple(sorted(qubit_support)) + tuple(set(full_support) - set(qubit_support))
    state = permute_basis(state, support_perm)

    state = state.reshape([2**n_support, (2 ** (2 * n_qubits - n_support))])
    state = einsum("ij,jk->ik", operator, state).reshape([2**n_qubits, 2**n_qubits])
    return permute_basis(state, support_perm, inv=True)
