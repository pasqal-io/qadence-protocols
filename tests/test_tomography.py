from __future__ import annotations

import pytest
from typing import List
from torch import allclose, autograd, flatten, manual_seed, ones_like, rand, tensor

from qadence import (
    AbstractBlock,
    QuantumCircuit,
    QuantumModel,
    add,
    chain,
    kron,
)
from qadence.operations import RX, RY, H, SDagger, X, Y, Z
from qadence.blocks.utils import unroll_block_with_scaling

from qadence_protocols.measurements.protocols import Measurements
from qadence_protocols.measurements.utils import (
    empirical_average,
    get_counts,
    get_qubit_indices_for_op,
    iterate_pauli_decomposition,
    rotate,
)


@pytest.mark.parametrize(
    "pauli_word, exp_indices_X, exp_indices_Y",
    [
        (kron(X(0), X(1)), [[0, 1]], [[]]),
        (kron(X(0), Y(1)), [[0]], [[1]]),
        (kron(Y(0), Y(1)), [[]], [[0, 1]]),
        (kron(Z(0), Z(1)), [[]], [[]]),
        (add(X(0), X(1)), [[0], [1]], [[], []]),
        (add(X(0), Y(1)), [[0], []], [[], [1]]),
        (add(Y(0), Y(1)), [[], []], [[0], [1]]),
        (add(Z(0), Z(1)), [[], []], [[], []]),
        (add(kron(X(0), Z(2)), 1.5 * kron(Y(1), Z(2))), [[0], []], [[], [1]]),
        (
            add(
                0.5 * kron(X(0), Y(1), X(2), Y(3)),
                1.5 * kron(Y(0), Z(1), Y(2), Z(3)),
                2.0 * kron(Z(0), X(1), Z(2), X(3)),
            ),
            [[0, 2], [], [1, 3]],
            [[1, 3], [0, 2], []],
        ),
    ],
)
def test_get_qubit_indices_for_op(
    pauli_word: tuple, exp_indices_X: list, exp_indices_Y: list
) -> None:
    pauli_decomposition = unroll_block_with_scaling(pauli_word)

    indices_X = []
    indices_Y = []
    for index, pauli_term in enumerate(pauli_decomposition):
        indices_x = get_qubit_indices_for_op(pauli_term, X(0))
        # if indices_x:
        indices_X.append(indices_x)
        indices_y = get_qubit_indices_for_op(pauli_term, Y(0))
        # if indices_y:
        indices_Y.append(indices_y)
    assert indices_X == exp_indices_X
    assert indices_Y == exp_indices_Y

@pytest.mark.parametrize(
    "circuit, observable, expected_circuit",
    [
        (
            QuantumCircuit(2, kron(X(0), X(1))),
            kron(X(0), Z(2)) + 1.5 * kron(Y(1), Z(2)),
            [
                QuantumCircuit(2, chain(kron(X(0), X(1)), Z(0) * H(0))),
                QuantumCircuit(2, chain(kron(X(0), X(1)), SDagger(1) * H(1))),
            ],
        ),
        (
            QuantumCircuit(4, kron(X(0), X(1), X(2), X(3))),
            add(
                0.5 * kron(X(0), Y(1), X(2), Y(3)),
                1.5 * kron(Y(0), Z(1), Y(2), Z(3)),
                2.0 * kron(Z(0), X(1), Z(2), X(3)),
            ),
            [
                QuantumCircuit(
                    4,
                    chain(
                        kron(X(0), X(1), X(2), X(3)),
                        Z(0) * H(0),
                        Z(2) * H(2),
                        SDagger(1) * H(1),
                        SDagger(3) * H(3),
                    ),
                ),
                QuantumCircuit(
                    4,
                    chain(
                        kron(X(0), X(1), X(2), X(3)),
                        SDagger(0) * H(0),
                        SDagger(2) * H(2),
                    ),
                ),
                QuantumCircuit(
                    4,
                    chain(
                        kron(X(0), X(1), X(2), X(3)),
                        Z(1) * H(1),
                        Z(3) * H(3),
                    ),
                ),
            ],
        ),
    ],
)
def test_rotate(
    circuit: QuantumCircuit,
    observable: AbstractBlock,
    expected_circuit: List[QuantumCircuit],
) -> None:
    pauli_decomposition = unroll_block_with_scaling(observable)
    for index, pauli_term in enumerate(pauli_decomposition):
        rotated_circuit = rotate(circuit, pauli_term)
        assert rotated_circuit == expected_circuit[index]