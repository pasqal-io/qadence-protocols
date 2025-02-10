from __future__ import annotations

from collections import Counter
from typing import Callable

import pytest
import strategies as st
from hypothesis import given, settings
from qadence import (
    AbstractBlock,
    PrimitiveBlock,
    QuantumCircuit,
    QuantumModel,
    add,
    chain,
    kron,
)
from qadence.blocks.utils import unroll_block_with_scaling
from qadence.ml_tools.utils import rand_featureparameters
from qadence.operations import H, I, SDagger, X, Y, Z
from qadence.types import BackendName, DiffMode
from torch import allclose, tensor

from qadence_protocols import Measurements
from qadence_protocols.measurements.utils_tomography import (
    empirical_average,
    get_counts,
    get_qubit_indices_for_op,
    rotate,
)
from qadence_protocols.types import MeasurementProtocol


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
        indices_x = get_qubit_indices_for_op(pauli_term, X)
        # if indices_x:
        indices_X.append(indices_x)
        indices_y = get_qubit_indices_for_op(pauli_term, Y)
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
                QuantumCircuit(2, chain(kron(X(0), X(1)), I(0), H(0))),
                QuantumCircuit(2, chain(kron(X(0), X(1)), SDagger(1), H(1))),
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
                        I(0),
                        H(0),
                        I(2),
                        H(2),
                        SDagger(1),
                        H(1),
                        SDagger(3),
                        H(3),
                    ),
                ),
                QuantumCircuit(
                    4,
                    chain(
                        kron(X(0), X(1), X(2), X(3)),
                        SDagger(0),
                        H(0),
                        SDagger(2),
                        H(2),
                    ),
                ),
                QuantumCircuit(
                    4,
                    chain(
                        kron(X(0), X(1), X(2), X(3)),
                        I(1),
                        H(1),
                        I(3),
                        H(3),
                    ),
                ),
            ],
        ),
    ],
)
def test_rotate(
    circuit: QuantumCircuit,
    observable: AbstractBlock,
    expected_circuit: list[QuantumCircuit],
) -> None:
    pauli_decomposition = unroll_block_with_scaling(observable)
    for index, pauli_term in enumerate(pauli_decomposition):
        rotated_circuit = rotate(circuit, pauli_term)
        assert rotated_circuit == expected_circuit[index]


def test_get_counts() -> None:
    samples = [Counter({"00": 10, "01": 50, "10": 20, "11": 20})]
    support = [0]
    counts = get_counts(samples, support)
    assert counts == [Counter({"0": 60, "1": 40})]
    support = [1]
    counts = get_counts(samples, support)
    assert counts == [Counter({"0": 30, "1": 70})]
    support = [0, 1]
    counts = get_counts(samples, support)
    assert counts == samples

    samples = [
        Counter(
            {
                "1111": 1653,
                "0000": 1586,
                "0001": 1463,
                "0110": 1286,
                "1110": 998,
                "0101": 668,
                "0111": 385,
                "1000": 327,
                "0011": 322,
                "1100": 281,
                "1001": 218,
                "1010": 213,
                "0100": 187,
                "1101": 172,
                "1011": 154,
                "0010": 87,
            }
        )
    ]
    support = [0, 1, 2, 3]
    counts = get_counts(samples, support)
    assert counts == samples


def test_empirical_average() -> None:
    samples = [Counter({"00": 10, "01": 50, "10": 20, "11": 20})]
    support = [0]
    assert allclose(empirical_average(samples, support), tensor([0.2]))
    support = [1]
    assert allclose(empirical_average(samples, support), tensor([-0.4]))
    support = [0, 1]
    assert allclose(empirical_average(samples, support), tensor([-0.4]))
    samples = [
        Counter(
            {
                "1111": 1653,
                "0000": 1586,
                "0001": 1463,
                "0110": 1286,
                "1110": 998,
                "0101": 668,
                "0111": 385,
                "1000": 327,
                "0011": 322,
                "1100": 281,
                "1001": 218,
                "1010": 213,
                "0100": 187,
                "1101": 172,
                "1011": 154,
                "0010": 87,
            }
        )
    ]
    support = [0, 1, 2, 3]
    assert allclose(empirical_average(samples, support), tensor([0.2454]))


@pytest.mark.parametrize("obs_base_op", [X, Z])
@pytest.mark.parametrize("obs_composition", [add, kron])
@given(st.digital_circuits())
@settings(deadline=None)
def test_tomography(
    obs_base_op: AbstractBlock,
    obs_composition: Callable,
    circuit: QuantumCircuit,
) -> None:
    inputs = rand_featureparameters(circuit, 1)
    observable = (
        obs_composition(obs_base_op(0), obs_base_op(1)) if circuit.n_qubits > 1 else obs_base_op(0)
    )
    backend = BackendName.PYQTORCH

    model = QuantumModel(circuit=circuit, observable=observable, backend=backend)
    expectation_analytical = model.expectation(inputs)

    tomo_measurement = Measurements(
        protocol=MeasurementProtocol.TOMOGRAPHY,
        options={"n_shots": 10000},
    )
    expectation_sampled = tomo_measurement(model, param_values=inputs)

    tomo_measurement_more_shots = Measurements(
        protocol=MeasurementProtocol.TOMOGRAPHY,
        options={"n_shots": 1000000},
    )
    expectation_sampled_more_shots = tomo_measurement_more_shots(model, param_values=inputs)

    assert allclose(expectation_sampled, expectation_analytical, atol=1.0e-01)
    assert allclose(expectation_sampled_more_shots, expectation_analytical, atol=1.0e-02)


@pytest.mark.parametrize("batchsize_values", [1, 2])
@pytest.mark.parametrize("base_op", [X, Y, Z])
@pytest.mark.parametrize("do_kron", [True, False])
@given(st.digital_circuits())
@settings(deadline=None)
def test_basic_tomography_for_parametric_circuit_forward_pass(
    batchsize_values: int, base_op: PrimitiveBlock, do_kron: bool, circuit: QuantumCircuit
) -> None:
    values = rand_featureparameters(circuit, batchsize_values)
    observable = base_op(0) ^ circuit.n_qubits if do_kron else base_op(min(1, circuit.n_qubits - 1))  # type: ignore[operator]
    model = QuantumModel(
        circuit=circuit,
        observable=observable,
        backend=BackendName.PYQTORCH,
        diff_mode=DiffMode.GPSR,
    )
    analytical_result = model.expectation(values)
    tomo = Measurements(
        protocol=MeasurementProtocol.TOMOGRAPHY,
        options={"n_shots": 100000},
    )
    estimated_values = tomo(
        model=model,
        param_values=values,
    )
    assert allclose(estimated_values, analytical_result, atol=0.01)


def test_tomography_raise_errors() -> None:
    with pytest.raises(KeyError):
        tomo_measurement = Measurements(
            protocol=MeasurementProtocol.TOMOGRAPHY,
            options={"nsamples": 10000},
        )
