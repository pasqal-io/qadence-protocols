from __future__ import annotations

from collections import Counter

import pytest
import torch
from qadence.backends.api import backend_factory
from qadence.blocks.abstract import AbstractBlock
from qadence.blocks.block_to_tensor import IMAT
from qadence.blocks.utils import add, chain, kron
from qadence.circuit import QuantumCircuit
from qadence.constructors import ising_hamiltonian, total_magnetization
from qadence.model import QuantumModel
from qadence.operations import RX, RY, H, I, X, Y, Z
from qadence.parameters import Parameter
from qadence.types import BackendName, DiffMode
from qadence.utils import P0_MATRIX, P1_MATRIX
from torch import Tensor

from qadence_protocols import Measurements
from qadence_protocols.measurements.utils_shadow import (
    UNITARY_TENSOR,
    _max_observable_weight,
    local_shadow,
    number_of_samples,
)
from qadence_protocols.types import MeasurementProtocols


@pytest.mark.parametrize(
    "observable, exp_weight",
    [
        (X(0), 1),
        (kron(*[X(0), Y(1), Z(2)]), 3),
        (add(*[X(0), Y(0), Z(0)]), 1),
        (kron(*[X(0), H(1), I(2), Z(3)]), 2),
        (total_magnetization(5), 1),
        (ising_hamiltonian(4), 2),
    ],
)
def test_weight(observable: AbstractBlock, exp_weight: int) -> None:
    qubit_weight = _max_observable_weight(observable)
    assert qubit_weight == exp_weight


@pytest.mark.parametrize(
    "observables, accuracy, confidence, exp_samples",
    [([total_magnetization(2)], 0.1, 0.1, (10200, 6))],
)
def test_number_of_samples(
    observables: list[AbstractBlock], accuracy: float, confidence: float, exp_samples: tuple
) -> None:
    N, K = number_of_samples(observables=observables, accuracy=accuracy, confidence=confidence)
    assert N == exp_samples[0]
    assert K == exp_samples[1]


@pytest.mark.parametrize(
    "sample, unitary_ids, exp_shadow",
    [
        (
            Counter({"10": 1}),
            [0, 2],
            torch.kron(
                3 * (UNITARY_TENSOR[0].adjoint() @ P1_MATRIX @ UNITARY_TENSOR[0]) - IMAT,
                3 * (UNITARY_TENSOR[2].adjoint() @ P0_MATRIX @ UNITARY_TENSOR[2]) - IMAT,
            ),
        ),
        (
            Counter({"0111": 1}),
            [2, 0, 2, 2],
            torch.kron(
                torch.kron(
                    3 * (UNITARY_TENSOR[2].adjoint() @ P0_MATRIX @ UNITARY_TENSOR[2]) - IMAT,
                    3 * (UNITARY_TENSOR[0].adjoint() @ P1_MATRIX @ UNITARY_TENSOR[0]) - IMAT,
                ),
                torch.kron(
                    3 * (UNITARY_TENSOR[2].adjoint() @ P1_MATRIX @ UNITARY_TENSOR[2]) - IMAT,
                    3 * (UNITARY_TENSOR[2].adjoint() @ P1_MATRIX @ UNITARY_TENSOR[2]) - IMAT,
                ),
            ),
        ),
    ],
)
def test_local_shadow(sample: Counter, unitary_ids: list, exp_shadow: Tensor) -> None:
    shadow = local_shadow(sample=sample, unitary_ids=unitary_ids)
    assert torch.allclose(shadow, exp_shadow)


theta = Parameter("theta")


# @pytest.mark.flaky(max_runs=5)
# @pytest.mark.parametrize(
#     "circuit, observable, values",
#     [
#         (QuantumCircuit(2, kron(X(0), X(1))), X(0) @ X(1), {}),
#         (QuantumCircuit(2, kron(X(0), X(1))), X(0) @ Y(1), {}),
#         (QuantumCircuit(2, kron(X(0), X(1))), Y(0) @ X(1), {}),
#         (QuantumCircuit(2, kron(X(0), X(1))), Y(0) @ Y(1), {}),
#         (QuantumCircuit(2, kron(Z(0), H(1))), X(0) @ Z(1), {}),
#         (
#             QuantumCircuit(2, kron(RX(0, theta), X(1))),
#             kron(Z(0), Z(1)),
#             {"theta": torch.tensor([0.5, 1.0])},
#         ),
#         (QuantumCircuit(2, kron(X(0), Z(1))), ising_hamiltonian(2), {}),
#     ],
# )
# def test_estimations_comparison_exact(
#     circuit: QuantumCircuit, observable: AbstractBlock, values: dict
# ) -> None:
#     backend = backend_factory(backend=BackendName.PYQTORCH, diff_mode=DiffMode.GPSR)
#     (conv_circ, _, embed, params) = backend.convert(circuit=circuit, observable=observable)
#     param_values = embed(params, values)

#     estimated_exp = expectation_estimations(
#         circuit=conv_circ.abstract,
#         observables=[observable],
#         param_values=param_values,
#         shadow_size=5000,
#     )
#     exact_exp = expectation(circuit, observable, values=values)
#     assert torch.allclose(estimated_exp, exact_exp, atol=0.2)


theta1 = Parameter("theta1", trainable=False)
theta2 = Parameter("theta2", trainable=False)
theta3 = Parameter("theta3", trainable=False)
theta4 = Parameter("theta4", trainable=False)


blocks = chain(
    kron(RX(0, theta1), RY(1, theta2)),
    kron(RX(0, theta3), RY(1, theta4)),
)

values = {
    "theta1": torch.tensor([0.5]),
    "theta2": torch.tensor([1.5]),
    "theta3": torch.tensor([2.0]),
    "theta4": torch.tensor([2.5]),
}

values2 = {
    "theta1": torch.tensor([0.5, 1.0]),
    "theta2": torch.tensor([1.5, 2.0]),
    "theta3": torch.tensor([2.0, 2.5]),
    "theta4": torch.tensor([2.5, 3.0]),
}


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize(
    "circuit, values, diff_mode",
    [
        (QuantumCircuit(2, blocks), values, DiffMode.AD),
        (QuantumCircuit(2, blocks), values2, DiffMode.GPSR),
    ],
)
def test_estimations_comparison_tomo_forward_pass(
    circuit: QuantumCircuit, values: dict, diff_mode: DiffMode
) -> None:
    observable = Z(0) ^ circuit.n_qubits

    pyq_backend = backend_factory(BackendName.PYQTORCH, diff_mode=diff_mode)
    (conv_circ, conv_obs, embed, params) = pyq_backend.convert(circuit, observable)
    pyq_exp_exact = pyq_backend.expectation(conv_circ, conv_obs, embed(params, values))

    model = QuantumModel(
        circuit=circuit,
        observable=observable,
        backend=BackendName.PYQTORCH,
        diff_mode=DiffMode.GPSR,
    )

    options = {"n_shots": 100000}
    tomo_measurements = Measurements(protocol=MeasurementProtocols.TOMOGRAPHY, options=options)
    estimated_exp_tomo = tomo_measurements(model, param_values=values)

    new_options = {"accuracy": 0.1, "confidence": 0.1}
    shadow_measurements = Measurements(protocol=MeasurementProtocols.SHADOW, options=new_options)
    estimated_exp_shadow = shadow_measurements(model, param_values=values)

    robust_options = {"shadow_size": 54400, "shadow_groups": 6, "robust_correlations": None}
    robust_shadows = Measurements(
        protocol=MeasurementProtocols.ROBUST_SHADOW, options=robust_options
    )
    robust_estimated_exp_shadow = robust_shadows(model, param_values=values)

    assert torch.allclose(estimated_exp_tomo, pyq_exp_exact, atol=1.0e-2)
    assert torch.allclose(estimated_exp_shadow, pyq_exp_exact, atol=0.1)
    assert torch.allclose(estimated_exp_shadow, pyq_exp_exact, atol=0.1)
    assert torch.allclose(robust_estimated_exp_shadow, pyq_exp_exact, atol=0.1)


def test_shadow_raise_errors() -> None:
    backend = BackendName.PYQTORCH
    model = QuantumModel(
        circuit=QuantumCircuit(2, kron(X(0), X(1))), observable=None, backend=backend
    )

    # Bad input keys
    options = {"accuracy": 0.1, "conf": 0.1}
    with pytest.raises(KeyError):
        shadow_measurement = Measurements(
            protocol=MeasurementProtocols.SHADOW,
            options=options,
        )

    options = {"accuracies": 0.1, "confidence": 0.1}
    with pytest.raises(KeyError):
        shadow_measurement = Measurements(
            protocol=MeasurementProtocols.SHADOW,
            options=options,
        )
