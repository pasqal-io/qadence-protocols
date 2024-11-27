from __future__ import annotations

import pytest
import torch
from qadence import (
    AbstractBlock,
    NoiseHandler,
    QuantumCircuit,
    QuantumModel,
    add,
    chain,
    kron,
)
from qadence.measurements import Measurements
from qadence.operations import CNOT, RX, Z
from qadence.types import BackendName, NoiseProtocol

from qadence_protocols import Mitigations


@pytest.mark.parametrize(
    "error_probability, n_shots, block, observable, backend",
    [
        (
            0.2,
            10000,
            chain(kron(RX(0, torch.pi / 3), RX(1, torch.pi / 3)), CNOT(0, 1)),
            [add(kron(Z(0), Z(1)) + Z(0))],
            BackendName.PYQTORCH,
        ),
        (
            0.1,
            10000,
            chain(kron(RX(0, torch.pi / 4), RX(1, torch.pi / 5)), CNOT(0, 1)),
            [2 * Z(1) + 3 * Z(0), 3 * kron(Z(0), Z(1)) - 1 * Z(0)],
            BackendName.PYQTORCH,
        ),
        (
            0.15,
            10000,
            chain(kron(RX(0, torch.pi / 3), RX(1, torch.pi / 6)), CNOT(0, 1)),
            [add(Z(1), -Z(0)), 3 * kron(Z(0), Z(1)) + 2 * Z(0)],
            BackendName.PYQTORCH,
        ),
        (
            0.2,
            10000,
            chain(kron(RX(0, torch.pi / 6), RX(1, torch.pi / 4)), CNOT(0, 1)),
            [add(Z(1), -2 * Z(0)), add(2 * kron(Z(0), Z(1)), 4 * Z(0))],
            BackendName.PYQTORCH,
        ),
    ],
)
def test_readout_twirl_mitigation(
    error_probability: float,
    n_shots: int,
    block: AbstractBlock,
    observable: AbstractBlock,
    backend: BackendName,
) -> None:
    circuit = QuantumCircuit(block.n_qubits, block)
    noise = NoiseHandler(
        protocol=NoiseProtocol.READOUT.INDEPENDENT, options={"error_probability": error_probability}
    )
    tomo_measurement = Measurements(
        protocol=Measurements.TOMOGRAPHY,
        options={"n_shots": n_shots},
    )

    model = QuantumModel(
        circuit=circuit, observable=observable, measurement=tomo_measurement, backend=backend
    )

    expectation_noiseless = model.expectation(
        measurement=tomo_measurement,
    )

    noisy_model = QuantumModel(
        circuit=circuit,
        observable=observable,
        measurement=tomo_measurement,
        noise=noise,
        backend=backend,
    )
    mitigate = Mitigations(protocol=Mitigations.TWIRL)
    expectation_mitigated = mitigate(noisy_model)
    assert torch.allclose(expectation_mitigated, expectation_noiseless, atol=1.0e-1, rtol=5.0e-2)
