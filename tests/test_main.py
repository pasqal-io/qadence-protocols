from __future__ import annotations

import pytest
import torch
from qadence import (
    AbstractBlock,
    QuantumCircuit,
    QuantumModel,
    chain,
    kron,
)
from qadence.measurements import Measurements
from qadence.noise.protocols import Noise
from qadence.operations import CNOT, RX, Z
from qadence.types import BackendName

# from qadence_protocols.twirl_mitigation import twirl_mitigation
import qadence_protocols


@pytest.mark.parametrize(
    "error_probability, n_shots, block, backend, optimization_type",
    [
        (
            0.2,
            100000,
            2,
            chain(kron(RX(0, torch.pi / 3), RX(1, torch.pi / 3)), CNOT(0, 1)),
            [[0, 1], [1]],
            BackendName.PYQTORCH,
        )
    ],
)
def test_readout_mitigation_quantum_model(
    error_probability: float,
    n_shots: int,
    n_qubits: int,
    block: AbstractBlock,
    observable: list,
    backend: BackendName,
) -> None:
    circuit = QuantumCircuit(block.n_qubits, block)
    Z_obs = sum([kron(*[Z(i) for i in a]) for a in observable])

    noise = Noise(protocol=Noise.READOUT, options={"error_probability": error_probability})
    tomo_measurement = Measurements(
        protocol=Measurements.TOMOGRAPHY,
        options={"n_shots": n_shots},
    )

    model = QuantumModel(
        circuit=circuit,
        backend=BackendName.PYQTORCH,
        diff_mode="gpsr",
        observable=Z_obs,
        measurement=tomo_measurement,
    )

    expectation_noisless = model.expectation(
        measurement=tomo_measurement,
    )
    # expectation_noisy = model.expectation(measurement=tomo_measurement,noise=noise)

    expectation_mitigated = twirl_mitigation(n_qubits, circuit, backend, noise, n_shots, observable)

    # print(expectation_noisless,expectation_noisy,expectation_mitigated)

    assert torch.allclose(expectation_mitigated, expectation_noisless, atol=1.0e-1)
