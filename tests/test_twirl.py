from __future__ import annotations

import pytest
import torch
from qadence import (
    AbstractBlock,
    QuantumCircuit,
    QuantumModel,
    add,
    chain,
    kron,
)
from qadence.measurements import Measurements
from qadence.noise.protocols import Noise
from qadence.operations import CNOT, RX, Z
from qadence.types import BackendName

from qadence_protocols.mitigations.protocols import Mitigations


@pytest.mark.parametrize(
    "error_probability, n_shots, n_qubits, block, observable, backend",
    [
        (
            0.2,
            10000,
            2,
            chain(kron(RX(0, torch.pi / 3), RX(1, torch.pi / 3)), CNOT(0, 1)),
            [add(2 * Z(1) + Z(0)), add(4 * kron(Z(0), Z(1)) + Z(0))],
            # [Z(0)],
            BackendName.PYQTORCH,
        )
    ],
)
def test_readout_twirl_mitigation(
    error_probability: float,
    n_shots: int,
    n_qubits: int,
    block: AbstractBlock,
    observable: AbstractBlock,
    backend: BackendName,
) -> None:
    circuit = QuantumCircuit(block.n_qubits, block)
    # Z_obs = sum([kron(*[Z(i) for i in a]) for a in observable])

    noise = Noise(protocol=Noise.READOUT, options={"error_probability": error_probability})
    tomo_measurement = Measurements(
        protocol=Measurements.TOMOGRAPHY,
        options={"n_shots": n_shots},
    )

    model = QuantumModel(
        circuit=circuit,
        backend=BackendName.PYQTORCH,
        diff_mode="gpsr",
        observable=observable,
        measurement=tomo_measurement,
    )

    expectation_noiseless = model.expectation(
        measurement=tomo_measurement,
    )

    mitigate = Mitigations(protocol=Mitigations.TWIRL).mitigation()

    expectation_mitigated = mitigate(n_qubits, circuit, backend, noise, n_shots, observable)

    assert torch.allclose(expectation_mitigated, expectation_noiseless, atol=1.0e-2, rtol=1.0e-1)
