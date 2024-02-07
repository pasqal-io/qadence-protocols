from __future__ import annotations

from collections import Counter

import pytest
from metrics import MIDDLE_ACCEPTANCE
from qadence import (
    AbstractBlock,
    QuantumCircuit,
    QuantumModel,
    hamiltonian_factory,
)
from qadence.divergences import js_divergence
from qadence.noise.protocols import Noise
from qadence.operations import CNOT, RX, RZ, HamEvo, X, Y, Z, add, kron
from qadence.types import BackendName, ReadOutOptimization

from qadence_protocols.mitigations.protocols import Mitigations


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize(
    "error_probability, n_shots, block, backend, optimization_type",
    [
        (0.1, 100, kron(X(0), X(1)), BackendName.BRAKET, ReadOutOptimization.MLE),
        (
            0.1,
            1000,
            kron(Z(0), Z(1), Z(2)) + kron(X(0), Y(1), Z(2)),
            BackendName.BRAKET,
            ReadOutOptimization.MLE,
        ),
        (0.15, 1000, add(Z(0), Z(1), Z(2)), BackendName.BRAKET, ReadOutOptimization.CONSTRAINED),
        (
            0.1,
            5000,
            kron(X(0), X(1)) + kron(Z(0), Z(1)) + kron(X(2), X(3)),
            BackendName.BRAKET,
            ReadOutOptimization.CONSTRAINED,
        ),
        (
            0.1,
            500,
            add(Z(0), Z(1), kron(X(2), X(3))) + add(X(2), X(3)),
            BackendName.BRAKET,
            ReadOutOptimization.MLE,
        ),
        (
            0.1,
            2000,
            add(kron(Z(0), Z(1)), kron(X(2), X(3))),
            BackendName.BRAKET,
            ReadOutOptimization.MLE,
        ),
        (
            0.1,
            1300,
            kron(Z(0), Z(1)) + CNOT(0, 1),
            BackendName.BRAKET,
            ReadOutOptimization.CONSTRAINED,
        ),
        (
            0.05,
            1500,
            kron(RZ(0, parameter=0.01), RZ(1, parameter=0.01))
            + kron(RX(0, parameter=0.01), RX(1, parameter=0.01)),
            BackendName.PULSER,
            ReadOutOptimization.CONSTRAINED,
        ),
        (
            0.001,
            5000,
            HamEvo(generator=kron(Z(0), Z(1)), parameter=0.05),
            BackendName.BRAKET,
            ReadOutOptimization.MLE,
        ),
        (
            0.12,
            2000,
            HamEvo(generator=kron(Z(0), Z(1), Z(2)), parameter=0.001),
            BackendName.BRAKET,
            ReadOutOptimization.MLE,
        ),
        (
            0.1,
            1000,
            HamEvo(generator=kron(Z(0), Z(1)) + kron(Z(0), Z(1), Z(2)), parameter=0.005),
            BackendName.BRAKET,
            ReadOutOptimization.CONSTRAINED,
        ),
        (0.1, 100, kron(X(0), X(1)), BackendName.PYQTORCH, ReadOutOptimization.CONSTRAINED),
        (
            0.1,
            200,
            kron(Z(0), Z(1), Z(2)) + kron(X(0), Y(1), Z(2)),
            BackendName.PYQTORCH,
            ReadOutOptimization.MLE,
        ),
        (0.01, 1000, add(Z(0), Z(1), Z(2)), BackendName.PYQTORCH, ReadOutOptimization.MLE),
        (
            0.1,
            2000,
            HamEvo(
                generator=kron(X(0), X(1)) + kron(Z(0), Z(1)) + kron(X(2), X(3)), parameter=0.005
            ),
            BackendName.PYQTORCH,
            ReadOutOptimization.CONSTRAINED,
        ),
        (
            0.1,
            500,
            add(Z(0), Z(1), kron(X(2), X(3))) + add(X(2), X(3)),
            BackendName.PYQTORCH,
            ReadOutOptimization.CONSTRAINED,
        ),
        (
            0.05,
            10000,
            add(kron(Z(0), Z(1)), kron(X(2), X(3))),
            BackendName.PYQTORCH,
            ReadOutOptimization.MLE,
        ),
        (
            0.2,
            1000,
            hamiltonian_factory(4, detuning=Z),
            BackendName.PYQTORCH,
            ReadOutOptimization.MLE,
        ),
        (
            0.1,
            500,
            kron(Z(0), Z(1)) + CNOT(0, 1),
            BackendName.PYQTORCH,
            ReadOutOptimization.CONSTRAINED,
        ),
    ],
)
def test_readout_mitigation_quantum_model(
    error_probability: float,
    n_shots: int,
    block: AbstractBlock,
    backend: BackendName,
    optimization_type: str,
) -> None:
    diff_mode = "ad" if backend == BackendName.PYQTORCH else "gpsr"
    circuit = QuantumCircuit(block.n_qubits, block)
    noise = Noise(protocol=Noise.READOUT)
    model = QuantumModel(circuit=circuit, backend=backend, diff_mode=diff_mode)

    noiseless_samples: list[Counter] = model.sample(n_shots=n_shots)
    # Run noisy simulations through samples.
    noisy_samples: list[Counter] = model.sample(noise=noise, n_shots=n_shots)
    # Pass the noisy samples to the mitigation protocol.
    mitigate = Mitigations(
        protocol=Mitigations.READOUT,
        options={"optimization_type": optimization_type, "samples": noisy_samples},
    ).mitigation()
    mitigated_samples = mitigate(model=model)

    js_mitigated = js_divergence(mitigated_samples[0], noiseless_samples[0])
    js_noisy = js_divergence(noisy_samples[0], noiseless_samples[0])
    assert js_mitigated < js_noisy

    # Noisy simulations through the QM.
    noisy_model = QuantumModel(circuit=circuit, backend=backend, diff_mode=diff_mode, noise=noise)
    noisy_samples = noisy_model.sample(noise=noise, n_shots=n_shots)
    mitigate = Mitigations(
        protocol=Mitigations.READOUT,
        options={"optimization_type": optimization_type, "samples": noisy_samples},
    ).mitigation()
    mitigated_samples = mitigate(model=model)
    js_mitigated = js_divergence(mitigated_samples[0], noiseless_samples[0])
    js_noisy = js_divergence(noisy_samples[0], noiseless_samples[0])
    assert js_mitigated < js_noisy

    # Noisy simulations through the protocol.
    model = QuantumModel(circuit=circuit, backend=backend, diff_mode=diff_mode)
    mitigate = Mitigations(
        protocol=Mitigations.READOUT,
        options={"optimization_type": optimization_type, "n_shots": n_shots},
    ).mitigation()
    mitigated_samples = mitigate(model=model, noise=noise)
    js_mitigated = js_divergence(mitigated_samples[0], noiseless_samples[0])
    js_noisy = js_divergence(noisy_samples[0], noiseless_samples[0])
    assert js_mitigated < js_noisy


@pytest.mark.parametrize(
    "error_probability, n_shots, block, backend",
    [
        (0.1, 100, kron(X(0), X(1)), BackendName.BRAKET),
        (0.1, 1000, kron(Z(0), Z(1), Z(2)) + kron(X(0), Y(1), Z(2)), BackendName.BRAKET),
        (0.1, 500, add(Z(0), Z(1), kron(X(2), X(3))) + add(X(2), X(3)), BackendName.BRAKET),
        (0.1, 2000, add(kron(Z(0), Z(1)), kron(X(2), X(3))), BackendName.BRAKET),
    ],
)
def test_compare_readout_methods(
    error_probability: float,
    n_shots: int,
    block: AbstractBlock,
    backend: BackendName,
) -> None:
    diff_mode = "ad" if backend == BackendName.PYQTORCH else "gpsr"
    circuit = QuantumCircuit(block.n_qubits, block)
    model = QuantumModel(circuit=circuit, backend=backend, diff_mode=diff_mode)

    noise = Noise(protocol=Noise.READOUT)

    noiseless_samples: list[Counter] = model.sample(n_shots=n_shots)

    mitigation_mle = Mitigations(
        protocol=Mitigations.READOUT,
        options={"optimization_type": ReadOutOptimization.CONSTRAINED, "n_shots": n_shots},
    ).mitigation()
    mitigated_samples_mle: list[Counter] = mitigation_mle(model=model, noise=noise)

    mitigation_constrained_opt = Mitigations(
        protocol=Mitigations.READOUT,
        options={"optimization_type": ReadOutOptimization.MLE, "n_shots": n_shots},
    ).mitigation()
    mitigated_samples_constrained_opt: list[Counter] = mitigation_constrained_opt(
        model=model, noise=noise
    )

    js_mitigated_mle = js_divergence(mitigated_samples_mle[0], noiseless_samples[0])
    js_mitigated_constrained_opt = js_divergence(
        mitigated_samples_constrained_opt[0], noiseless_samples[0]
    )
    assert abs(js_mitigated_constrained_opt - js_mitigated_mle) <= MIDDLE_ACCEPTANCE
