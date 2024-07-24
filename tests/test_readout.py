from __future__ import annotations

from collections import Counter
from functools import reduce

import numpy as np
import numpy.typing as npt
import pytest
from metrics import LOW_ACCEPTANCE
from qadence import (
    AbstractBlock,
    Noise,
    QuantumCircuit,
    QuantumModel,
    add,
    chain,
    kron,
)
from qadence.divergences import js_divergence
from qadence.operations import CNOT, RX, X, Y, Z
from qadence.types import BackendName
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import gmres
from scipy.stats import wasserstein_distance

from qadence_protocols import Mitigations
from qadence_protocols.mitigations.readout import (
    majority_vote,
    matrix_inv,
    mle_solve,
    normalized_subspace_kron,
    tensor_rank_mult,
)
from qadence_protocols.types import ReadOutOptimization


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize(
    "error_probability, n_shots, block, backend, optimization_type",
    [
        (0.1, 100, kron(X(0), X(1)), BackendName.BRAKET, ReadOutOptimization.MLE),
        # (
        #     0.1,
        #     1000,
        #     kron(Z(0), Z(1), Z(2)) + kron(X(0), Y(1), Z(2)),
        #     BackendName.BRAKET,
        #     ReadOutOptimization.MLE,
        # ),
        # (0.15, 1000, add(Z(0), Z(1), Z(2)), BackendName.BRAKET, ReadOutOptimization.CONSTRAINED),
        # (
        #     0.1,
        #     5000,
        #     kron(X(0), X(1)) + kron(Z(0), Z(1)) + kron(X(2), X(3)),
        #     BackendName.BRAKET,
        #     ReadOutOptimization.CONSTRAINED,
        # ),
        # (
        #     0.1,
        #     500,
        #     add(Z(0), Z(1), kron(X(2), X(3))) + add(X(2), X(3)),
        #     BackendName.BRAKET,
        #     ReadOutOptimization.MLE,
        # ),
        # (
        #     0.1,
        #     2000,
        #     add(kron(Z(0), Z(1)), kron(X(2), X(3))),
        #     BackendName.BRAKET,
        #     ReadOutOptimization.MLE,
        # ),
        # (
        #     0.1,
        #     1300,
        #     kron(Z(0), Z(1)) + CNOT(0, 1),
        #     BackendName.BRAKET,
        #     ReadOutOptimization.CONSTRAINED,
        # ),
        # (
        #     0.05,
        #     1500,
        #     kron(RZ(0, parameter=0.01), RZ(1, parameter=0.01))
        #     + kron(RX(0, parameter=0.01), RX(1, parameter=0.01)),
        #     BackendName.PULSER,
        #     ReadOutOptimization.CONSTRAINED,
        # ),
        # (
        #     0.001,
        #     5000,
        #     HamEvo(generator=kron(Z(0), Z(1)), parameter=0.05),
        #     BackendName.BRAKET,
        #     ReadOutOptimization.MLE,
        # ),
        # (
        #     0.12,
        #     2000,
        #     HamEvo(generator=kron(Z(0), Z(1), Z(2)), parameter=0.001),
        #     BackendName.BRAKET,
        #     ReadOutOptimization.MLE,
        # ),
        # (
        #     0.1,
        #     1000,
        #     HamEvo(generator=kron(Z(0), Z(1)) + kron(Z(0), Z(1), Z(2)), parameter=0.005),
        #     BackendName.BRAKET,
        #     ReadOutOptimization.CONSTRAINED,
        # ),
        # (0.1, 100, kron(X(0), X(1)), BackendName.PYQTORCH, ReadOutOptimization.CONSTRAINED),
        # (
        #     0.1,
        #     200,
        #     kron(Z(0), Z(1), Z(2)) + kron(X(0), Y(1), Z(2)),
        #     BackendName.PYQTORCH,
        #     ReadOutOptimization.MLE,
        # ),
        # (0.01, 1000, add(Z(0), Z(1), Z(2)), BackendName.PYQTORCH, ReadOutOptimization.MLE),
        # (
        #     0.1,
        #     2000,
        #     HamEvo(
        #         generator=kron(X(0), X(1)) + kron(Z(0), Z(1)) + kron(X(2), X(3)), parameter=0.005
        #     ),
        #     BackendName.PYQTORCH,
        #     ReadOutOptimization.CONSTRAINED,
        # ),
        # (
        #     0.1,
        #     500,
        #     add(Z(0), Z(1), kron(X(2), X(3))) + add(X(2), X(3)),
        #     BackendName.PYQTORCH,
        #     ReadOutOptimization.CONSTRAINED,
        # ),
        # (
        #     0.05,
        #     10000,
        #     add(kron(Z(0), Z(1)), kron(X(2), X(3))),
        #     BackendName.PYQTORCH,
        #     ReadOutOptimization.MLE,
        # ),
        # (
        #     0.2,
        #     1000,
        #     hamiltonian_factory(4, detuning=Z),
        #     BackendName.PYQTORCH,
        #     ReadOutOptimization.MLE,
        # ),
        # (
        #     0.1,
        #     500,
        #     kron(Z(0), Z(1)) + CNOT(0, 1),
        #     BackendName.PYQTORCH,
        #     ReadOutOptimization.CONSTRAINED,
        # ),
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
    noise = Noise(protocol=Noise.READOUT, options={"error_probability": error_probability})
    model = QuantumModel(circuit=circuit, backend=backend, diff_mode=diff_mode)

    noiseless_samples: list[Counter] = model.sample(n_shots=n_shots)
    # Run noisy simulations through samples.
    noisy_samples: list[Counter] = model.sample(noise=noise, n_shots=n_shots)
    # Pass the noisy samples to the mitigation protocol.
    mitigate = Mitigations(
        protocol=Mitigations.READOUT,
        options={"optimization_type": optimization_type, "samples": noisy_samples},
    )
    mitigated_samples = mitigate(model=model, noise=noise)

    js_mitigated = js_divergence(mitigated_samples[0], noiseless_samples[0])
    js_noisy = js_divergence(noisy_samples[0], noiseless_samples[0])
    assert js_mitigated < js_noisy

    # Noisy simulations through the QM.
    noisy_model = QuantumModel(circuit=circuit, backend=backend, diff_mode=diff_mode, noise=noise)
    noisy_samples = noisy_model.sample(noise=noise, n_shots=n_shots)
    mitigate = Mitigations(
        protocol=Mitigations.READOUT,
        options={"optimization_type": optimization_type, "samples": noisy_samples},
    )
    mitigated_samples = mitigate(model=noisy_model)
    js_mitigated = js_divergence(mitigated_samples[0], noiseless_samples[0])
    js_noisy = js_divergence(noisy_samples[0], noiseless_samples[0])
    assert js_mitigated < js_noisy
    # Noisy simulations through the protocol.
    mitigate = Mitigations(
        protocol=Mitigations.READOUT,
        options={"optimization_type": optimization_type, "n_shots": n_shots},
    )
    mitigated_samples = mitigate(model=model, noise=noise)
    js_mitigated = js_divergence(mitigated_samples[0], noiseless_samples[0])
    js_noisy = js_divergence(noisy_samples[0], noiseless_samples[0])
    assert js_mitigated < js_noisy


@pytest.mark.parametrize(
    "error_probability, n_shots, block, backend",
    [
        (0.1, 5000, kron(X(0), X(1)), BackendName.BRAKET),
        (0.1, 5000, kron(Z(0), Z(1), Z(2)) + kron(X(0), Y(1), Z(2)), BackendName.BRAKET),
        (0.1, 5000, add(Z(0), Z(1), kron(X(2), X(3))) + add(X(2), X(3)), BackendName.BRAKET),
        (0.1, 5000, add(kron(Z(0), Z(1)), kron(X(2), X(3))), BackendName.BRAKET),
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

    noise = Noise(protocol=Noise.READOUT, options={"error_probability": error_probability})

    noiseless_samples: list[Counter] = model.sample(n_shots=n_shots)

    mitigation_mle = Mitigations(
        protocol=Mitigations.READOUT,
        options={"optimization_type": ReadOutOptimization.MLE, "n_shots": n_shots},
    )
    mitigated_samples_mle: list[Counter] = mitigation_mle(model=model, noise=noise)

    mitigation_constrained_opt = Mitigations(
        protocol=Mitigations.READOUT,
        options={"optimization_type": ReadOutOptimization.CONSTRAINED, "n_shots": n_shots},
    )
    mitigated_samples_constrained_opt: list[Counter] = mitigation_constrained_opt(
        model=model, noise=noise
    )

    js_mitigated_mle = js_divergence(mitigated_samples_mle[0], noiseless_samples[0])
    js_mitigated_constrained_opt = js_divergence(
        mitigated_samples_constrained_opt[0], noiseless_samples[0]
    )
    assert js_mitigated_mle <= js_mitigated_constrained_opt + LOW_ACCEPTANCE


@pytest.mark.parametrize(
    "qubit_ops,input_vec",
    [
        ([np.random.rand(2, 2) for i in range(4)], np.random.rand(2**4)),
        ([np.random.rand(2, 2) for i in range(4)], np.random.rand(2**4)),
        ([np.random.rand(2, 2) for i in range(4)], np.random.rand(2**4)),
        ([np.random.rand(2, 2) for i in range(4)], np.random.rand(2**4)),
    ],
)
def test_tensor_rank_mult(qubit_ops: list[npt.NDarray], input_vec: npt.NDArray) -> None:
    full_tensor = reduce(np.kron, qubit_ops)

    assert (
        np.linalg.norm(tensor_rank_mult(qubit_ops, input_vec) - full_tensor @ input_vec)
        < LOW_ACCEPTANCE
    )


@pytest.mark.parametrize(
    "error_probability, n_shots, block, backend",
    [
        (
            0.2,
            10000,
            chain(kron(RX(0, np.pi / 3), RX(1, np.pi / 3)), CNOT(0, 1)),
            BackendName.PYQTORCH,
        ),
        (
            0.1,
            10000,
            chain(kron(RX(0, np.pi / 4), RX(1, np.pi / 5)), CNOT(0, 1)),
            BackendName.PYQTORCH,
        ),
        (
            0.15,
            10000,
            chain(kron(RX(0, np.pi / 3), RX(1, np.pi / 6)), CNOT(0, 1)),
            BackendName.PYQTORCH,
        ),
        (
            0.2,
            10000,
            chain(kron(RX(0, np.pi / 6), RX(1, np.pi / 4)), CNOT(0, 1)),
            BackendName.PYQTORCH,
        ),
    ],
)
def test_readout_mthree_mitigation(
    error_probability: float,
    n_shots: int,
    block: AbstractBlock,
    backend: BackendName,
) -> None:
    circuit = QuantumCircuit(block.n_qubits, block)
    noise = Noise(protocol=Noise.READOUT, options={"error_probability": error_probability})

    model = QuantumModel(circuit=circuit, backend=backend)

    ordered_bitstrings = [bin(k)[2:].zfill(block.n_qubits) for k in range(2**block.n_qubits)]

    mitigation_mle = Mitigations(
        protocol=Mitigations.READOUT,
        options={"optimization_type": ReadOutOptimization.MLE, "n_shots": n_shots},
    )

    samples_mle = mitigation_mle(model=model, noise=noise)[0]
    p_mle = np.array([samples_mle[bs] for bs in ordered_bitstrings]) / sum(samples_mle.values())

    mitigation_mthree = Mitigations(
        protocol=Mitigations.READOUT,
        options={"optimization_type": ReadOutOptimization.MTHREE, "n_shots": n_shots},
    )
    samples_mthree = mitigation_mthree(model=model, noise=noise)[0]
    p_mthree = np.array([samples_mthree[bs] for bs in ordered_bitstrings]) / sum(
        samples_mthree.values()
    )

    assert wasserstein_distance(p_mle, p_mthree) < LOW_ACCEPTANCE


def test_readout_mthree_sparse() -> None:
    n_qubits = 10
    exact_prob = np.random.rand(2 ** (n_qubits - 5))
    exact_prob = exact_prob / sum(exact_prob)
    exact_prob = np.concatenate([exact_prob, np.zeros(2**n_qubits - len(exact_prob))], axis=0)
    exact_prob = 0.90 * exact_prob + 0.1 * np.ones(2**n_qubits) / 2**n_qubits
    np.random.shuffle(exact_prob)

    observed_prob = np.array(exact_prob, copy=True)
    observed_prob[exact_prob < 1 / 2 ** (n_qubits)] = 0

    noise_matrices = []
    for t in range(n_qubits):
        t_a, t_b = np.random.rand(2) / 8
        K = np.array([[1 - t_a, t_a], [t_b, 1 - t_b]]).transpose()  # column sum be 1
        noise_matrices.append(K)

    confusion_matrix_subspace = normalized_subspace_kron(noise_matrices, observed_prob.nonzero()[0])

    input_csr = csr_matrix(observed_prob, shape=(1, 2**n_qubits)).T

    p_corr_mthree_gmres = gmres(confusion_matrix_subspace, input_csr.toarray())[0]
    p_corr_mthree_gmres_mle = mle_solve(p_corr_mthree_gmres)

    noise_matrices_inv = list(map(matrix_inv, noise_matrices))
    p_corr_inv_mle = mle_solve(tensor_rank_mult(noise_matrices_inv, exact_prob))

    assert wasserstein_distance(p_corr_mthree_gmres_mle, p_corr_inv_mle) < LOW_ACCEPTANCE


@pytest.mark.flaky(max_runs=2)
@pytest.mark.parametrize(
    "n_qubits,index",
    [
        (4, 6),
        (4, 1),
        (5, 25),
        (5, 20),
        (5, 9),
        (5, 11),
        (4, 15),
        (5, 7),
    ],
)
def test_readout_majority_vote(n_qubits: int, index: int) -> None:
    prob = np.zeros(2**n_qubits)
    # let the all zero string be the most possible solution
    prob[index] = 1

    noise_matrices = []
    for t in range(n_qubits):
        t_a, t_b = np.random.rand(2) / 20
        K = np.array([[1 - t_a, t_a], [t_b, 1 - t_b]]).transpose()  # column sum be 1
        noise_matrices.append(K)

    confusion_matrix = reduce(np.kron, noise_matrices)
    noisy_prob = confusion_matrix @ prob

    # remove the expected measurement from output
    # generate a bit string of len n_qubits

    noisy_prob[index] = 0
    noisy_prob /= sum(noisy_prob)

    assert majority_vote(noise_matrices, noisy_prob).argmax() == index
