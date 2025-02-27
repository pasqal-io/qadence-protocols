from __future__ import annotations

from collections import Counter
from functools import reduce

import numpy as np
import numpy.typing as npt
import pytest
import strategies as st
from hypothesis import given, settings
from metrics import LOW_ACCEPTANCE
from qadence import (
    AbstractBlock,
    HamEvo,
    NoiseHandler,
    QuantumCircuit,
    QuantumModel,
    add,
    hamiltonian_factory,
    kron,
)
from qadence.divergences import js_divergence
from qadence.ml_tools.utils import rand_featureparameters
from qadence.operations import CNOT, X, Y, Z
from qadence.types import BackendName, NoiseProtocol
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import gmres
from scipy.stats import wasserstein_distance

from qadence_protocols import Mitigations
from qadence_protocols.mitigations.readout import (
    ham_dist_redistribution,
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
def test_readout_mitigation(
    error_probability: float,
    n_shots: int,
    block: AbstractBlock,
    backend: BackendName,
    optimization_type: str,
) -> None:
    diff_mode = "ad" if backend == BackendName.PYQTORCH else "gpsr"
    circuit = QuantumCircuit(block.n_qubits, block)
    noise = NoiseHandler(
        protocol=NoiseProtocol.READOUT.INDEPENDENT, options={"error_probability": error_probability}
    )
    model = QuantumModel(circuit=circuit, backend=backend, diff_mode=diff_mode)

    noiseless_samples: list[Counter] = model.sample(n_shots=n_shots)
    # Run noisy simulations through samples.
    noisy_samples: list[Counter] = model.sample(noise=noise, n_shots=n_shots)
    # Pass the noisy samples to the mitigation protocol.
    mitigate = Mitigations(
        protocol=Mitigations.READOUT,
        options={"optimization_type": optimization_type, "samples": noisy_samples},
    )
    mitigated_samples = mitigate(noise=noise, model=model)

    js_mitigated = js_divergence(mitigated_samples[0], noiseless_samples[0])
    js_noisy = js_divergence(noisy_samples[0], noiseless_samples[0])
    assert js_mitigated < js_noisy

    # Pass the noisy samples to the mitigation protocol and run without model
    mitigate = Mitigations(
        protocol=Mitigations.READOUT,
        options={"optimization_type": optimization_type, "samples": noisy_samples},
    )
    mitigated_samples = mitigate(noise=noise)

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
    mitigated_samples = mitigate(noise=noise, model=noisy_model)
    js_mitigated = js_divergence(mitigated_samples[0], noiseless_samples[0])
    js_noisy = js_divergence(noisy_samples[0], noiseless_samples[0])
    assert js_mitigated < js_noisy
    # Noisy simulations through the protocol.
    mitigate = Mitigations(
        protocol=Mitigations.READOUT,
        options={"optimization_type": optimization_type, "n_shots": n_shots},
    )
    mitigated_samples = mitigate(noise=noise, model=model)
    js_mitigated = js_divergence(mitigated_samples[0], noiseless_samples[0])
    js_noisy = js_divergence(noisy_samples[0], noiseless_samples[0])
    assert js_mitigated < js_noisy


@given(st.digital_circuits())
@settings(deadline=None)
def test_compare_readout_methods(circuit: QuantumCircuit) -> None:
    error_probability = 0.1
    n_shots = 5000
    backend = BackendName.PYQTORCH
    inputs = rand_featureparameters(circuit, 1)

    diff_mode = "ad"
    model = QuantumModel(circuit=circuit, backend=backend, diff_mode=diff_mode)

    noise = NoiseHandler(
        protocol=NoiseProtocol.READOUT.INDEPENDENT, options={"error_probability": error_probability}
    )

    noiseless_samples: list[Counter] = model.sample(inputs, n_shots=n_shots)

    mitigation_mle = Mitigations(
        protocol=Mitigations.READOUT,
        options={"optimization_type": ReadOutOptimization.MLE, "n_shots": n_shots},
    )
    mitigated_samples_mle: list[Counter] = mitigation_mle(
        noise=noise, model=model, param_values=inputs
    )

    mitigation_constrained_opt = Mitigations(
        protocol=Mitigations.READOUT,
        options={"optimization_type": ReadOutOptimization.CONSTRAINED, "n_shots": n_shots},
    )
    mitigated_samples_constrained_opt: list[Counter] = mitigation_constrained_opt(
        noise=noise, model=model, param_values=inputs
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


@given(st.digital_circuits())
def test_readout_mthree_mitigation(
    circuit: QuantumCircuit,
) -> None:
    values = rand_featureparameters(circuit, 1)
    n_shots: int = 10000
    error_probability = np.random.rand()
    backend = BackendName.PYQTORCH
    noise = NoiseHandler(
        protocol=NoiseProtocol.READOUT.INDEPENDENT, options={"error_probability": error_probability}
    )

    model = QuantumModel(circuit=circuit, backend=backend)

    ordered_bitstrings = [bin(k)[2:].zfill(circuit.n_qubits) for k in range(2**circuit.n_qubits)]

    mitigation_mle = Mitigations(
        protocol=Mitigations.READOUT,
        options={"optimization_type": ReadOutOptimization.MLE, "n_shots": n_shots},
    )

    samples_mle = mitigation_mle(model=model, noise=noise, param_values=values)[0]
    p_mle = np.array([samples_mle[bs] for bs in ordered_bitstrings]) / sum(samples_mle.values())

    mitigation_mthree = Mitigations(
        protocol=Mitigations.READOUT,
        options={"optimization_type": ReadOutOptimization.MTHREE, "n_shots": n_shots},
    )
    samples_mthree = mitigation_mthree(model=model, noise=noise, param_values=values)[0]
    p_mthree = np.array([samples_mthree[bs] for bs in ordered_bitstrings]) / sum(
        samples_mthree.values()
    )

    assert wasserstein_distance(p_mle, p_mthree) < LOW_ACCEPTANCE


def test_readout_mthree_sparse() -> None:
    n_qubits = 10
    exact_prob = np.random.rand(2 ** (n_qubits))
    exact_prob[2 ** (n_qubits // 2) :] = 0
    exact_prob = 0.90 * exact_prob + 0.1 * np.ones(2**n_qubits) / 2**n_qubits
    exact_prob = exact_prob / sum(exact_prob)
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
    p_corr_inv_mle = mle_solve(tensor_rank_mult(noise_matrices_inv, observed_prob))

    assert wasserstein_distance(p_corr_mthree_gmres_mle, p_corr_inv_mle) < LOW_ACCEPTANCE


def test_readout_mthree_sparse_ham() -> None:
    n_qubits = 10
    exact_prob = np.random.rand(2 ** (n_qubits))
    exact_prob[2 ** (n_qubits // 2) :] = 0
    exact_prob = 0.90 * exact_prob + 0.1 * np.ones(2**n_qubits) / 2**n_qubits
    exact_prob = exact_prob / sum(exact_prob)
    np.random.shuffle(exact_prob)

    observed_prob = np.array(exact_prob, copy=True)
    observed_prob[exact_prob < 1 / 2 ** (n_qubits)] = 0

    noise_matrices = []
    for t in range(n_qubits):
        t_a, t_b = np.random.rand(2) / 8
        K = np.array([[1 - t_a, t_a], [t_b, 1 - t_b]]).transpose()  # column sum be 1
        noise_matrices.append(K)

    confusion_matrix_subspace = normalized_subspace_kron(noise_matrices, observed_prob.nonzero()[0])

    # we consider a small hamming distance for this method and set it to 2
    confusion_matrix_subspace_ham = ham_dist_redistribution(confusion_matrix_subspace, 2)

    input_csr = csr_matrix(observed_prob, shape=(1, 2**n_qubits)).T

    p_corr_mthree_gmres_ham = gmres(confusion_matrix_subspace_ham, input_csr.toarray())[0]
    p_corr_mthree_gmres_mle_ham = mle_solve(p_corr_mthree_gmres_ham)

    noise_matrices_inv = list(map(matrix_inv, noise_matrices))
    p_corr_inv_mle = mle_solve(tensor_rank_mult(noise_matrices_inv, observed_prob))

    assert wasserstein_distance(p_corr_mthree_gmres_mle_ham, p_corr_inv_mle) < LOW_ACCEPTANCE


@pytest.mark.flaky(max_runs=5)
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
