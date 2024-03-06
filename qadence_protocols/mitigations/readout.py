from __future__ import annotations

from collections import Counter

import numpy as np
import numpy.typing as npt
from numpy.linalg import inv, matrix_rank, pinv
from qadence import QuantumModel
from qadence.noise.protocols import Noise
from scipy.linalg import norm
from scipy.optimize import LinearConstraint, minimize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import gmres

from qadence_protocols.types import ReadOutOptimization


def subspace_kron(
    noise_matrices: npt.NDArrary, subspace: npt.NDArray, normalize: bool = True
) -> npt.NDArray:
    n_qubits = len(noise_matrices)
    conf_matrix = csr_matrix((2**n_qubits, 2**n_qubits))

    for j in subspace:
        for i in subspace:
            bin_i = bin(i)[2:].zfill(n_qubits)
            bin_j = bin(j)[2:].zfill(n_qubits)

            ## manually computing the entries of tensor product for only the subspace
            conf_matrix[i, j] = np.prod(
                [noise_matrices[k][int(bin_i[k])][int(bin_j[k])] for k in range(n_qubits)]
            )

        if normalize:
            conf_matrix[:, j] /= np.sum(conf_matrix[:, j])

    return conf_matrix


def tensor_rank_mult(qubit_ops: npt.NDArray, prob_vect: npt.NDArray) -> npt.NDArray:
    N = int(np.log2(len(prob_vect)))
    """
    Fast multiplication of single qubit operators on a probability vector.

    Similar to how gate operations are implemented on PyqTorch
    Needs to be replaced.
    """

    # Reshape probability vector into a rank N tensor
    prob_vect_t = prob_vect.reshape(N * [2]).transpose()

    # Contract each tensor index (qubit) with the inverse of the single-qubit
    for i in range(N):
        prob_vect_t = np.tensordot(qubit_ops[N - 1 - i], prob_vect_t, axes=(1, i))

    # Obtain corrected measurements by shaping back into a vector
    return prob_vect_t.reshape(2**N)


def corrected_probas(
    p_corr: npt.NDArray, noise_matrices: npt.NDArray, p_raw: npt.NDArray
) -> np.double:
    ## Computing rectified probabilites without computing the full T matrix
    p_estim = tensor_rank_mult(noise_matrices, p_corr.T)
    return norm(p_estim - p_raw.T, ord=2) ** 2


def mle_solve(p_raw: npt.NDArray) -> npt.NDArray:
    """
    Compute the MLE probability vector.

    Algorithmic details can be found in https://arxiv.org/pdf/1106.5458.pdf Page(3).
    """
    # Sort p_raw by values while keeping track of indices.
    index_sort = p_raw.argsort()
    p_sort = p_raw[index_sort]
    neg_sum = 0
    breakpoint = len(p_sort) - 1

    for i in range(len(p_sort)):
        ## if neg_sum cannot be distributed among other probabilities, continue to accumulate
        if p_sort[i] + neg_sum / (len(p_sort) - i) < 0:
            neg_sum += p_sort[i]
            p_sort[i] = 0
        # set breakpoint to current index
        else:
            breakpoint = i
            break
    ## number of entries to which i can distribute(includes breakpoint)
    size = len(p_sort) - breakpoint
    p_sort[breakpoint:] += neg_sum / size

    re_index_sort = index_sort.argsort()
    p_corr = p_sort[re_index_sort]

    return p_corr


def renormalize_counts(corrected_counts: npt.NDArray, n_shots: int) -> npt.NDArray:
    """Renormalize counts rounding discrepancies."""
    total_counts = sum(corrected_counts)
    if total_counts != n_shots:
        counts_diff = total_counts - n_shots
        corrected_counts -= counts_diff
        corrected_counts = np.where(corrected_counts < 0, 0, corrected_counts)
        sum_corrected_counts = sum(corrected_counts)

        renormalization_factor = n_shots / sum_corrected_counts
        corrected_counts = np.rint(corrected_counts * renormalization_factor).astype(int)

    # At this point, the count should be off by at most 2, added or substracted to/from the
    # max count.
    if sum(corrected_counts) != n_shots:
        count_diff = sum(corrected_counts) - n_shots
        max_count_bs = np.argmax(corrected_counts)
        corrected_counts[max_count_bs] -= count_diff

    return corrected_counts


def matrix_inv(K: npt.NDArray) -> npt.NDArray:
    return inv(K) if matrix_rank(K) == K.shape[0] else pinv(K)


def constrained_inversion(noise_matrices: npt.NDArray, p_raw: npt.NDArray) -> npt.NDArray:
    # Initial random guess in [0,1].
    p_corr0 = np.random.rand(len(p_raw))
    # Stochasticity constraints.
    normality_constraint = LinearConstraint(np.ones(len(p_raw)).astype(int), lb=1.0, ub=1.0)
    positivity_constraint = LinearConstraint(np.eye(len(p_raw)).astype(int), lb=0.0, ub=1.0)
    constraints = [normality_constraint, positivity_constraint]
    # Minimize the corrected probabilities.
    res = minimize(corrected_probas, p_corr0, args=(noise_matrices, p_raw), constraints=constraints)

    return res.x


def mitigation_minimization(
    noise: Noise,
    options: dict,
    samples: list[Counter],
) -> list[Counter]:
    """Minimize a correction matrix subjected to stochasticity constraints.

    See Equation (5) in https://arxiv.org/pdf/2001.09980.pdf.
    See Page(3) in https://arxiv.org/pdf/1106.5458.pdf for MLE implementation
    See Equation (5) in https://arxiv.org/pdf/2108.12518.pdf for MTHREE implementation
    (matrix free measurement mitigation)
    This method is supposed to be be reserved for large implementations of 20 plus qubits

    Args:
        noise: Specifies confusion matrix and default error probability
        mitigation: Selects additional mitigation options based on noise choice.
        For readout we have the following mitigation options for optimization
        1. constrained 2. mle. Default : mle
        samples: List of samples to be mitigated

    Returns:
        Mitigated counts computed by the algorithm
    """
    noise_matrices = noise.options.get("noise_matrix", noise.options["confusion_matrices"]).numpy()
    optimization_type = options.get("optimization_type", ReadOutOptimization.MLE)
    n_qubits = len(list(samples[0].keys())[0])
    n_shots = sum(samples[0].values())
    corrected_counters: list[Counter] = []

    for sample in samples:
        ordered_bitstrings = [bin(k)[2:].zfill(n_qubits) for k in range(2**n_qubits)]
        # Array of raw probabilites.
        p_raw = np.array([sample[bs] for bs in ordered_bitstrings]) / n_shots

        if optimization_type == ReadOutOptimization.CONSTRAINED:
            p_corr = constrained_inversion(noise_matrices, p_raw)

        elif optimization_type == ReadOutOptimization.MLE:
            noise_matrices_inv = list(map(matrix_inv, noise_matrices))
            # Compute corrected inverse using matrix inversion and run MLE.
            p_corr = mle_solve(tensor_rank_mult(noise_matrices_inv, p_raw))

        elif optimization_type == ReadOutOptimization.MTHREE:
            Confusion_matrix_subspace = subspace_kron(noise_matrices, p_raw.nonzero())
            # GMRES is best suited for higher dimensional problems
            p_corr = gmres(Confusion_matrix_subspace, p_raw)[0]
            # To ensure that we are not working with negative probabilities
            p_corr = mle_solve(p_corr)

        else:
            raise NotImplementedError(
                f"Requested method {optimization_type} does not match supported protocols."
            )

        corrected_counts = np.rint(p_corr * n_shots).astype(int)
        # Renormalize if total counts differs from n_shots.
        corrected_counts = renormalize_counts(corrected_counts=corrected_counts, n_shots=n_shots)

        assert (
            corrected_counts.sum() == n_shots
        ), f"Corrected counts sum: {corrected_counts.sum()}, n_shots: {n_shots}"
        corrected_counters.append(
            Counter(
                {bs: count for bs, count in zip(ordered_bitstrings, corrected_counts) if count > 0}
            )
        )
    return corrected_counters


def mitigate(model: QuantumModel, options: dict, noise: Noise | None = None) -> list[Counter]:
    if noise is None or noise.protocol != Noise.READOUT:
        if model._noise is None or model._noise.protocol != Noise.READOUT:
            raise ValueError(
                "A Noise.READOUT model must be provided either to .mitigate()"
                " or through the <class QuantumModel>."
            )
        noise = model._noise
    samples = options.get("samples", None)
    if samples is None:
        n_shots = options.get("n_shots", None)
        if n_shots is None:
            raise ValueError("A n_shots option must be provided.")
        samples = model.sample(noise=noise, n_shots=n_shots)
    return mitigation_minimization(noise=noise, options=options, samples=samples)
