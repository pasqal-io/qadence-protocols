from __future__ import annotations

from collections import Counter

import numpy as np
import numpy.typing as npt
from numpy.linalg import inv, matrix_rank, pinv
from pyqtorch.noise import CorrelatedReadoutNoise
from qadence import QuantumModel
from qadence.backends.pyqtorch.convert_ops import convert_readout_noise
from qadence.logger import get_logger
from qadence.noise.protocols import NoiseHandler
from qadence.types import NoiseProtocol
from scipy.linalg import norm
from scipy.optimize import LinearConstraint, minimize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import gmres
from torch import Tensor

from qadence_protocols.types import ReadOutOptimization

logger = get_logger(__name__)


def normalized_subspace_kron(noise_matrices: npt.NDArrary, subspace: npt.NDArray) -> npt.NDArray:
    """
    Compute a specified tensor producted subspace of index locations.

    Args:
        noise: Specifies an array of noise_matrices acting indpedent qubits
        subspace: List of index locations that defines the subspace for computation

    Returns:
        A sparse matrix construced from the tensorproduct of noise matrices in the subspace
    """

    n_qubits = len(noise_matrices)
    conf_matrix = csr_matrix((2**n_qubits, 2**n_qubits))

    for j in subspace:
        for i in subspace:
            bin_i = bin(i)[2:].zfill(n_qubits)
            bin_j = bin(j)[2:].zfill(n_qubits)

            # Manually computing the entries of tensor product for only the subspace
            conf_matrix[i, j] = np.prod(
                [noise_matrices[k][int(bin_i[k])][int(bin_j[k])] for k in range(n_qubits)]
            )

        conf_matrix[:, j] /= np.sum(conf_matrix[:, j])

    return conf_matrix


def tensor_rank_mult(qubit_ops: npt.NDArray, prob_vect: npt.NDArray) -> npt.NDArray:
    """
    Fast multiplication of single qubit operators on a probability vector.

    Similar to how gate operations are implemented on PyqTorch
    Needs to be replaced.
    """

    n_qubits = int(np.log2(len(prob_vect)))

    # Reshape probability vector into a rank N tensor
    prob_vect_t = prob_vect.reshape(n_qubits * [2]).transpose()

    # Contract each tensor index (qubit) with the inverse of the single-qubit
    for i in range(n_qubits):
        prob_vect_t = np.tensordot(qubit_ops[n_qubits - 1 - i], prob_vect_t, axes=(1, i))

    # Obtain corrected measurements by shaping back into a vector
    return prob_vect_t.reshape(2**n_qubits)


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
        # If neg_sum cannot be distributed among other probabilities, continue to accumulate
        if p_sort[i] + neg_sum / (len(p_sort) - i) < 0:
            neg_sum += p_sort[i]
            p_sort[i] = 0
        # Set breakpoint to current index
        else:
            breakpoint = i
            break
    # Number of entries to which i can distribute(includes breakpoint)
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


def majority_vote(noise_matrices: npt.NDArray, p_raw: npt.NDArray) -> npt.NDArray:
    """
    Compute the bitstring with the highest probability of occurrrence via voting.

    For implementation, see Equation (6) and Page 4 in https://arxiv.org/pdf/2402.11830.pdf
    """
    n_qubits = len(noise_matrices)
    output = 0

    for i in range(n_qubits):
        p_raw_resize = p_raw.reshape([2] * n_qubits)
        transposed_axes = [i] + list(range(0, i)) + list(range(i + 1, n_qubits))
        p_raw_resize = np.transpose(p_raw_resize, axes=transposed_axes).reshape(
            2, 2**n_qubits // 2
        )
        probs = np.sum(p_raw_resize, axis=1)

        # Given the output to be 0, the probability of observed outcomes
        prob_zero = noise_matrices[i][1][0] ** probs[1] * noise_matrices[i][0][0] ** probs[0]
        # Given the output to be 0, the probability of observed outcomes
        prob_one = noise_matrices[i][1][1] ** probs[1] * noise_matrices[i][0][1] ** probs[0]

        if prob_one > prob_zero:
            output += 2 ** (n_qubits - 1 - i)

    p_corr = np.zeros(2**n_qubits)
    p_corr[output] = 1

    return p_corr


def mitigation_minimization(
    noise: NoiseHandler,
    options: dict,
    samples: list[Counter],
) -> list[Counter]:
    """Minimize a correction matrix subjected to stochasticity constraints.

    See Equation (5) in https://arxiv.org/pdf/2001.09980.pdf.
    See Page(3) in https://arxiv.org/pdf/1106.5458.pdf for MLE implementation
    See Equation (5) in https://arxiv.org/pdf/2108.12518.pdf for MTHREE implementation
    (matrix free measurement mitigation)
    This method is reserved for implementations of over 20 qubits
    See Equation (6) and Page 4 in https://arxiv.org/pdf/2402.11830.pdf for MV implementation
    This method is reserved for algorithms with a single bit string output
    See page (5) for extension to more than one solution (inefficient)

    Args:
        noise: Specifies confusion matrix and default error probability
        mitigation: Selects additional mitigation options based on noise choice.
        For readout we have the following mitigation options for optimization
        1. constrained 2. mle (Default) mle 3. mthree 4. majority vote (mv)
        samples: List of samples to be mitigated

    Returns:
        Mitigated counts computed by the algorithm
    """
    n_qubits = len(list(samples[0].keys())[0])
    readout_noise = convert_readout_noise(n_qubits, noise)
    if readout_noise is None or isinstance(readout_noise, CorrelatedReadoutNoise):
        raise ValueError("Specify a noise source of type NoiseProtocol.READOUT.INDEPENDENT.")
    n_shots = sum(samples[0].values())
    noise_matrices = readout_noise.confusion_matrix
    if readout_noise._compute_confusion:
        noise_matrices = readout_noise.create_noise_matrix(n_shots)
        noise_matrices = readout_noise.confusion_matrix
    noise_matrices = noise_matrices.numpy()

    optimization_type = options.get("optimization_type", ReadOutOptimization.MLE)
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
            confusion_matrix_subspace = normalized_subspace_kron(noise_matrices, p_raw.nonzero()[0])
            # GMRES (Generalized minimal residual) for linear equations in higher dimension
            p_corr, exit_code = gmres(confusion_matrix_subspace, p_raw)

            if exit_code != 0:
                logger.warning(f"GMRES did not converge, exited with code {exit_code}")

            # To ensure that we are not working with negative probabilities
            p_corr = mle_solve(p_corr)

        elif optimization_type == ReadOutOptimization.MAJ_VOTE:
            # Majority vote : lets return just that one bit string with all the counts for now
            p_corr = majority_vote(noise_matrices, p_raw)

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


def mitigate(
    model: QuantumModel,
    options: dict,
    noise: NoiseHandler | None = None,
    param_values: dict[str, Tensor] = dict(),
) -> list[Counter]:
    if noise is None or noise.filter(NoiseProtocol.READOUT) is None:
        if model._noise is None or model._noise.filter(NoiseProtocol.READOUT) is None:
            raise ValueError(
                "A NoiseProtocol.READOUT model must be provided either to .mitigate()"
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
