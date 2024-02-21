from __future__ import annotations

import itertools
from collections import Counter

import torch
from qadence import QuantumCircuit, QuantumModel, X, chain, kron
from qadence.blocks.utils import unroll_block_with_scaling
from qadence.measurements.samples import compute_expectation
from torch import tensor


def twirl_swap(n_qubits: int, twirl: tuple, samples_twirl: dict) -> dict:
    """
    Applies the corresponding twirl operations (bit flip).

    Achieved by remapping the measurements appropriately
    """
    output = {}
    operand = "".join(map(str, [1 if i in twirl else 0 for i in range(n_qubits)]))
    for key in samples_twirl.keys():
        output[bin(int(key, 2) ^ int(operand, 2))[2:].zfill(n_qubits)] = samples_twirl[key]

    return Counter(output)


def mitigate(
    model: QuantumModel,
    options: dict,
) -> tensor:
    """Corrects for readout errors on expectation values using all possible twirl operations.

    See Equation at the end of page 3
    See Page(2) in https://arxiv.org/pdf/2012.09738.pdf for implementation

    Args:
        noise: Specifies confusion matrix and default error probability
        observable: a list of pauli Z strings specified by index locations

    Returns:
        Mitigated output is returned
    """
    block = model._circuit.original.block
    n_shots = model._measurement.options["n_shots"]

    # Generates a list of all possible X gate string combinations
    # Applied at the end of circuit before measurements are made
    twirls = list(
        itertools.chain.from_iterable(
            [
                list(itertools.combinations(range(block.n_qubits), k))
                for k in range(1, block.n_qubits + 1)
            ]
        )
    )
    # Generate samples for all twirls of circuit
    samples_twirl_num_list = []
    samples_twirl_den_list = []
    for twirl in twirls:
        block_twirl = chain(block, kron(X(i) for i in twirl))

        # Twirl outputs for given circuit (Numerator)
        circ_twirl_num = QuantumCircuit(block_twirl.n_qubits, block_twirl)
        model_twirl_num = QuantumModel(circuit=circ_twirl_num, backend=model._backend_name)
        samples_twirl_num = model_twirl_num.sample(noise=model._noise, n_shots=n_shots)[0]
        samples_twirl_num_list.append(twirl_swap(block_twirl.n_qubits, twirl, samples_twirl_num))

        # Twirl outputs on input state (Denominator)
        circ_twirl_den = QuantumCircuit(block_twirl.n_qubits, kron(X(i) for i in twirl))
        model_twirl_den = QuantumModel(circuit=circ_twirl_den, backend=model._backend_name)
        samples_twirl_den = model_twirl_den.sample(noise=model._noise, n_shots=n_shots)[0]
        samples_twirl_den_list.append(twirl_swap(block_twirl.n_qubits, twirl, samples_twirl_den))

    output_exp = []

    for observable in model._observable:
        expectation: float = 0

        coeffs = torch.tensor(
            [pauli[1] for pauli in unroll_block_with_scaling(observable.original)], dtype=float
        )
        obs_list = [pauli[0] for pauli in unroll_block_with_scaling(observable.original)]

        expectation_num = torch.stack(compute_expectation(obs_list, samples_twirl_num_list))
        expectation_den = torch.stack(compute_expectation(obs_list, samples_twirl_den_list))

        expectation = torch.sum(expectation_num, dim=1) / torch.sum(expectation_den, dim=1)

        output_exp.append(torch.sum(torch.dot(coeffs, expectation)))

    return tensor(output_exp)
