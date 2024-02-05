from __future__ import annotations

import itertools

import torch

# import pytest
from qadence import QuantumCircuit, QuantumModel, block_to_tensor, chain
from qadence.noise.protocols import Noise
from qadence.operations import I, X, Z, kron
from qadence.types import BackendName
from torch import tensor


def twirl_swap(n_qubits: int, twirl: tuple, samples_twirl: dict) -> dict:
    output = {}
    operand = "".join(map(str, [1 if i in twirl else 0 for i in range(n_qubits)]))
    for key in samples_twirl.keys():
        output["{0:b}".format(int(key, 2) ^ int(operand, 2))] = samples_twirl[key]

    return output


def compute_exp(n_qubits: int, samples_twirl: list, observable: list) -> tensor:
    out = 0
    o = block_to_tensor(
        kron(*[Z(i) if i in observable else I(i) for i in range(n_qubits)])
    ).squeeze()

    for sample_twirl in samples_twirl:
        shots = sum(sample_twirl.values())
        for key in sample_twirl.keys():
            out += o[int(key, 2), int(key, 2)] * sample_twirl[key] / shots
    return out


def mitigate(
    n_qubits: int,
    circuit: QuantumCircuit,
    backend: BackendName,
    noise: Noise,
    n_shots: int,
    observable: list,
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

    twirls = list(
        itertools.chain.from_iterable(
            [list(itertools.combinations(range(n_qubits), k)) for k in range(1, n_qubits)]
        )
    )

    # Generate samples for all twirls of circuit
    samples_twirl_num_list = []
    samples_twirl_den_list = []
    for twirl in twirls:
        layer = [X(i) for i in twirl]
        block_twirl = chain(circuit.block, kron(*layer))

        circ_twirl_num = QuantumCircuit(block_twirl.n_qubits, block_twirl)
        model_twirl_num = QuantumModel(circuit=circ_twirl_num, backend=backend)
        samples_twirl_num = model_twirl_num.sample(noise=noise, n_shots=n_shots)[0]
        samples_twirl_num_list.append(twirl_swap(n_qubits, twirl, samples_twirl_num))

        circ_twirl_den = QuantumCircuit(block_twirl.n_qubits, kron(*layer))
        model_twirl_den = QuantumModel(circuit=circ_twirl_den, backend=backend)
        samples_twirl_den = model_twirl_den.sample(noise=noise, n_shots=n_shots)[0]
        samples_twirl_den_list.append(twirl_swap(n_qubits, twirl, samples_twirl_den))

    expectation: float = 0

    for o in observable:
        expectation_num = compute_exp(n_qubits, samples_twirl_num_list, o)
        expectation_den = compute_exp(n_qubits, samples_twirl_den_list, o)
        expectation += expectation_num / expectation_den

    return torch.real(expectation)
