from __future__ import annotations

import itertools
from functools import reduce

import torch

# import pytest
from qadence import (
    QuantumCircuit,
    QuantumModel,
    chain,
)
from qadence.noise.protocols import Noise
from qadence.operations import X, kron
from qadence.types import BackendName

# from metrics import MIDDLE_ACCEPTANCE


def twirl_swap(n_qubits: int, twirl: tuple, samples_twirl: dict) -> dict:
    output = {}
    operand = "".join(map(str, [1 if i in twirl else 0 for i in range(n_qubits)]))
    for key in samples_twirl.keys():
        output["{0:b}".format(int(key, 2) ^ int(operand, 2))] = samples_twirl[key]

    return output


def compute_exp(n_qubits: int, samples_twirl_list: list, observable: list) -> float:
    out: float = 0
    Z_mat = torch.eye(2)
    Z_mat[1, 1] = -1
    o = reduce(torch.kron, [Z_mat if i in observable else torch.eye(2) for i in range(n_qubits)])

    for samples_twirl in samples_twirl_list:
        shots = sum(samples_twirl.values())
        for key in samples_twirl.keys():
            out += o[int(key, 2), int(key, 2)] * samples_twirl[key] / shots
    return out


def twirl_mitigation(
    n_qubits: int,
    circuit: QuantumCircuit,
    backend: BackendName,
    noise: Noise,
    n_shots: int,
    observable: list,
) -> float:
    ## check if observable is in the Z basis
    ## extend protocol to sum of Z basis observables
    ### by default lets do all twirling and specialize later for user inputs

    twirl_list = list(
        itertools.chain.from_iterable(
            [list(itertools.combinations(range(n_qubits), k)) for k in range(1, n_qubits)]
        )
    )

    ###generate samples for all twirls of circuit
    samples_twirl_num_list = []
    samples_twirl_den_list = []
    for twirl in twirl_list:
        layer = [X(i) for i in twirl]
        block_twirl = chain(circuit.block, kron(*layer))

        circ_twirl_num = QuantumCircuit(block_twirl.n_qubits, block_twirl)
        model_twirl_num = QuantumModel(circuit=circ_twirl_num, backend=BackendName.PYQTORCH)
        samples_twirl_num = model_twirl_num.sample(noise=noise, n_shots=n_shots)[0]

        samples_twirl_num = twirl_swap(n_qubits, twirl, samples_twirl_num)
        samples_twirl_num_list.append(samples_twirl_num)

        circ_twirl_den = QuantumCircuit(block_twirl.n_qubits, kron(*layer))
        model_twirl_den = QuantumModel(circuit=circ_twirl_den, backend=backend)
        samples_twirl_den = model_twirl_den.sample(noise=noise, n_shots=n_shots)[0]

        samples_twirl_den = twirl_swap(n_qubits, twirl, samples_twirl_den)
        samples_twirl_den_list.append(samples_twirl_den)

    expectation: float = 0

    for o in observable:
        expectation_num = compute_exp(n_qubits, samples_twirl_num_list, o)
        expectation_den = compute_exp(n_qubits, samples_twirl_den_list, o)
        expectation += expectation_num / expectation_den

    return expectation
