from __future__ import annotations

import itertools
from collections import Counter

import torch

# import pytest
from qadence import QuantumCircuit, QuantumModel, chain
from qadence.blocks.utils import unroll_block_with_scaling
from qadence.operations import RX, RY, X, kron
from torch import tensor

from qadence.measurements import Measurements


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
    block = model._circuit.abstract.block

    twirls = list(
        itertools.chain.from_iterable(
            [
                list(itertools.combinations(range(block.n_qubits), k))
                for k in range(1, block.n_qubits)
            ]
        )
    )

    # Generate samples for all twirls of circuit
    samples_twirl_num_list = []
    samples_twirl_den_list = []
    for twirl in twirls:
        layer = [X(i) for i in twirl]
        block_twirl = chain(block, kron(*layer))

        circ_twirl_num = QuantumCircuit(block_twirl.n_qubits, block_twirl)
        model_twirl_num = QuantumModel(circuit=circ_twirl_num, backend=model.backend.backend.name)
        samples_twirl_num = model_twirl_num.sample(noise=model._noise)[0]
        samples_twirl_num_list.append(twirl_swap(block_twirl.n_qubits, twirl, samples_twirl_num))

        circ_twirl_den = QuantumCircuit(block_twirl.n_qubits, kron(*layer))
        model_twirl_den = QuantumModel(circuit=circ_twirl_den, backend=model.backend.backend.name)
        samples_twirl_den = model_twirl_den.sample(noise=model._noise)[0]

        samples_twirl_den_list.append(twirl_swap(block_twirl.n_qubits, twirl, samples_twirl_den))

    # I am creating a dummy circuit that has nothing to do with the protocol
    # This needs to be changed
    dummy = QuantumCircuit(
        2,
        chain(
            kron(RX(0, 0), RY(1, 0)),
            kron(RX(0, 0), RY(1, 0)),
        ),
    )

    output_exp = []
    for observable in model._observable:
        expectation: float = 0

        for pauli in unroll_block_with_scaling(observable.original):
            sample_measurement_num = Measurements(
                protocol=Measurements.SAMPLES, options={"samples": samples_twirl_num_list}
            )
            expectation_num = torch.sum(
                QuantumModel(circuit=dummy, observable=pauli[0]).expectation(
                    measurement=sample_measurement_num
                )
            )

            sample_measurement_den = Measurements(
                protocol=Measurements.SAMPLES, options={"samples": samples_twirl_den_list}
            )
            expectation_den = torch.sum(
                QuantumModel(circuit=dummy, observable=pauli[0]).expectation(
                    measurement=sample_measurement_den
                )
            )
            expectation += expectation_num / expectation_den * torch.tensor(pauli[1], dtype=float)

        output_exp.append(expectation)

    return torch.tensor(output_exp)
