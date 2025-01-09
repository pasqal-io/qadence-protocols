from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from qadence import NoiseHandler, NoiseProtocol
from qadence.backend import Backend
from qadence.backends.pyqtorch import Backend as PyQBackend
from qadence.circuit import QuantumCircuit
from qadence.engines.differentiable_backend import DifferentiableBackend
from qadence.operations import I
from qadence.types import Endianness
from torch import Tensor

from qadence_protocols.measurements.utils_shadow.data_acquisition import extract_operators
from qadence_protocols.utils_trace import partial_trace


def _get_noiseless_probas(
    n_qubits: int,
    rotations: list,
    backend: Backend | DifferentiableBackend = PyQBackend(),
    endianness: Endianness = Endianness.BIG,
) -> Tensor:
    """Get noiseless probas for a zero circuit with rotations for the zero state calibration.

    Args:
        n_qubits (int): Number of qubits.
        rotations (list): Sampled rotations for calibration.
        backend (Backend | DifferentiableBackend, optional): Backend to run circuits on.
            Defaults to PyQBackend().
        endianness (Endianness, optional): Endianness of operations. Defaults to Endianness.BIG.

    Returns:
        Tensor: The probabilities per qubit for each rotation.
    """
    zero_circ = backend.circuit(QuantumCircuit(n_qubits))
    noiseless_probas = torch.zeros((len(rotations), n_qubits, 2))
    for r, rot in enumerate(rotations):
        wave_fct = backend.run(
            backend.circuit(QuantumCircuit(n_qubits, rot)) if rot else zero_circ,
            endianness=endianness,
        )
        for i in range(n_qubits):
            noiseless_probas[r][i] = torch.diagonal(
                partial_trace(wave_fct, [i]), dim1=1, dim2=2
            ).real.squeeze()
    return noiseless_probas


def _samples_frequencies(
    n_qubits: int,
    samples: Counter,
    endianness: Endianness = Endianness.BIG,
) -> Tensor:
    """Get the probabilities of 0,1 for each qubit from n-bits samples.

    Args:
        n_qubits (int): Number of qubits.
        samples (Counter): Samples obtained from a circuit.
        endianness (Endianness, optional): Endianness of operations. Defaults to Endianness.BIG.

    Returns:
        Tensor: _description_
    """
    freqs = torch.zeros((n_qubits, 2))
    for bitstring, freq in samples.items():
        for qubit in range(n_qubits):
            kj = (
                int(bitstring[qubit], 2)
                if endianness == Endianness.BIG
                else int(bitstring[::-1][qubit], 2)
            )
            freqs[qubit][kj] += freq
    return freqs


def zero_state_calibration(
    n_unitaries: int,
    n_qubits: int,
    n_shots: int = 1,
    backend: Backend | DifferentiableBackend = PyQBackend(),
    noise: NoiseHandler | None = None,
    endianness: Endianness = Endianness.BIG,
) -> torch.Tensor:
    """Calculate the calibration coefficients for Robust shadows.

    They correspond to (2 G - 1) / 3 in PRXQuantum.5.030338
    (also https://arxiv.org/html/2307.16882v2)
    and can be used directly in the robust shadow protocol.

    Args:
        n_unitaries (int): Number of pauli unitary to sample.
        n_qubits (int): Number of qubits
        n_shots (int, optional): Number of shots per circuit.
            Defaults to 1.
        backend (Backend | DifferentiableBackend, optional): Backend to run circuits.
            Defaults to PyQBackend().
        noise (NoiseHandler | None, optional): NoiseHandler model. Defaults to None.
        endianness (Endianness, optional): Endianness of operations. Defaults to Endianness.BIG.

    Returns:
        torch.Tensor: Calibration coefficients
    """
    unitary_ids = np.random.randint(0, 3, size=(n_unitaries, n_qubits))
    param_values: dict = dict()

    # get measurement rotations
    all_rotations = extract_operators(unitary_ids, n_qubits)

    # set an input state depending on digital noise with target options
    noisy_zero_circ = QuantumCircuit(n_qubits)
    if noise is not None:
        digital_part = noise.filter(NoiseProtocol.DIGITAL)
        if digital_part is not None:
            noisy_identities = list()
            for proto, options in zip(digital_part.protocol, digital_part.options):
                target = options.get("target", None)
                if target is not None:
                    noisy_identities.append(I(target=target, noise=NoiseHandler(proto, options)))
                else:
                    for target in range(n_qubits):
                        noisy_identities.append(
                            I(target=target, noise=NoiseHandler(proto, options))
                        )
            noisy_zero_circ = QuantumCircuit(n_qubits, *noisy_identities)

    all_circuits = [
        QuantumCircuit(n_qubits, noisy_zero_circ.block, rots) if rots else noisy_zero_circ
        for rots in all_rotations
    ]

    noiseless_probas = _get_noiseless_probas(n_qubits, all_rotations, backend)

    estimated_probas = list()
    for i in range(n_unitaries):
        conv_circ = backend.circuit(all_circuits[i])
        samples = backend.sample(
            circuit=conv_circ,
            param_values=param_values,
            n_shots=n_shots,
            noise=noise.filter(NoiseProtocol.READOUT) if noise is not None else None,
            endianness=endianness,
        )
        estimated_probas.append(_samples_frequencies(n_qubits, samples[0], endianness) / n_shots)
    estimated_probas = torch.stack(estimated_probas)

    calibrations = torch.sum(
        (
            3.0 * torch.einsum("nij,nij->ni", estimated_probas - noiseless_probas, noiseless_probas)
            + 1.0
        )
        / n_unitaries,
        axis=0,
    )
    return (calibrations + 1) / 6.0
