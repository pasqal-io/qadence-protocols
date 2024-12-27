from __future__ import annotations

import numpy as np
import torch
from qadence.backend import Backend
from qadence.backends.pyqtorch import Backend as PyQBackend
from qadence.circuit import QuantumCircuit
from qadence.engines.differentiable_backend import DifferentiableBackend
from qadence.noise import NoiseHandler, NoiseProtocol
from qadence.operations import I
from qadence.types import Endianness

from qadence_protocols.measurements.utils_shadow.data_acquisition import extract_operators
from qadence_protocols.measurements.utils_shadow.unitaries import (
    UNITARY_TENSOR,
    UNITARY_TENSOR_ADJOINT,
)


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

    calibrations = torch.zeros(n_qubits, dtype=torch.float64)
    divider = 3.0 * n_shots * n_unitaries

    all_rotations = extract_operators(unitary_ids, n_qubits)
    all_rotations = [
        QuantumCircuit(n_qubits, rots) if rots else QuantumCircuit(n_qubits)
        for rots in all_rotations
    ]

    # set an input state depending on digital noise with target options
    state = None
    if noise is not None:
        digital_part = noise.filter(NoiseProtocol.DIGITAL)
        if digital_part is not None:
            noisy_identities = list()
            for proto, options in zip(digital_part.protocols, digital_part.options):
                target = options.get("target", None)
                if target:
                    noisy_identities.append(I(target, noise=NoiseProtocol(proto, options)))

            state = QuantumCircuit(n_qubits, noisy_identities)

    for i in range(n_unitaries):
        conv_circ = backend.circuit(all_rotations[i])
        samples = backend.sample(
            circuit=conv_circ,
            param_values=param_values,
            n_shots=n_shots,
            state=state,
            noise=noise,
            endianness=endianness,
        )[0]

        for bitstring, freq in samples.items():
            calibrations += (
                freq
                * torch.tensor(
                    [
                        2.0
                        * torch.real(
                            UNITARY_TENSOR[unitary_ids[i][qubit]][int(bitstring[qubit]), 0]
                            * UNITARY_TENSOR_ADJOINT[unitary_ids[i][qubit]][
                                int(bitstring[qubit]), 0
                            ]
                        )
                        - 1
                        for qubit in range(n_qubits)
                    ]
                )
                / divider
            )

    return calibrations
