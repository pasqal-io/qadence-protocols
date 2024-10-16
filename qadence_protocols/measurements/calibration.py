from __future__ import annotations

import numpy as np
import torch
from qadence import kron
from qadence.backend import Backend
from qadence.backends.pyqtorch import Backend as PyQBackend
from qadence.blocks.block_to_tensor import XMAT, YMAT, ZMAT
from qadence.circuit import QuantumCircuit
from qadence.engines.differentiable_backend import DifferentiableBackend
from qadence.noise import Noise
from qadence.operations import X, Y, Z
from qadence.types import Endianness

pauli_gates = [X, Y, Z]
pauli_tensors = [XMAT[0], YMAT[0], ZMAT[0]]


def zero_state_calibration(
    n_unitaries: int,
    n_qubits: int,
    n_shots: int = 1,
    backend: Backend | DifferentiableBackend = PyQBackend(),
    noise: Noise | None = None,
    endianness: Endianness = Endianness.BIG,
) -> torch.Tensor:
    """Calculate the calibration coefficients for Robust shadows.

    They correspond to (2 G - 1) / 3 in PRXQuantum.5.030338
    and can be used directly in the robust shadow protocol.

    Args:
        n_unitaries (int): Number of pauli unitary to sample.
        n_qubits (int): Number of qubits
        n_shots (int, optional): Number of shots per circuit.
            Defaults to 1.
        backend (Backend | DifferentiableBackend, optional): Backend to run circuits.
            Defaults to PyQBackend().
        noise (Noise | None, optional): Noise model. Defaults to None.
        endianness (Endianness, optional): Endianness of operations. Defaults to Endianness.BIG.

    Returns:
        torch.Tensor: Calibration coefficients
    """
    unitary_ids = np.random.randint(0, 3, size=(n_unitaries, n_qubits))
    param_values: dict = dict()

    calibrations = torch.zeros(n_qubits, dtype=torch.float64)
    divider = 3.0 * n_shots * n_unitaries
    for i in range(n_unitaries):
        random_unitary = [pauli_gates[unitary_ids[i][qubit]](qubit) for qubit in range(n_qubits)]

        if len(random_unitary) == 1:
            random_unitary_block = random_unitary[0]
        else:
            random_unitary_block = kron(*random_unitary)

        random_circuit = QuantumCircuit(
            n_qubits,
            random_unitary_block,
        )
        conv_circ = backend.circuit(random_circuit)
        samples = backend.sample(
            circuit=conv_circ,
            param_values=param_values,
            n_shots=n_shots,
            state=None,
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
                            pauli_tensors[unitary_ids[i][qubit]][int(bitstring[qubit]), 0]
                            * pauli_tensors[unitary_ids[i][qubit]][int(bitstring[qubit]), 0].conj()
                        )
                        - 1
                        for qubit in range(n_qubits)
                    ]
                )
                / divider
            )

    return calibrations
