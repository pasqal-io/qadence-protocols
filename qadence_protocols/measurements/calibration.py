from __future__ import annotations

import torch
from torch import Tensor

from qadence import kron, QuantumCircuit, QuantumModel, MatrixBlock

def random_operation(qubit: int) -> MatrixBlock:
    """Create a random unitary for a calibration circuit.

    Args:
        qubit (int): Qubit it acts on

    Returns:
        MatrixBlock: Operation for circuit
    """
    U = (torch.randn(2,2) + 1j * torch.randn(2,2)) / torch.sqrt(2.0)
    Q, R = torch.linalg.qr(U)

    d = torch.diagonal(R)
    d = d / torch.abs(d)
    U = torch.multiply(Q,d,Q)
    return MatrixBlock(U, qubit)

def calibration_circuits(N: int, n_qubits: int) -> list[QuantumModel]:
    """Create calibration circuits.

    Args:
        N (int): Number of circuits
        n_qubits (int): Number of qubits

    Returns:
        list[QuantumModel]: List of calibration circuits.
    """

    return [QuantumModel(circuit=QuantumCircuit(n_qubits, kron([random_operation(i) for i in range(n_qubits)]))) for _ in range(N)]

