from __future__ import annotations

import torch
from qadence.blocks.block_to_tensor import HMAT, IMAT, SDAGMAT
from qadence.operations import H, SDagger, X, Y, Z
from qadence.utils import one_qubit_projector_matrix

pauli_gates = [X, Y, Z]
pauli_rotations = [
    lambda index: H(index),
    lambda index: SDagger(index) * H(index),
    lambda index: None,
]

UNITARY_TENSOR = [
    HMAT.squeeze(dim=0),
    (HMAT @ SDAGMAT).squeeze(dim=0),
    IMAT.squeeze(dim=0),
]
UNITARY_TENSOR_ADJOINT = [unit.adjoint() for unit in UNITARY_TENSOR]

idmat = UNITARY_TENSOR[-1]

hamming_one_qubit = torch.tensor([[1.0, -0.5], [-0.5, 1.0]], dtype=torch.double)


P0_MATRIX = one_qubit_projector_matrix("0")
P1_MATRIX = one_qubit_projector_matrix("1")
