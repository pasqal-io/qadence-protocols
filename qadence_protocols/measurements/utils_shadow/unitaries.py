from __future__ import annotations

import torch
from qadence.blocks.block_to_tensor import HMAT, IMAT, SDAGMAT
from qadence.operations import H, SDagger, X, Y, Z

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
UNITARY_TENSOR_adjoint = [unit.adjoint() for unit in UNITARY_TENSOR]

idmat = UNITARY_TENSOR[-1]

HammingMatrix = torch.tensor([[1.0, -0.5], [-0.5, 1.0]], dtype=torch.double)
