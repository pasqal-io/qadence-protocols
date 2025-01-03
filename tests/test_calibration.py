from __future__ import annotations

import torch
from qadence import NoiseHandler
from qadence.types import NoiseProtocol

from qadence_protocols.measurements.calibration import zero_state_calibration


def test_zero_state_calibration() -> None:
    N = 3
    coeffs = zero_state_calibration(2000, N, 1000)
    assert torch.allclose(coeffs, torch.ones(N) / 3.0, atol=0.1)

    torch.manual_seed(0)
    p = torch.clamp(0.1 + 0.02 * torch.randn(N), min=0, max=1)
    expected_coeffs = 1 - p / 2.0
    noise = NoiseHandler(
        protocol=NoiseProtocol.DIGITAL.DEPOLARIZING,
        options={"error_probability": p[0], "target": 0},
    )

    for i, proba in enumerate(p[1:]):
        noise.digital_depolarizing(options={"error_probability": proba, "target": i + 1})

    coeffs = zero_state_calibration(2000, N, 1000, noise=noise)
    assert torch.allclose((6 * coeffs - 1), expected_coeffs, atol=0.1)
