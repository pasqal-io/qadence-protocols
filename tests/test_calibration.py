from __future__ import annotations

import torch
from qadence import NoiseHandler
from qadence.types import NoiseProtocol

from qadence_protocols.measurements.calibration import zero_state_calibration


def test_zero_state_calibration() -> None:
    coeffs = zero_state_calibration(10, 3, 1000)
    assert torch.allclose(coeffs, torch.ones(3) / 3.0)

    error_probability = 0.1
    noise = NoiseHandler(
        protocol=NoiseProtocol.READOUT.INDEPENDENT, options={"error_probability": error_probability}
    )
    coeffs = zero_state_calibration(10, 3, 1000, noise=noise)
    assert torch.allclose((3 * coeffs + 1) / 2.0, torch.ones(3) - error_probability, atol=0.1)
