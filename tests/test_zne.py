from __future__ import annotations

import pytest
import torch
from qadence import (
    AbstractBlock,
    AnalogRX,
    AnalogRZ,
    Mitigations,
    QuantumCircuit,
    QuantumModel,
    chain,
    entangle,
    hamiltonian_factory,
)
from qadence.noise.protocols import Noise
from qadence.operations import RY, Z
from qadence.types import PI, BackendName, DiffMode
from torch import Tensor


@pytest.mark.parametrize(
    "analog_block, observable, noise_probs, noise_type",
    [
        (
            chain(AnalogRX(PI / 2.0), AnalogRZ(PI)),
            [Z(0) + Z(1)],
            torch.linspace(0.1, 0.5, 8),
            Noise.DEPOLARIZING,
        ),
        (
            # Hardcoded time and angle for Bell state preparation.
            chain(
                entangle(383, qubit_support=(0, 1)),
                RY(0, 3.0 * PI / 2.0),
            ),
            [hamiltonian_factory(2, detuning=Z)],
            torch.linspace(0.1, 0.5, 8),
            Noise.DEPHASING,
        ),
    ],
)
def test_analog_zne_with_noise_levels(
    analog_block: AbstractBlock, observable: AbstractBlock, noise_probs: Tensor, noise_type: str
) -> None:
    circuit = QuantumCircuit(2, analog_block)
    model = QuantumModel(
        circuit=circuit, observable=observable, backend=BackendName.PULSER, diff_mode=DiffMode.GPSR
    )
    options = {"noise_probs": noise_probs}
    noise = Noise(protocol=noise_type, options=options)
    mitigation = Mitigations(protocol=Mitigations.ANALOG_ZNE)
    exact_expectation = model.expectation()
    mitigated_expectation = model.expectation(noise=noise, mitigation=mitigation)
    assert torch.allclose(mitigated_expectation, exact_expectation, atol=1.0e-2)


# FIXME: Consider a stretchable replacement for entangle.
@pytest.mark.parametrize(
    "analog_block, observable, noise_probs, noise_type, param_values",
    [
        (
            chain(AnalogRX(PI / 2.0), AnalogRZ(PI)),
            [Z(0) + Z(1)],
            torch.tensor([0.1]),
            Noise.DEPOLARIZING,
            {},
        ),
        # (
        #     # Parameter time and harcoded angle for Bell state preparation.
        #     chain(
        #         entangle("t", qubit_support=(0, 1)),
        #         RY(0, 3.0 * PI / 2.0),
        #     ),
        #     [hamiltonian_factory(2, detuning=Z)],
        #     torch.tensor([0.1]),
        #     Noise.DEPHASING,
        #     {"t": torch.tensor([1.0])},
        # ),
    ],
)
def test_analog_zne_with_pulse_stretching(
    analog_block: AbstractBlock,
    observable: AbstractBlock,
    noise_probs: Tensor,
    noise_type: str,
    param_values: dict,
) -> None:
    circuit = QuantumCircuit(2, analog_block)
    model = QuantumModel(
        circuit=circuit, observable=observable, backend=BackendName.PULSER, diff_mode=DiffMode.GPSR
    )
    options = {"noise_probs": noise_probs}
    noise = Noise(protocol=noise_type, options=options)
    options = {"stretches": torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0])}
    mitigation = Mitigations(protocol=Mitigations.ANALOG_ZNE, options=options)
    mitigated_expectation = model.expectation(
        values=param_values, noise=noise, mitigation=mitigation
    )
    exact_expectation = model.expectation(values=param_values)
    assert torch.allclose(mitigated_expectation, exact_expectation, atol=2.0e-1)
