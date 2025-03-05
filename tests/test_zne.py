from __future__ import annotations

import numpy as np
import pytest
import torch
from qadence import (
    AbstractBlock,
    AnalogRX,
    AnalogRZ,
    QuantumCircuit,
    QuantumModel,
    chain,
    hamiltonian_factory,
)
from qadence.noise.protocols import NoiseHandler
from qadence.operations import RY, Z, entangle
from qadence.types import PI, BackendName, DiffMode, NoiseProtocol
from torch import Tensor

from qadence_protocols.mitigations.protocols import Mitigations


@pytest.mark.parametrize(
    "analog_block, observable, noise_type",
    [
        (
            chain(AnalogRX(PI / 2.0), AnalogRZ(PI)),
            [Z(0) + Z(1)],
            NoiseProtocol.ANALOG.DEPOLARIZING,
        ),
        (
            # Hardcoded time and angle for Bell state preparation.
            chain(
                entangle(383, qubit_support=(0, 1)),
                RY(0, 3.0 * PI / 2.0),
            ),
            [hamiltonian_factory(2, detuning=Z)],
            NoiseProtocol.ANALOG.DEPHASING,
        ),
    ],
)
def test_analog_zne_with_noise_levels(
    analog_block: AbstractBlock,
    observable: AbstractBlock,
    noise_type: NoiseProtocol.ANALOG,
) -> None:
    circuit = QuantumCircuit(2, analog_block)
    model = QuantumModel(
        circuit=circuit, observable=observable, backend=BackendName.PULSER, diff_mode=DiffMode.GPSR
    )
    noise_probs = np.linspace(0.1, 0.5, 8)
    options = {"noise_probs": noise_probs}
    noise = NoiseHandler(protocol=noise_type, options=options)
    mitigate = Mitigations(protocol=Mitigations.ANALOG_ZNE)
    exact_expectation = model.expectation()
    mitigated_expectation = mitigate(model=model, noise=noise)
    assert torch.allclose(mitigated_expectation, exact_expectation, atol=1.0e-2)


# FIXME: Consider a stretchable replacement for entangle.
@pytest.mark.parametrize(
    "analog_block, observable, noise_probs, noise_type, param_values",
    [
        (
            chain(AnalogRX(PI / 2.0), AnalogRZ(PI)),
            [Z(0) + Z(1)],
            [0.1],
            NoiseProtocol.ANALOG.DEPOLARIZING,
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
        #     NoiseProtocol.ANALOG.DEPHASING,
        #     {"t": torch.tensor([1.0])},
        # ),
    ],
)
def test_analog_zne_with_pulse_stretching(
    analog_block: AbstractBlock,
    observable: AbstractBlock,
    noise_probs: Tensor,
    noise_type: NoiseProtocol.ANALOG,
    param_values: dict,
) -> None:
    circuit = QuantumCircuit(2, analog_block)
    model = QuantumModel(
        circuit=circuit, observable=observable, backend=BackendName.PULSER, diff_mode=DiffMode.GPSR
    )
    options = {"noise_probs": noise_probs}
    noise = NoiseHandler(protocol=noise_type, options=options)
    options = {"stretches": torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0])}
    mitigate = Mitigations(protocol=Mitigations.ANALOG_ZNE, options=options)
    mitigated_expectation = mitigate(model=model, noise=noise, param_values=param_values)
    exact_expectation = model.expectation(values=param_values)
    assert torch.allclose(mitigated_expectation, exact_expectation, atol=2.0e-1)


@pytest.mark.parametrize(
    "analog_block, observable, noise_type",
    [
        (
            chain(AnalogRX(PI / 2.0), AnalogRZ(PI)),
            [Z(0) + Z(1)],
            NoiseProtocol.ANALOG.DEPOLARIZING,
        ),
        (
            # Hardcoded time and angle for Bell state preparation.
            chain(
                entangle(383, qubit_support=(0, 1)),
                RY(0, 3.0 * PI / 2.0),
            ),
            [hamiltonian_factory(2, detuning=Z)],
            NoiseProtocol.ANALOG.DEPHASING,
        ),
    ],
)
def test_analog_zne_with_noise_levels_exp(
    analog_block: AbstractBlock,
    observable: AbstractBlock,
    noise_type: NoiseProtocol.ANALOG,
) -> None:
    circuit = QuantumCircuit(2, analog_block)
    model = QuantumModel(
        circuit=circuit, observable=observable, backend=BackendName.PULSER, diff_mode=DiffMode.GPSR
    )
    noise_probs = np.linspace(0.1, 0.5, 8)
    options = {"noise_probs": noise_probs, "zne_type": "exp"}
    noise = NoiseHandler(protocol=noise_type, options=options)
    mitigate = Mitigations(protocol=Mitigations.ANALOG_ZNE)
    exact_expectation = model.expectation()
    mitigated_expectation = mitigate(model=model, noise=noise)
    assert torch.allclose(mitigated_expectation, exact_expectation, atol=1.0e-2)


@pytest.mark.parametrize(
    "analog_block, observable, noise_probs, noise_type, param_values",
    [
        (
            chain(AnalogRX(PI / 2.0), AnalogRZ(PI)),
            [Z(0) + Z(1)],
            [0.1],
            NoiseProtocol.ANALOG.DEPOLARIZING,
            {},
        ),
    ],
)
def test_analog_zne_with_pulse_stretching_exp(
    analog_block: AbstractBlock,
    observable: AbstractBlock,
    noise_probs: Tensor,
    noise_type: NoiseProtocol.ANALOG,
    param_values: dict,
) -> None:
    circuit = QuantumCircuit(2, analog_block)
    model = QuantumModel(
        circuit=circuit, observable=observable, backend=BackendName.PULSER, diff_mode=DiffMode.GPSR
    )
    options = {"noise_probs": noise_probs}
    noise = NoiseHandler(protocol=noise_type, options=options)
    options = {"stretches": torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0]), "zne_type": "exp"}
    mitigate = Mitigations(protocol=Mitigations.ANALOG_ZNE, options=options)
    mitigated_expectation = mitigate(model=model, noise=noise, param_values=param_values)
    exact_expectation = model.expectation(values=param_values)
    assert torch.allclose(mitigated_expectation, exact_expectation, atol=2.0e-1)
