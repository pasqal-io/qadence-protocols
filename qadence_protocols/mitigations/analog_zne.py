from __future__ import annotations

import numpy as np
import torch
from qadence import BackendName, QuantumModel
from qadence.backends.api import backend_factory
from qadence.backends.pulser.backend import Backend
from qadence.blocks.abstract import AbstractBlock
from qadence.blocks.analog import ConstantAnalogRotation, InteractionBlock
from qadence.circuit import QuantumCircuit
from qadence.noise import NoiseHandler
from qadence.operations import AnalogRot
from qadence.transpile import apply_fn_to_blocks
from qadence.types import NoiseProtocol
from qadence.utils import Endianness
from torch import Tensor

supported_noise_models = [NoiseProtocol.ANALOG.DEPOLARIZING, NoiseProtocol.ANALOG.DEPHASING]


def zne(noise_levels: Tensor, zne_datasets: list[list]) -> Tensor:
    poly_fits = []
    for dataset in zne_datasets:  # Looping over batched observables.
        poly_fit = np.poly1d(np.polyfit(noise_levels, dataset, len(noise_levels) - 1))
        # Return the zero-noise extrapolated value.
        poly_fits.append(poly_fit(0.0))
    return torch.tensor(poly_fits)


def pulse_experiment(
    backend: Backend,
    circuit: QuantumCircuit,
    observable: list[AbstractBlock],
    param_values: dict[str, Tensor],
    noise: NoiseHandler,
    stretches: Tensor,
    endianness: Endianness,
    state: Tensor | None = None,
) -> Tensor:
    def mutate_params(block: AbstractBlock, stretch: float) -> AbstractBlock:
        """Closure to retrieve and stretch analog parameters."""
        # Check for stretchable analog block.
        if isinstance(block, (ConstantAnalogRotation, InteractionBlock)):
            stretched_duration = block.parameters.duration * stretch
            stretched_omega = block.parameters.omega / stretch
            stretched_delta = block.parameters.delta / stretch
            # The Hamiltonian scaling has no effect on the phase parameter
            phase = block.parameters.phase
            qubit_support = block.qubit_support
            return AnalogRot(
                duration=stretched_duration,
                omega=stretched_omega,
                delta=stretched_delta,
                phase=phase,
                qubit_support=qubit_support,
            )
        return block

    converted_observables = [backend.observable(obs) for obs in observable]
    noisy_density_matrices: list = []
    for stretch in stretches:
        # FIXME: Iterating through the circuit for every stretch
        # and rebuilding the block leaves is inefficient.
        # Best to retrieve the parameters once
        # and rebuild the blocks.
        stre = stretch.item()
        block = apply_fn_to_blocks(circuit.block, mutate_params, stre)
        stretched_register = circuit.register.rescale_coords(stre)
        stretched_circuit = QuantumCircuit(stretched_register, block)
        converted_circuit = backend.circuit(stretched_circuit)
        noisy_density_matrices.append(
            # Contain a single experiment result for the stretch.
            backend.run(
                converted_circuit,
                param_values=param_values,
                state=state,
                noise=noise,
                endianness=endianness,
            )[0]
        )
    support = sorted(list(circuit.register.support))
    # It is not possible to use the expectation method as the pulse stretching concerns the circuit.
    res_list = [
        [
            obs.native(dm, param_values, qubit_support=support, noise=noise)
            for dm in noisy_density_matrices
        ]
        for obs in converted_observables
    ]
    res = torch.stack([torch.transpose(torch.stack(res), 0, -1) for res in res_list])
    res = res if len(res.shape) > 0 else res.reshape(1)

    # Zero-noise extrapolate.
    extrapolated_exp_values = zne(
        noise_levels=stretches,
        zne_datasets=res.real,
    )
    return extrapolated_exp_values


def noise_level_experiment(
    backend: Backend,
    circuit: QuantumCircuit,
    observable: list[AbstractBlock],
    param_values: dict[str, Tensor],
    noise: NoiseHandler,
    endianness: Endianness,
    state: Tensor | None = None,
) -> Tensor:
    noise_probs = noise.options[-1].get("noise_probs")
    converted_circuit = backend.circuit(circuit)
    converted_observables = [backend.observable(obs) for obs in observable]
    zne_datasets = backend.expectation(
        converted_circuit,
        converted_observables,
        param_values=param_values,
        state=state,
        noise=noise,
        endianness=endianness,
    )
    return zne(noise_levels=noise_probs, zne_datasets=zne_datasets)


def analog_zne(
    model: QuantumModel,
    options: dict,
    noise: NoiseHandler,
    param_values: dict[str, Tensor],
    state: Tensor | None = None,
    endianness: Endianness = Endianness.BIG,
) -> Tensor:
    if model._backend_name != BackendName.PULSER:
        raise ValueError("Only BackendName.PULSER supports analog simulations.")
    backend = backend_factory(backend=model._backend_name, diff_mode=None)
    stretches = options.get("stretches", None)
    if stretches is not None:
        extrapolated_exp_values = pulse_experiment(
            backend=backend,
            circuit=model._circuit.original,
            observable=[obs.original for obs in model._observable],
            param_values=param_values,
            noise=noise,
            stretches=stretches,
            endianness=endianness,
            state=state,
        )
    else:
        extrapolated_exp_values = noise_level_experiment(
            backend=backend,
            circuit=model._circuit.original,
            observable=[obs.original for obs in model._observable],
            param_values=param_values,
            noise=noise,
            endianness=endianness,
            state=state,
        )
    return extrapolated_exp_values


def mitigate(
    model: QuantumModel,
    options: dict,
    noise: NoiseHandler | None = None,
    param_values: dict[str, Tensor] = dict(),
) -> Tensor:
    if noise is None or noise.protocol[-1] not in supported_noise_models:
        if model._noise is None or model._noise.protocol[-1] not in supported_noise_models:
            raise ValueError(
                "A NoiseProtocol.DEPOLARIZING or NoiseProtocol.DEPHASING model must be provided"
                " either to .mitigate() or through the <class QuantumModel>."
            )
        noise = model._noise

    mitigation_zne = analog_zne(
        model=model, options=options, noise=noise, param_values=param_values
    )
    return mitigation_zne
