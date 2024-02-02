from __future__ import annotations

import numpy as np
import torch
from qadence import BackendName, QuantumModel
from qadence.backends.api import backend_factory
from qadence.backends.pulser.backend import Backend
from qadence.blocks import block_to_tensor
from qadence.blocks.abstract import AbstractBlock
from qadence.blocks.analog import ConstantAnalogRotation, InteractionBlock
from qadence.circuit import QuantumCircuit
from qadence.noise import Noise
from qadence.operations import AnalogRot
from qadence.transpile import apply_fn_to_blocks
from qadence.utils import Endianness
from torch import Tensor

supported_noise_models = [Noise.DEPOLARIZING, Noise.DEPHASING]


def zne(noise_levels: Tensor, zne_datasets: list[list]) -> Tensor:
    poly_fits = []
    for dataset in zne_datasets:  # Looping over batched observables.
        batched_observable: list = []
        n_params = len(dataset[0])
        for p in range(n_params):  # Looping over the batched parameters.
            rearranged_dataset = [s[p] for s in dataset]
            # Polynomial fit function.
            poly_fit = np.poly1d(
                np.polyfit(noise_levels, rearranged_dataset, len(noise_levels) - 1)
            )
            # Return the zero-noise extrapolated value.
            batched_observable.append(poly_fit(0.0))
        poly_fits.append(batched_observable)

    return torch.tensor(poly_fits)


def pulse_experiment(
    backend: Backend,
    circuit: QuantumCircuit,
    observable: list[AbstractBlock],
    param_values: dict[str, Tensor],
    noise: Noise,
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

    zne_datasets = []
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
        conv_circuit = backend.circuit(stretched_circuit)
        noisy_density_matrices.append(
            # Contain a single experiment result for the stretch.
            backend.run_dm(
                conv_circuit,
                param_values=param_values,
                state=state,
                noise=noise,
                endianness=endianness,
            )[0]
        )
    # Convert observable to Numpy types compatible with QuTip simulations.
    # Matrices are flipped to match QuTip conventions.
    converted_observable = [np.flip(block_to_tensor(obs).numpy()) for obs in observable]
    # Create ZNE datasets by looping over batches.
    for observable in converted_observable:
        # Get expectation values at the end of the time serie [0,t]
        # at intervals of the sampling rate.
        zne_datasets.append(
            [
                [dm.expect(observable)[0][-1] for dm in density_matrices]
                for density_matrices in noisy_density_matrices
            ]
        )
    # Zero-noise extrapolate.
    extrapolated_exp_values = zne(
        noise_levels=stretches,
        zne_datasets=zne_datasets,
    )
    return extrapolated_exp_values


def noise_level_experiment(
    backend: Backend,
    circuit: QuantumCircuit,
    observable: list[AbstractBlock],
    param_values: dict[str, Tensor],
    noise: Noise,
    endianness: Endianness,
    state: Tensor | None = None,
) -> Tensor:
    noise_probs = noise.options.get("noise_probs")
    zne_datasets: list = []
    # Get noisy density matrices.
    conv_circuit = backend.circuit(circuit)
    noisy_density_matrices = backend.run_dm(
        conv_circuit, param_values=param_values, state=state, noise=noise, endianness=endianness
    )
    # Convert observable to Numpy types compatible with QuTip simulations.
    # Matrices are flipped to match QuTip conventions.
    converted_observable = [np.flip(block_to_tensor(obs).numpy()) for obs in observable]
    # Create ZNE datasets by looping over batches.
    for observable in converted_observable:
        # Get expectation values at the end of the time serie [0,t]
        # at intervals of the sampling rate.
        zne_datasets.append(
            [
                [dm.expect(observable)[0][-1] for dm in density_matrices]
                for density_matrices in noisy_density_matrices
            ]
        )
    # Zero-noise extrapolate.
    extrapolated_exp_values = zne(noise_levels=noise_probs, zne_datasets=zne_datasets)
    return extrapolated_exp_values


def analog_zne(
    model: QuantumModel,
    options: dict,
    noise: Noise,
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
    noise: Noise | None = None,
    param_values: dict[str, Tensor] = dict(),
) -> Tensor:
    if noise is None or noise.protocol not in supported_noise_models:
        if model._noise is None or model._noise.protocol not in supported_noise_models:
            raise ValueError(
                "A Noise.DEPOLARIZING or Noise.DEPHASING model must be provided"
                " either to .mitigate() or through the <class QuantumModel>."
            )
        noise = model._noise

    mitigation_zne = analog_zne(
        model=model, options=options, noise=noise, param_values=param_values
    )
    return mitigation_zne
