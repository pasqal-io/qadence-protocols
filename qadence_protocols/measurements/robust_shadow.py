from __future__ import annotations

from qadence import QuantumModel
from qadence.blocks.abstract import AbstractBlock
from torch import Tensor

from qadence_protocols.measurements.utils_shadow import estimations


def compute_measurements(
    model: QuantumModel,
    observables: list[AbstractBlock],
    options: dict,
    param_values: dict[str, Tensor] = dict(),
    state: Tensor | None = None,
) -> Tensor:
    """
    Construct a classical shadow of a state to estimate observable expectation values.

    Args:
        model (QuantumModel): Model to evaluate.
        observables (list[AbstractBlock]): a list of observables
            to estimate the expectation values from.
        param_values (dict): a dict of values to substitute the
            symbolic parameters for.
        options (dict): a dict of options for the measurement protocol.
            Here, shadow_size (int), accuracy (float) and confidence (float) are supported.
        state (Tensor | None): an initial input state.

    Returns:
        expectations (Tensor): an estimation of the expectation values.
    """

    circuit = model._circuit.original
    shadow_size = options["shadow_size"]
    shadow_groups = options["shadow_groups"]
    calibration = options["calibration"]
    if calibration is None:
        calibration = [1.0 / 3.0] * circuit.n_qubits

    return estimations(
        circuit=circuit,
        observables=observables,
        param_values=model.embedding_fn(model._params, param_values),
        shadow_size=shadow_size,
        accuracy=0.0,
        confidence_or_groups=shadow_groups,
        state=state,
        backend=model.backend,
        noise=model._noise,
        return_shadows=True,
        robust_shadow=True,
        calibration=calibration,
    )


def compute_expectation(
    model: QuantumModel,
    observables: list[AbstractBlock],
    options: dict,
    param_values: dict[str, Tensor] = dict(),
    state: Tensor | None = None,
) -> Tensor:
    """
    Construct a classical shadow of a state to estimate observable expectation values.

    Args:
        model (QuantumModel): Model to evaluate.
        observables (list[AbstractBlock]): a list of observables
            to estimate the expectation values from.
        param_values (dict): a dict of values to substitute the
            symbolic parameters for.
        options (dict): a dict of options for the measurement protocol.
            Here, shadow_size (int), accuracy (float) and confidence (float) are supported.
        state (Tensor | None): an initial input state.

    Returns:
        expectations (Tensor): an estimation of the expectation values.
    """

    circuit = model._circuit.original
    shadow_size = options["shadow_size"]
    shadow_groups = options["shadow_groups"]
    calibration = options["calibration"]
    if calibration is None:
        calibration = [1.0 / 3.0] * circuit.n_qubits

    return estimations(
        circuit=circuit,
        observables=observables,
        param_values=model.embedding_fn(model._params, param_values),
        shadow_size=shadow_size,
        accuracy=0.0,
        confidence_or_groups=shadow_groups,
        state=state,
        backend=model.backend,
        noise=model._noise,
        return_shadows=False,
        robust_shadow=True,
        calibration=calibration,
    )
