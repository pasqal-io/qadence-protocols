from __future__ import annotations

from qadence import QuantumModel
from qadence.blocks.abstract import AbstractBlock
from torch import Tensor

from qadence_protocols.measurements.utils_shadow import estimations


def process_shadow_options(options: dict) -> tuple:
    """Extract shadow_size, accuracy and confidence from options."""

    shadow_size = options.get("shadow_size", None)
    if shadow_size is None:
        raise KeyError("Robust Shadow protocol requires an option 'shadow_size' of type 'int'.")
    shadow_groups = options.get("shadow_groups", None)
    if shadow_groups is None:
        raise KeyError("Shadow protocol requires either an option" "'shadow_groups' of type 'int'.")

    robust_shadow_correlations = options.get("robust_correlations", None)

    return shadow_size, shadow_groups, robust_shadow_correlations


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
    (
        shadow_size,
        shadow_groups,
        robust_shadow_correlations,
    ) = process_shadow_options(options=options)

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
        robust_correlations=robust_shadow_correlations,
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
    (
        shadow_size,
        shadow_groups,
        robust_shadow_correlations,
    ) = process_shadow_options(options=options)

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
        robust_correlations=robust_shadow_correlations,
    )
