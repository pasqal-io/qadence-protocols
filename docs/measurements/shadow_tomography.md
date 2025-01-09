# Robust shadow tomography

In this tutorial, we will estimate a physical property out of a quantum system, namely the purity of the partial traces, in the presence of measurement noise. To do so, we will use the formalism of classical shadows[^1], and especially their robust version[^2]. This tutorial is inspired from a notebook example of robust shadow tomography from the randomized measurements toolbox[^5] in Julia[^3].


## Setting the model

First, we will set the noise model and a circuit from which we will estimate the purity.

### Noise model

We will use a depolarizing noise model with a different error probability per qubit.

```python exec="on" source="material-block" session="shadow_tomo" result="json"
import torch
from qadence import NoiseHandler, NoiseProtocol

torch.manual_seed(0)
n_qubits = 2
error_probs = torch.clamp(0.1 + 0.02 * torch.randn(n_qubits), min=0, max=1)
print(f"Error probabilities = {error_probs}") # markdown-exec: hide

noise = NoiseHandler(protocol=NoiseProtocol.DIGITAL.DEPOLARIZING, options={"error_probability": error_probs[0], "target": 0})

for i, proba in enumerate(error_probs[1:]):
    noise.digital_depolarizing(options={"error_probability": proba, "target": i+1})
```

### Noiseless circuit and model

Let us set the circuit without noise and calculating the expected purities:

```python exec="on" source="material-block" session="shadow_tomo" result="json"
from qadence import *

theta1 = Parameter("theta1", trainable=False)
theta2 = Parameter("theta2", trainable=False)
theta3 = Parameter("theta3", trainable=False)
theta4 = Parameter("theta4", trainable=False)

blocks = chain(
    kron(RX(0, theta1), RY(1, theta2)),
    kron(RX(0, theta3), RY(1, theta4,),),
)

circuit = QuantumCircuit(2, blocks)

values = {
    "theta1": torch.tensor([0.5]),
    "theta2": torch.tensor([1.5]),
    "theta3": torch.tensor([2.0]),
    "theta4": torch.tensor([2.5]),
}

model = QuantumModel(
    circuit=circuit,
    observable=[], # no observable needed here
)
```

For calculating purities, we can use the utility functions `partial_trace` and `purity`:

```python exec="on" source="material-block" session="shadow_tomo" result="json"
from qadence_protocols.utils_trace import partial_trace, purity

def partial_purities(density_mat):
    purities = []
    for i in range(n_qubits):
        partial_trace_i = partial_trace(density_mat, [i]).squeeze()
        purities.append(purity(partial_trace_i))

    return torch.tensor(purities)

expected_purities = partial_purities(model.run(values))
print(f"Expected purities = {expected_purities}") # markdown-exec: hide
```

### Add noise to circuit

The circuit is defined as follows where we set the previous noise model in the last operations as measurement noise.

```python exec="on" source="material-block" session="shadow_tomo" result="json"
noisy_blocks = chain(
    kron(RX(0, theta1), RY(1, theta2)),
    kron(RX(0, theta3, NoiseHandler(protocol=NoiseProtocol.DIGITAL.DEPOLARIZING, options={"error_probability": error_probs[0], "target": 0})),
        RY(1, theta4, NoiseHandler(protocol=NoiseProtocol.DIGITAL.DEPOLARIZING, options={"error_probability": error_probs[1], "target": 1})),
        ),
)

noisy_circuit = QuantumCircuit(2, noisy_blocks)
noisy_model = QuantumModel(
    circuit=noisy_circuit,
    observable=[], # no observable needed here
)
```

## Shadow estimations

### Vanilla classical shadows

We will first run vanilla shadows to reconstruct the density matrix representation of the circuit, from which we can estimate the purities.


```python exec="on" source="material-block" session="shadow_tomo" result="json"
from qadence_protocols import Measurements, MeasurementProtocol

shadow_options = {"shadow_size": 10200, "shadow_medians": 6, "n_shots":1000}
shadow_measurements = Measurements(protocol=MeasurementProtocol.SHADOW, options=shadow_options)
shadow_measurements.measure(noisy_model, param_values=values)
vanilla_purities = partial_purities(shadow_measurements.reconstruct_state())

print(f"Purities with classical shadows = {vanilla_purities}") # markdown-exec: hide
```

As we can see, the estimated purities diverge from the expected ones due to the presence of noise. Next, we will use robust shadows to mitigate the noise effect.

### Robust shadows

We now use an efficient calibration method based on the experimental demonstration of classical shadows[^4]. A first set of measurements are used to determine calibration coefficients. The latter are used within robust shadows to mitigate measurement errors.
Indeed, we witness below the estimated purities being closer to the analytical ones.

```python exec="on" source="material-block" session="shadow_tomo" result="json"
from qadence_protocols.measurements.calibration import zero_state_calibration

calibration = zero_state_calibration(n_unitaries=2000, n_qubits=circuit.n_qubits, n_shots=10000, noise=noise)
robust_options = {"shadow_size": 10200, "shadow_medians": 6, "n_shots":1000, "calibration": calibration}
robust_shadow_measurements = Measurements(protocol=MeasurementProtocol.ROBUST_SHADOW, options=robust_options)
robust_shadow_measurements.measure(noisy_model, param_values=values)
robust_purities = partial_purities(robust_shadow_measurements.reconstruct_state())

print(f"Expected purities = {expected_purities}") # markdown-exec: hide
print(f"Purities with robust shadows = {robust_purities}") # markdown-exec: hide
print(f"Purities with classical shadows = {vanilla_purities}") # markdown-exec: hide
```


[^1]: [Hsin-Yuan Huang, Richard Kueng and John Preskill, Predicting Many Properties of a Quantum System from Very Few Measurements (2020)](https://arxiv.org/abs/2002.08953)

[^2]: [Senrui Chen, Wenjun Yu, Pei Zeng, and Steven T. Flammia, Robust Shadow Estimation (2021)](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.030348)

[^3]: [RandomMeas.jl tutorial robust shadow tomography](https://github.com/bvermersch/RandomMeas.jl/blob/main/examples/RobustShadowTomography.ipynb)

[^4]: [Vittorio Vitale, Aniket Rath, Petar Jurcevic, Andreas Elben, Cyril Branciard, and Benoît Vermersch, Robust Estimation of the Quantum Fisher Information on a Quantum Processor (2024)](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.030338)

[^5]: [Andreas Elben, Steven T. Flammia, Hsin-Yuan Huang, Richard Kueng, John Preskill, Benoît Vermersch, and Peter Zoller, The randomized measurement toolbox (2022)](https://www.nature.com/articles/s42254-022-00535-2)
