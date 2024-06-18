## Zero-noise extrapolation for analog blocks

Zero-noise extrapolation (ZNE) is an error mitigation technique in which the expectation value computed at different noise levels is extrapolated to the zero noise limit (ideal expectation) using a class of functions. In digital computing, this is typically implemented by "folding" the circuit at a local (involves inverting gates locally) or global level (involves inverting blocks of gates). This allows to artificially increase the noise levels by integer folds[^1]. In the analog ZNE variation, analog blocks are time stretched to again artificially increase in noise[^1]. Using ZNE on neutral atoms would require stretching the register to scale the interaction hamiltonian appropriately.

```python exec="on" source="material-block" session="zne" result="json"
from qadence import QuantumModel, QuantumCircuit,kron, chain, AnalogRX, AnalogRZ, PI, BackendName, DiffMode,Z
import numpy as np


analog_block = chain(AnalogRX(PI / 2.0), AnalogRZ(PI))
observable = [Z(0) + Z(1)]
circuit = QuantumCircuit(2, analog_block)
model_noiseless = QuantumModel(
    circuit=circuit, observable=observable, backend=BackendName.PULSER, diff_mode=DiffMode.GPSR
)

print("noiseless_expectation", model_noiseless.expectation()) # markdown-exec: hide

```

```python exec="on" source="material-block" session="zne" result="json"

from qadence.noise import Noise
from qadence_protocols.mitigations.protocols import Mitigations
import torch

noise = Noise(protocol=Noise.DEPOLARIZING, options={"noise_probs": [0.2]})
model = QuantumModel(
    circuit=circuit, observable=observable, backend=BackendName.PULSER, diff_mode=DiffMode.GPSR
)

for data_points in [2,5]:
    options = {"stretches": torch.linspace(1, 3, data_points)}
    mitigate = Mitigations(protocol=Mitigations.ANALOG_ZNE, options=options).mitigation()

    mitigated_expectation = mitigate(model=model, noise=noise)

    print(f"noiseless_expectation with {data_points} data points", mitigated_expectation) # markdown-exec: hide

```

## References

[^1]: [Mitiq: What's the theory behind ZNE?](https://mitiq.readthedocs.io/en/stable/guide/zne-5-theory.html)
