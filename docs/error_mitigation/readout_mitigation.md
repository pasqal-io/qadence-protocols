## Readout error mitigation

Readout errors are introduced during measurements in the computation basis via probabilistic bitflip operators characterized by the readout matrix (also known as confusion matrix) defined over the system of qubits of dimension $2^n\times2^n$. The complete implementation of the mitigation technique involves using the characterized readout matrix for the system of qubits $(T)$ and classically applying an inversion  $(T^{−1})$ to the measured probability distributions. However there are several limitations of this approach:

- The complete implementation requires $2^n$ characterization experiments (probability measurements), which is not scalable.
- Classical overhead from full matrix inversion for large system of qubits is expensive
- The matrix $T$ may become singular for large $n$, preventing direct inversion.
- The inverse $T^{−1}$ might not be a stochastic matrix, meaning that it can produce negative corrected probabilities.
- The correction is not rigorously justified, so we cannot be sure that we are only removing SPAM errors and not otherwise corrupting an estimated probability distribution.

Qadence relies on the assumption of _uncorrelated_ readout errors, this gives us:

$$
T=T_1\otimes T_2\otimes \dots \otimes T_n
$$

for which the inversion is straightforward:

$$
T^{-1}=T_1^{-1}\otimes T_2^{-1}\otimes \dots \otimes T_n^{-1}
$$



```python exec="on" source="material-block" session="mitigation" result="json"
from qadence import QuantumModel, QuantumCircuit, hamiltonian_factory, kron, H, Z, I
from qadence import NoiseProtocol, NoiseHandler


# Simple circuit and observable construction.
block = kron(H(0), I(1))
circuit = QuantumCircuit(2, block)
n_shots = 10000

# Construct a quantum model and noise
model = QuantumModel(circuit=circuit)
error_probability = 0.2
noise = NoiseHandler(protocol=NoiseProtocol.READOUT.INDEPENDENT,options={"error_probability": error_probability})

noiseless_samples = model.sample(n_shots=n_shots)
noisy_samples = model.sample(noise=noise, n_shots=n_shots)

print(f"noiseless samples: {noiseless_samples}") # markdown-exec: hide
print(f"noisy samples: {noisy_samples}") # markdown-exec: hide
```

Note that the noisy states have samples with the second qubit flipped. In the below protocols, we describe ways to reconstruct the noiseless distribution (untargeted mitigation). Besides this one might just be interrested in mitigating the expectation value (targeted mitigation).

!!! warning "Mitigation noise"
    Note it is necessary to pass the `noise` parameter for the `mitigation` function because the mitigation process sees `QuantumModel` as a black box.

### Constrained optimization

However, even for a reduced $n$ the forth limitation holds. This can be avoided by reformulating into a minimization problem[^1]:

$$
\lVert Tp_{\textrm{corr}}-p_{\textrm{raw}}\rVert_{2}^{2}
$$

subjected to physicality constraints $0 \leq p_{corr}(x) \leq 1$ and $\lVert p_{corr} \rVert = 1$. At this point, two methods are implemented to solve this problem. The method involves solving a constrained optimization problem and can be computationally expensive.

```python exec="on" source="material-block" session="mitigation" result="json"

from qadence_protocols.mitigations.protocols import Mitigations
from qadence_protocols.types import ReadOutOptimization


# Define the mitigation method solving the minimization problem:
options={"optimization_type": ReadOutOptimization.CONSTRAINED, "n_shots": n_shots}
mitigation = Mitigations(protocol=Mitigations.READOUT, options=options)

# Run noiseless, noisy and mitigated simulations.
mitigated_samples_opt = mitigation(model=model, noise=noise)

print(f"Optimization based mitigation: {mitigated_samples_opt}") # markdown-exec: hide
```



### Maximum Likelihood estimation (MLE)
This method replaces the constraints with additional post processing for correcting probability distributions with negative entries. The runtime of the method is linear in the size of the distribution and thus is very efficient. The optimality of the solution is however not always guaranteed. The method redistributes any negative probabilities on using the inverse operation equally and can be shown to maximize the likelihood with minimal effort[^2].


```python exec="on" source="material-block" session="mitigation" result="json"

# Define the mitigation method solving the minimization problem:
options={"optimization_type": ReadOutOptimization.MLE, "n_shots": n_shots}
mitigation = Mitigations(protocol=Mitigations.READOUT, options=options)
mitigated_samples_mle = mitigation(model=model, noise=noise)
print(f"MLE based mitigation {mitigated_samples_mle}") # markdown-exec: hide
```

### Matrix free measurement mitigation (MTHREE)

This method relies on inverting the probability distribution within a restricted subspace of measured bitstrings[^3]. The method is better suited for computations that exceed 20 qubits where the corrected probability distribution would require a state in a unreasonably high dimensional Hilbert space. Thus, the idea here is to stick to the basis states that show up in the measurement alone. Additionally, one might want to include states that are $k$ hamming distance away from it.

```python exec="on" source="material-block" session="m3" result="json"


import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import gmres

n_qubits = 10

# Prepare a probability distribution with sparsity
exact_prob = np.random.rand(2 ** n_qubits)
exact_prob[2 ** (n_qubits // 2):] = 0
exact_prob = 0.90 * exact_prob + 0.1 * np.ones(2 ** n_qubits) / 2 ** n_qubits
exact_prob /= sum(exact_prob)
np.random.shuffle(exact_prob)

# Create an observed probability distribution with thresholded values
observed_prob = np.array(exact_prob, copy=True)
observed_prob[exact_prob < 1 / 2 ** n_qubits] = 0
observed_prob /= sum(observed_prob)

# Convert the observed probability distribution into a sparse matrix
input_csr_matrix = csr_matrix(observed_prob, shape=(1, 2**n_qubits)).T

# Print the binary representation of states with nonzero probabilities
print({
    bin(x)[2:].zfill(n_qubits): np.round(input_csr_matrix[x, 0], 3)
    for x in input_csr_matrix.nonzero()[0]
}) # markdown-exec: hide

# Compute and display the percentage of nonzero entries
filling_percentage = len(input_csr_matrix.nonzero()[0]) / 2**n_qubits
print(f"Filling percentage: {filling_percentage:.6f} %") # markdown-exec: hide
```

We have constructed a probability distribution over a small subspace of bitstrings, leveraging a `csr_matrix` for efficient storage and computation. Now, we apply `MTHREE` to mitigate errors in the probability distribution. Within `MTHREE`, sparsity can be further improved by incorporating the Hamming distance approach. This method considers only the noise matrix elements corresponding to quantum states within a specified Hamming distance of the correct state. This feature can be used by setting the `hamming_dist` option.

```python exec="on" source="material-block" session="m3" result="json"
from scipy.stats import wasserstein_distance
from qadence_protocols.mitigations.readout import (
    normalized_subspace_kron,
    mle_solve,
    matrix_inv,
    tensor_rank_mult
)

# Generate noise transition matrices for each qubit
noise_matrices = []
for qubit_idx in range(n_qubits):
    transition_prob_a, transition_prob_b = np.random.rand(2) / 8
    transition_matrix = np.array([[1 - transition_prob_a, transition_prob_a],
                                  [transition_prob_b, 1 - transition_prob_b]]).T  # Ensure column sums to 1
    noise_matrices.append(transition_matrix)



# Compute the subspace confusion matrix using noise transition matrices. We set the hamming distance for filtering out noisy states that are far from the correct state
subspace_confusion_matrix = normalized_subspace_kron(noise_matrices, observed_prob.nonzero()[0], hamming_dist=1)

# Apply GMRES (Generalized Minimal Residual Method) to correct the probability distribution using MTHREE. Then we apply Maximum Likelihood Estimation (MLE) to ensure the validity of the probability distribution.
corrected_prob_mthree_mle = mle_solve(gmres(subspace_confusion_matrix, input_csr_matrix.toarray())[0])

# Next, we use tensor rank multiplication to apply the inverse noise matrices to the observed probability distribution, followed by the same MLE correction
inverse_noise_matrices = list(map(matrix_inv, noise_matrices))
corrected_prob_inverse_mle = mle_solve(tensor_rank_mult(inverse_noise_matrices, observed_prob))

# Finally, we compute the Wasserstein distance between the two corrected probability distributions
wasserstein_dist = wasserstein_distance(corrected_prob_mthree_mle, corrected_prob_inverse_mle)
print(f"Wasserstein distance between the two distributions: {wasserstein_dist}")  # markdown-exec: hide
```

In `MTHREE`, we assume quantum circuits that exceed 20 qubits, which results in a high sparsity in the probability distribution of the output bitstrings, leading to many 0 probability bitstrings. Therefore, we use `Wasserstein Distance` instead of `KL divergence` and its derivative, `JS divergence`, as they put true values (which is 0 here) in the denominator and may diverge in such cases, whereas `Wasserstein Distance` remains stable for comparisons.


### Majority Voting

Mitigation protocol to be used only when the circuit output has a single expected bitstring as the solution [^4]. The method votes on the likeliness of each qubit to be a 0 or 1 assuming a tensor product structure for the output. The method is valid only when the readout errors are not correlated.

```python exec="on" source="material-block" session="mv" result="json"
from qadence import QuantumModel, QuantumCircuit,kron, H, Z, I
from qadence import NoiseHandler, NoiseProtocol
from qadence_protocols.mitigations.readout import majority_vote
import numpy as np

# Simple circuit and observable construction.
n_qubits = 4
block = kron(*[I(i) for i in range(n_qubits)])
circuit = QuantumCircuit(n_qubits, block)
n_shots = 1000

# Construct a quantum model.
model = QuantumModel(circuit=circuit)

# Sampling the noisy solution
error_p = 0.2
noise = NoiseHandler(protocol=NoiseProtocol.READOUT.INDEPENDENT,options={"error_probability": error_p})
noisy_samples = model.sample(noise=noise, n_shots=n_shots)[0]

# Removing samples that correspond to actual solution
noisy_samples['0000'] = 0

# Constructing the probability vector
ordered_bitstrings = [bin(k)[2:].zfill(n_qubits) for k in range(2**n_qubits)]
observed_prob = np.array([noisy_samples[bs] for bs in ordered_bitstrings]) / n_shots


print(f"noisy samples: {noisy_samples}") # markdown-exec: hide
print(f"observed probability: {np.around(observed_prob,3)}") # markdown-exec: hide

```

We have removed the actual solution from the observed distribution and will use this as the observed probability.

```python exec="on" source="material-block" session="mv" result="json"

noise_matrices = [np.array([[1 - error_p, error_p], [error_p, 1 - error_p]])]*n_qubits
result_index = majority_vote(noise_matrices, observed_prob).argmax()
print(f"mitigated solution index: {result_index}" ) # markdown-exec: hide
```

### Model free mitigation

You can perform the mitigation without a `quantum model` if you have sampled results from previous executions. This eliminates the need to reinitialize the circuit and sample again. Instead, you can directly apply the mitigation method to the existing data. To do this, you need to insert the `samples` at your `options` when initializing the `Mitigations` class.

```python exec="on" source="material-block" session="mitigation" result="json"
from qadence import QuantumModel, QuantumCircuit, hamiltonian_factory, kron, H, Z, I
from qadence import NoiseProtocol, NoiseHandler
from qadence_protocols.mitigations.protocols import Mitigations
from qadence_protocols.types import ReadOutOptimization

# Simple circuit and observable construction.
block = kron(H(0), I(1))
circuit = QuantumCircuit(2, block)
n_shots = 10000

# Construct a quantum model and noise
model = QuantumModel(circuit=circuit)
error_probability = 0.2
noise = NoiseHandler(protocol=NoiseProtocol.READOUT.INDEPENDENT,options={"error_probability": error_probability})

noiseless_samples = model.sample(n_shots=n_shots)
noisy_samples = model.sample(noise=noise, n_shots=n_shots)

# Define the mitigation method with the sample results
options={"optimization_type": ReadOutOptimization.MLE, "n_shots": n_shots, "samples": noisy_samples}
mitigation = Mitigations(protocol=Mitigations.READOUT, options=options)

# Run noiseless, noisy and mitigated simulations.
mitigated_samples_opt = mitigation(noise=noise)

print(f"Noisy samples: {noisy_samples}") # markdown-exec: hide
print(f"Mitigates samples: {mitigated_samples_opt}") # markdown-exec: hide

```

### Twirl mitigation

This protocol makes use of all possible so-called twirl operations to average out the effect of readout errors into an effective scaling. The twirl operation consists of using bit flip operators before and after the measurement [^5]. The number of twirl operations can be reduced through random sampling with the `twirl_samples` option. The method is exact in that it requires no calibration which might be prone to modelling errors.

```python exec="on" source="material-block" session="mfm" result="json"
from qadence import NoiseHandler, NoiseProtocol
from qadence.measurements import Measurements
from qadence.operations import CNOT, RX, Z
from qadence_protocols import Mitigations

import torch
from qadence import (
    QuantumCircuit,
    QuantumModel,
    chain,
    kron,
)

error_probability=0.15
n_shots=10000
block= chain(kron(RX(0, torch.pi / 3), RX(1, torch.pi / 6)), CNOT(0, 1))
observable=[3 * kron(Z(0), Z(1)) + 2 * Z(0)]

circuit = QuantumCircuit(block.n_qubits, block)
noise = NoiseHandler(protocol=NoiseProtocol.READOUT.INDEPENDENT, options={"error_probability": error_probability})
tomo_measurement = Measurements(
    protocol=Measurements.TOMOGRAPHY,
    options={"n_shots": n_shots},
)

model = QuantumModel(
    circuit=circuit, observable=observable,
)

noisy_model = QuantumModel(
    circuit=circuit,
    observable=observable,
    measurement=tomo_measurement,
    noise=noise,
)
print(f"noiseless expectation value {model.expectation(measurement=tomo_measurement,)}") # markdown-exec: hide
print(f"noisy expectation value {noisy_model.expectation(measurement=tomo_measurement,)}") # markdown-exec: hide

mitigate = Mitigations(protocol=Mitigations.TWIRL)
expectation_mitigated = mitigate(noise=noise, model=noisy_model)


# We set a number of qubits as the sample count. The number of twirl_samples can range from 1 to the maximum number of qubit index combinations. For example, using a higher number of samples can improve accuracy.
options={"twirl_samples": block.n_qubits}
mitigate_sample = Mitigations(protocol=Mitigations.TWIRL, options=options)
expectation_mitigated_sample = mitigate_sample(noise=noise, model=noisy_model)

print(f"expected mitigation value {expectation_mitigated}") # markdown-exec: hide
print(f"expected sampled mitigation value {expectation_mitigated_sample}") # markdown-exec: hide

```

## References

[^1]: [Michael R. Geller and Mingyu Sun, Efficient correction of multiqubit measurement errors, (2020)](https://arxiv.org/abs/2001.09980)

[^2]: [Smolin _et al._, Maximum Likelihood, Minimum Effort, (2011)](https://arxiv.org/abs/1106.5458)

[^3]: [Gambetta _et al._: Scalable mitigation of measurement errors on quantum computers](https://arxiv.org/pdf/2108.12518)

[^4]: [Dror Baron _et al._: Maximum Likelihood Quantum Error Mitigation for Algorithms with a Single Correct Output](https://arxiv.org/pdf/2402.11830)

[^5]: [Kristan Temme _et al._ : Model-free readout-error mitigation for quantum expectation values ](https://arxiv.org/pdf/2012.09738)
