## Readout error mitigation

Readout errors are introduced during measurements in the computation basis via probabilistic bitflips operators characterized by the readout matrix (also known as confusion matrix) defined over the system of qubits of dimension $2^n\times2^n$. The complete implementation of the mitigation technique involves using the characterized readout matrix for the system of qubits $(T)$ and classically applying an inversion  $(T^{−1})$ to the measured probability distributions. However there are several limitations of this approach:

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
from qadence import QuantumModel, QuantumCircuit,hamiltonian_factory,kron, H, Z, I
from qadence.noise import Noise


# Simple circuit and observable construction.
block = kron(H(0), I(1))
circuit = QuantumCircuit(2, block)
n_shots = 10000

# Construct a quantum model and noise
model = QuantumModel(circuit=circuit)
error_probability = 0.2
noise = Noise(protocol=Noise.READOUT,options={"error_probability": error_probability})

noiseless_samples = model.sample(n_shots=n_shots)
noisy_samples = model.sample(noise=noise, n_shots=n_shots)

print(f"noiseless samples: {noiseless_samples}") # markdown-exec: hide
print(f"noisy samples: {noisy_samples}") # markdown-exec: hide
```

Note that the noisy states have samples with the second qubit flipped. In the below protocols, we describe ways to reconstruct the noiseless distribution (untargeted mitigation). Besides this one might just be interrested in mitigating the expectation value (targeted mitigation).

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
mitigation = Mitigations(protocol=Mitigations.READOUT, options=options).mitigation()

# Run noiseless, noisy and mitigated simulations.
mitigated_samples_opt = mitigation(model=model, noise=noise)

print(f"Optimization based mitigation: {mitigated_samples_opt}") # markdown-exec: hide
```



### Maximum Likelihood estimation (MLE)
This method replaces the constraints with additional post processing for correcting probability distributions with negative entries. The runtime of the method is linear in the size of the distribution and thus is very efficient. The optimality of the solution is however not always guaranteed. The method redistributes any negative probabilities on using the inverse operation equally and can be shown to maximize the likelihood with minimal effort[^2].


```python exec="on" source="material-block" session="mitigation" result="json"

# Define the mitigation method solving the minimization problem:
options={"optimization_type": ReadOutOptimization.MLE, "n_shots": n_shots}
mitigation = Mitigations(protocol=Mitigations.READOUT, options=options).mitigation()
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
exact_prob = np.random.rand(2 ** (n_qubits))
exact_prob[2 ** (n_qubits//2):]=0
exact_prob = 0.90 * exact_prob + 0.1 * np.ones(2**n_qubits) / 2**n_qubits
exact_prob = exact_prob / sum(exact_prob)
np.random.shuffle(exact_prob)

observed_prob = np.array(exact_prob, copy=True)
observed_prob[exact_prob < 1 / 2 ** (n_qubits)] = 0

observed_prob = observed_prob / sum(observed_prob)

input_csr = csr_matrix(observed_prob, shape=(1, 2**n_qubits)).T
print({bin(x)[2:].zfill(n_qubits):np.round(input_csr[x,0],3) for x in input_csr.nonzero()[0]}) # markdown-exec: hide
print(f"Filling percentage {len(input_csr.nonzero()[0])/2**n_qubits} %") # markdown-exec: hide
```

We have generated a probability distribution within a small subspace of bitstrings being filled. We use `csr_matrix` for efficient representation and computation. Now we use MTHREE to do mitigation on the probability distribution


```python exec="on" source="material-block" session="m3" result="json"
from scipy.stats import wasserstein_distance
from qadence_protocols.mitigations.readout import (
    normalized_subspace_kron,
    mle_solve,
    matrix_inv,
    tensor_rank_mult
    )

noise_matrices = []
for t in range(n_qubits):
    t_a, t_b = np.random.rand(2) / 8
    K = np.array([[1 - t_a, t_a], [t_b, 1 - t_b]]).transpose()  # column sum be 1
    noise_matrices.append(K)

confusion_matrix_subspace = normalized_subspace_kron(noise_matrices, observed_prob.nonzero()[0])

p_corr_mthree_gmres = gmres(confusion_matrix_subspace, input_csr.toarray())[0]
p_corr_mthree_gmres_mle = mle_solve(p_corr_mthree_gmres)

noise_matrices_inv = list(map(matrix_inv, noise_matrices))
p_corr_inv_mle = mle_solve(tensor_rank_mult(noise_matrices_inv, observed_prob))

distance = wasserstein_distance(p_corr_mthree_gmres_mle, p_corr_inv_mle)
print("Wassertein distance between the 2 distributions",distance)  # markdown-exec: hide


```

We have used `wassestein_distance` instead of `kl_divergence` as many of the bistrings have 0 probabilites. If the expected solution lies outside the space of observed bitstrings, `MTHREE` will fail. We next look at majority voting to circument this problem when the expected output is a single bitstring.

### Majority Voting

Mitigation protocol to be used only when the circuit output has a single expected bitstring as the solution [^4]. The method votes on the likeliness of each qubit to be a 0 or 1 assuming a tensor product structure for the output. The method is valid only when the readout errors are not correlated.

```python exec="on" source="material-block" session="mv" result="json"
from qadence import QuantumModel, QuantumCircuit,kron, H, Z, I
from qadence.noise import Noise
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
noise = Noise(protocol=Noise.READOUT,options={"error_probability": error_p})
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
print("mitigated solution index:", result_index ) # markdown-exec: hide
```

### Model free mitigation

This protocol makes use of all possible twirl operations to average out the effect of readout errors into an effective scaling. The twirl operation consists of using bit flip operators before the measurement and after the measurement is obtained[^5]. The number of twirl operations can be reduced through random sampling. The method is exact in that it requires no calibration which might be prone to errors of modelling.

```python exec="on" source="material-block" session="mfm" result="json"
from qadence.measurements import Measurements
from qadence.noise.protocols import Noise
from qadence.operations import CNOT, RX, Z
from qadence_protocols.mitigations.protocols import Mitigations

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
noise = Noise(protocol=Noise.READOUT, options={"error_probability": error_probability})
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
print("noiseless expectation value ", model.expectation(measurement=tomo_measurement,)) # markdown-exec: hide
print("noisy expectation value ", noisy_model.expectation(measurement=tomo_measurement,)) # markdown-exec: hide

mitigate = Mitigations(protocol=Mitigations.TWIRL).mitigation()
expectation_mitigated = mitigate(noisy_model)

print("expected mitigation value",expectation_mitigated) # markdown-exec: hide

```

## References

[^1]: [Michael R. Geller and Mingyu Sun, Efficient correction of multiqubit measurement errors, (2020)](https://arxiv.org/abs/2001.09980)

[^2]: [Smolin _et al._, Maximum Likelihood, Minimum Effort, (2011)](https://arxiv.org/abs/1106.5458)

[^3]: [Gambetta _et al._: Scalable mitigation of measurement errors on quantum computers](https://arxiv.org/pdf/2108.12518)

[^4]: [Dror Baron _et al._: Maximum Likelihood Quantum Error Mitigation for Algorithms with a Single Correct Output](https://arxiv.org/pdf/2402.11830)

[^5]: [Kristan Temme _et al._ : Model-free readout-error mitigation for quantum expectation values ](https://arxiv.org/pdf/2012.09738)
