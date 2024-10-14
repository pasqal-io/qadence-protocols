Sample-based measurement protocols are fundamental tools for the prediction and estimation of a quantum state as the result of NISQ programs executions. Their resource efficient implementation is a current and active research field. Currently, quantum state tomography is implemented in qadence-protocols.

## Quantum state tomography

The fundamental task of quantum state tomography is to learn an approximate classical description of an output quantum state described by a density matrix $\rho$, from repeated measurements of copies on a chosen basis. To do so, $\rho$ is expanded in a basis of observables (the tomography step) and for a given observable $\hat{\mathcal{O}}$, the expectation value is calculated with $\langle \hat{\mathcal{O}} \rangle=\textrm{Tr}(\hat{\mathcal{O}}\rho)$. A number of measurement repetitions in a suitable basis is then required to estimate $\langle \hat{\mathcal{O}} \rangle$.

The main drawback is the scaling in measurements for the retrieval of the classical expression for a $n$-qubit quantum state as $2^n \times 2^n$, together with a large amount of classical post-processing.

For an observable expressed as a Pauli string $\hat{\mathcal{P}}$, the expectation value for a state $|\psi \rangle$ can be derived as:

$$
\langle \hat{\mathcal{P}} \rangle=\langle \psi | \hat{\mathcal{P}} |\psi \rangle=\langle \psi | \hat{\mathcal{R}}^\dagger \hat{\mathcal{D}} \hat{\mathcal{R}} |\psi \rangle
$$

The operator $\hat{\mathcal{R}}$ diagonalizes $\hat{\mathcal{P}}$ and rotates the state into an eigenstate in the computational basis. Therefore, $\hat{\mathcal{R}}|\psi \rangle=\sum\limits_{z}a_z|z\rangle$ and the expectation value can finally be expressed as:


$$
\langle \hat{\mathcal{P}} \rangle=\sum_{z,z'}\langle z |\bar{a}_z\hat{\mathcal{D}}a_{z'}|z'\rangle = \sum_{z}|a_z|^2(-1)^{\phi_z(\hat{\mathcal{P}})}
$$


In Qadence, running a tomographical experiment is made simple by defining a `Measurements` object that captures all options for execution:

```python exec="on" source="material-block" session="measurements" result="json"
from torch import tensor
from qadence import hamiltonian_factory, BackendName, DiffMode, Noise
from qadence import chain, kron, X, Z, QuantumCircuit, QuantumModel
from qadence_protocols import Measurements

blocks = chain(
    kron(X(0), X(1)),
    kron(Z(0), Z(1)),
)

# Create a circuit and an observable.
circuit = QuantumCircuit(2, blocks)
observable = hamiltonian_factory(2, detuning=Z)

# Create a model.
model = QuantumModel(
    circuit=circuit,
    observable=observable,
    backend=BackendName.PYQTORCH,
    diff_mode=DiffMode.GPSR,
)

# Define a measurement protocol by passing the shot budget as an option.
tomo_options = {"n_shots": 100000}
tomo_measurement = Measurements(protocol=Measurements.TOMOGRAPHY, options=tomo_options)

# Get the exact expectation value.
exact_values = model.expectation()

# Run the tomography experiment.
estimated_values_tomo = tomo_measurement(model)

print(f"Exact expectation value = {exact_values}") # markdown-exec: hide
print(f"Estimated expectation value tomo = {estimated_values_tomo}") # markdown-exec: hide
```

## Classical shadows

A much less resource demanding protocol based on _classical shadows_ has been proposed[^1]. It combines ideas from shadow tomography[^2] and randomized measurement protocols [^3] capable of learning a classical shadow of an unknown quantum state $\rho$. It relies on deliberately discarding the full classical characterization of the quantum state, and instead focuses on accurately predicting a restricted set of properties that provide efficient resources for the study of the system.

A random measurement consists of applying random unitary rotations before a fixed measurement on each copy of a state. Appropriately averaging over these measurements produces an efficient estimator for the expectation value of an observable. This protocol therefore creates a robust classical representation of the quantum state or classical shadow. The captured measurement information is then reuseable for multiple purposes, _i.e._ any observable expected value and available for noise mitigation postprocessing.

A classical shadow is therefore an unbiased estimator of a quantum state $\rho$. Such an estimator is obtained with the following procedure[^1]: first, apply a random unitary gate $U$ to rotate the state: $\rho \rightarrow U \rho U^\dagger$ and then perform a basis measurement to obtain a $n$-bit measurement $|\hat{b}\rangle \in \{0, 1\}^n$. Both unitary gates $U$ and the measurement outcomes $|\hat{b}\rangle$ are stored on a classical computer for postprocessing v $U^\dagger |\hat{b}\rangle\langle \hat{b}|U$, a classical snapshot of the state $\rho$. The whole procedure can be seen as a quantum channel $\mathcal{M}$ that maps the initial unknown quantum state $\rho$ to the average result of the measurement protocol:

$$
\mathbb{E}[U^\dagger |\hat{b}\rangle\langle \hat{b}|U] = \mathcal{M}(\rho) \Rightarrow \rho = \mathbb{E}[\mathcal{M}^{-1}(U^\dagger |\hat{b}\rangle\langle \hat{b}|U)]
$$

It is worth noting that the single classical snapshot $\hat{\rho}=\mathcal{M}^{-1}(U^\dagger |\hat{b}\rangle\langle \hat{b}|U)$ equals $\rho$ in expectation: $\mathbb{E}[\hat{\rho}]=\rho$ despite $\mathcal{M}^{-1}$ not being a completely positive map. Repeating this procedure $N$ times results in an array of $N$ independent, classical snapshots of $\rho$ called the classical shadow:

$$
S(\rho, N) = \{ \hat{\rho}_1=\mathcal{M}^{-1}(U_1^\dagger |\hat{b}_1\rangle\langle \hat{b}_1|U_1),\cdots,\hat{\rho}_N=\mathcal{M}^{-1}(U_N^\dagger |\hat{b}_N\rangle\langle \hat{b}_N|U_N)\}
$$

Along the same lines as the example before, estimating the expectation value using classical shadows in Qadence only requires to pass the right set of parameters to the `Measurements` object:

```python exec="on" source="material-block" session="measurements" result="json"
# Classical shadows are defined up to some accuracy and confidence.
from qadence_protocols.measurements.utils_shadow import number_of_samples

shadow_options = {"accuracy": 0.1, "confidence": 0.1}
N, K = number_of_samples(observable, shadow_options["accuracy"], shadow_options["confidence"])
print("Shadow size and groups: ", N, K)
shadow_measurement = Measurements(protocol=Measurements.SHADOW, options=shadow_options)

# Run the shadow experiment.
estimated_values_shadow = shadow_measurement(model)

print(f"Estimated expectation value shadow = {estimated_values_shadow}") # markdown-exec: hide
```

## Robust shadows

Robust shadows [^4] were built upon the classical shadow scheme but have the particularity to be noise-resilient. Using an experimentally friendly calibration procedure, one can eﬃciently characterize and mitigate noises in the shadow estimation scheme, given only minimal assumptions on the experimental conditions. Such a procedure has been used in [^5] to estimate the Quantum Fisher information out of a quantum system. Note that robust shadows are equivalent to classical shadows in non-noisy settings by setting `calibration_coefficients` to $\frac{1}{3}$ for each qubit.

```python exec="on" source="material-block" session="measurements" result="json"
from qadence_protocols.measurements.calibration import zero_state_calibration

error_probability = 0.1
noise = Noise(protocol=Noise.READOUT, options={"error_probability": error_probability})

model = QuantumModel(
    circuit=circuit,
    observable=observable,
    backend=BackendName.PYQTORCH,
    diff_mode=DiffMode.GPSR,
    noise=noise
)

calibration_coefficients = zero_state_calibration(N, n_qubits=2, n_measurement_random_unitary=100, backend=model.backend, noise=noise)
# This linear transformation should give us the probability error
print(0.5 * (3.0 * calibration_coefficients + 1))

Rshadow_options = {"shadow_size": N, "shadow_groups": K, "calibration_coefficients": calibration_coefficients}
robust_shadow_measurement = Measurements(protocol=Measurements.ROBUST_SHADOW, options=Rshadow_options)
estimated_values_robust_shadow = robust_shadow_measurement(model)

print(f"Estimated expectation value shadow = {estimated_values_robust_shadow}") # markdown-exec: hide
```

### Getting measurements/shadows

If we are interested in accessing the measurements or shadows for computing different quantities of interest other than the expectation values, we can simply pass `return_expectations=False` as follows:

```python exec="on" source="material-block" session="measurements" result="json"

measurements_tomo = tomo_measurement(model, return_expectations=False)

print(measurements_tomo)
```

## References

[^1]: [Hsin-Yuan Huang, Richard Kueng and John Preskill, Predicting Many Properties of a Quantum System from Very Few Measurements (2020)](https://arxiv.org/abs/2002.08953)

[^2]: S. Aaronson. Shadow tomography of quantum states. In _Proceedings of the 50th Annual A ACM SIGACT Symposium on Theory of Computing_, STOC 2018, pages 325–338, New York, NY, USA, 2018. ACM

[^3]: Aniket Rath. Probing entanglement on quantum platforms using randomized measurements. Physics \[physics\]. Université Grenoble Alpes \[2020-..\], 2023. English. ffNNT : 2023GRALY072ff. fftel-04523142

[^4]: [Senrui Chen, Wenjun Yu, Pei Zeng, and Steven T. Flammia, Robust Shadow Estimation (2021)](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.030348)

[^5]: [Vittorio Vitale, Aniket Rath, Petar Jurcevic, Andreas Elben, Cyril Branciard, and Benoît Vermersch, Robust Estimation of the Quantum Fisher Information on a Quantum Processor (2024)](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.030338)
