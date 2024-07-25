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
from qadence import hamiltonian_factory, BackendName, DiffMode
from qadence import chain, kron, X, Z, QuantumCircuit, QuantumModel
from qadence_protocols.measurements.protocols import Measurements

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
exact_values = model.expectation(
	values=values,
)

# Run the tomography experiment.
estimated_values_tomo = tomo_measurement(
    model,
    param_values=values,
)

print(f"Exact expectation value = {exact_values}") # markdown-exec: hide
print(f"Estimated expectation value tomo = {estimated_values_tomo}") # markdown-exec: hide
```
