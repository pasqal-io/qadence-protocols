Errors may show up in various forms when trying to extract information from a quantum circuit. These include coherent and incoherent errors that effect gate execution, readout error during measurement and statistical errors.


- Incoherent errors are modelled through noisy channels, _i.e._ depolarizing, dephazing and erasure.
- Coherent errors are modelled as parameteric noise sampled from a distribution that shifts the operation to be performed.
- Measurement errors are introduced as bitflip operators that occur in the computational basis modelled using a confusion matrix.
- Statistical noises are introduced through finite sampling

## Incoherent errors

### Digital errors

Errors here are modelled to be executed at the end of each gate execution. The errors might be local or global. Global errors cannot be factorized as tensor products of independent channels. Implementation of digital errors is now supported on [PyQ](https://github.com/pasqal-io/pyqtorch)

### Analog errors
Errors are incorporated as a part of the open system dynamics coupled with the effects of environment showing up as Krauss operators. The dynamics of an open quantum system happens through the Linbladian equation defined for markovian systems (memoryless systems). Its given by

$$
    \frac{d\rho}{dt} = L[\rho] = -i[H,\rho] + \sum \gamma_i \bigg(L_i\rho L_i^{\dagger} - \frac{1}{2} \{L_i^{\dagger}L_i,\rho\}\bigg)
$$

We use qutip `mesolve` for the computation as a Pulser backend invoked for analog circuits written in Qadence. Qadence Protocols offers a number of noise mitigation techniques to achieve better accuracy of simulation outputs. Currently supported methods mitigate primarily measurement readout errors. For analog blocks we support mitigating depolarizing and dephasing noise via Zero NoiseHandler Extrapolation.
