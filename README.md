# Lanthanide

This is Python 3 package to calculate the energy levels of multi-electron systems
populating the 4f configuration, which means the lanthanide or rare-earth ions from
Ce<sup>3+</sup> (4f<sup>1</sup>) to Yb<sup>3+</sup> (4f<sup>13</sup>). The
calculation is based on Racahs single electron unit tensor operators as developed
in my PhD thesis [1]. You find a copy of the thesis and corrections in the
folder `docs`.

The matrix elements are calculated in the space of determinantal product 
states and then transformed to the SLJM space using the chain of operators:

$$ \mathbf{S}^2 \to \mathbf{G}(R_7) \to \mathbf{G}(G_2)
\to \mathbf{L}^2 \to \mathbf{J}^2 \to \mathrm{J}_z $$

All six perturbation Hamiltonians known from the literature are used to calculate
the energy levels of a given Lathanide ion. In the order of their relative magnitude,
they are:

1. Coulomb interaction between the electrons (1st order):
$\mathbf{H}_1 = F^2 \mathbf{f}_2 + F^4 \mathbf{f}_4 + F^6 \mathbf{f}_6$,
with the radial integrals $F^2$, $F^4$, and $F^6$
and the respective angular two-electron operators $\mathbf{f}_2$, $\mathbf{f}_4$, and $\mathbf{f}_6$.
The Lanthanide package uses the keys `H1/2`, `H1/4`, and `H1/6` for radial parameters (integrals)
and angular matrices (operators).
2. Magnetic spin-orbit interaction of each electron (1st order):
$\mathbf{H}_2 = \zeta \mathbf{z}$,
with the radial integral $\zeta$ and the angular one-electron operator $\mathbf{z}$.
The Lanthanide package uses the key `H2` for the radial parameter and the angular matrix.
3. Coulomb inter-configuration interactions (2nd order):
$\mathbf{H}_3 = \alpha \mathbf{L}^2 + \beta \mathbf{G}(G_2) + \gamma \mathbf{G}(R_7)$,
with the radial integrals $\alpha$, $\beta$, and $\gamma$.
The respective effective angular two-electron operators $\mathbf{L}^2$, $\mathbf{G}(G_2)$, and $\mathbf{G}(R_7)$
are the squared operator of the total orbital angular momentum and
the Casimir operators of the symmetry groups $R_7$ and $G_2$.
The Lanthanide package uses the keys `H3/0`, `H3/1`, and `H3/2` for radial parameters and angular matrices.
4. More Coulomb inter-configuration interactions (2nd order):
$\mathbf{H}_4 = \sum_c T^c \mathbf{t}_c$, with $c = 2, 3, 4, 6, 7, 8$.
The radial integrals are $T^c$ and the respective effective angular three-electron operators
$\mathbf{t}_c$, $\mathbf{G}(G_2)$, and $\mathbf{G}(R_7)$.
The Lanthanide package uses the keys `H4/2`, `H4/3`, `H4/4`, `H4/6`, `H4/7`, and `H3/8` for radial parameters
and angular matrices.
5. Magnetic interactions on the spin of one electron with the spin (ss) or the orbital angular momentum (soo)
of another electron (1st order):
$\mathbf{H}_5 = M^0 \mathbf{m}_0 + M^2 \mathbf{m}_2 + M^4 \mathbf{m}_4$,
with the Marvin integrals $M^0$, $M^2$, and $M^4$ and the respective angular two-electron operators
$\mathbf{m}_0$, $\mathbf{m}_2$, and $\mathbf{m}_4$.
The Lanthanide package uses the keys `H5/0`, `H5/2`, and `H5/4` for radial parameters and angular matrices.
6. Magnetic inter-configuration spin-orbit interactions (2nd order):
$\mathbf{H}_6 = P^2 \mathbf{p}_2 + P^4 \mathbf{p}_4 + P^6 \mathbf{p}_6$,
with the radial integrals $P^2$, $P^4$, and $P^6$ and the respective angular two-electron operators
$\mathbf{p}_2$, $\mathbf{p}_4$, and $\mathbf{p}_6$.
The Lanthanide package uses the keys `H6/2`, `H6/4`, and `H6/6` for radial parameters and angular matrices. 

The first order perturbations are interactions inside the 4f configuration, while second order perturbations
take interactions with all other configuration into account. They are treated by effective operators 
mathematically operating inside the 4f configuration.

## Installation

This package is work-in-progress yet. Everything is available but has not
settled to its final shape yet. Expect interfaces to be changed.

To build and install the package, download the files from GitHub and run
the command

```
python -m pip install .
```

## Usage

You start by importing the most important symbols:

```
from lanthanide import Lanthanide, Coupling
```

A dictionary of radial radial parameters (integrals) looks like this:

```
radial = { "base": 327.39, "H1/2": 68576.05, "H1/4": 49972.76, "H1/6": 32415.29, "H2": 728.18,
    "H3/0": 16.99, "H3/1": -417.98, "H3/2": 1371, "H5fix": 0.19, "H6fix": 1.67 }
```

where `"base"` fixes the energy of the ground state and the special parameters `"H5fix"` and `"H6fix"` are
abbreviations for the common choices $M^2 = 0.56 M^0$, $M^4 = 0.38 M^0$ and $P^4 = 0.75 P^2$, $P^6 = 0.50 P^2$. 

You may choose operation in either the SLJM or the SLJ space (default). The latter is of special importance for
Lanthanides in glasses where the Stark splitting cannot be resolved and thus the magnetic quantum number
M of the total angular momentum does not matter. Operation in the SLJ space is much faster. Select your choice by

```
coupling = Coupling.SLJ
```

and initialize the Lanthanide ion by giving the number of 4f electrons in the range from 1 to 13:

```
ion = Lanthanide(2, coupling, radial)
print(ion)
```

The Lanthanide object builds the matrix of the total perturbation Hamiltonian and diagonalises it, which results 
in the energy level and the SLJ composition of each state in intermediate coupling.
The process is accelerated by diagonalising the much smaller J sub-spaces individually.
Each state in intermediate coupling is a mixture of different SLJ states with the same total angular momentum J.
You can access all state objects in a list with the ground state in first position.
These commands give you energy, weight factors and the respective SLJ components of the first excited state:

```
state = ion.intermediate.states[1]
print(state.energy, state.weights, state.states)
```

For each state object there is a long string representation, which you can access by `str(state)` or
`state.long(min_weight=0.0)` and a short version by `state.short()`.
The parameter `min_weight` is useful for ions with a large number of states.
It gives the minimum weight of a SLJ state to appear in the list.
This allows to show the most important components only.
A shortcut to the list of energies is the attribute `ion.energies`.
The method `ion.str_levels(min_weight=0.0)` provides a convenient way to display the energy level spectrum.

For radiative transitions inside the 4f configuration of Lanthanides only electric and magnetic
dipole moments are relevant.
The calculation of the respective transition strengths according to the Judd-Ofelt theory is based on
the reduced matrix elements
$\langle J'\parallel \mathbf{U}^{(2)}\parallel J \rangle$,
$\langle J'\parallel \mathbf{U}^{(4)}\parallel J \rangle$, and
$\langle J'\parallel \mathbf{U}^{(6)}\parallel J \rangle$ for electric and
$\langle J'\parallel \mathbf{M}\parallel J \rangle$
for magnetic dipole transitions.
The method `Lanthanide.reduced()` delivers a `Reduced` object, which contains all four squared reduced matrices 
as attributes `U2`, `U4`, `U6`, and `LS` as required for the calculation of transition strengths.
The matrix element for a transition from an initial state `i` to a final state `j` is addressed by `array[j,i]`. 
This command shows the squared elements
$|\langle J_j\parallel \mathbf{U}^{(4)}\parallel J_0 \rangle|^2$ for transitions from the ground state
to all excited states:

```
reduced = ion.reduced()
print(reduced.U4[1:, 0])
```

There is no universally accepted definition of the line strength of a transition. For electric dipole transitions,
the Lanthanide package uses the definition

$$ S_{ed} = \frac{e^2}{4\pi\varepsilon_0} \frac{1}{3(2J_i+1)} \sum\limits_{\lambda=2,4,6} 
\Omega_\lambda |\langle J_j\parallel \mathbf{U}^{(\lambda)}\parallel J_i\rangle|^2 $$

with the Judd-Ofelt parameters $\Omega_2$, $\Omega_4$, and $\Omega_6$. For magnetic dipole transitions we use  

$$ S_{md} = \frac{e^2}{16\pi\varepsilon_0 m_e^2} \frac{1}{3(2J_i+1)} 
|\langle J_j\parallel \mathbf{L}+g_s\mathbf{S}\parallel J_i\rangle|^2 $$

For a given set of Judd-Ofelt parameters you can also get the radiative line strength of each transition using
the method `line_strengths(judd_ofelt)`:

```
judd_ofelt = { "JO/2": 1.981, "JO/4": 4.645, "JO/6": 6.972 }
strength = ion.line_strengths(judd_ofelt)
print(strength.Sed[1:, 0])
print(strength.Smd[1:, 0])
```

The line strengths are often used to calculate the oscillator strength of a transition $i\to j$:

$$ f_{ij} = \frac{4\pi\varepsilon_0}{e^2} \frac{8\pi^2m_e\bar{\nu}}{h}\[\chi'_{ed}S_{ed} + \chi'_{md}S_{md}\] $$

with...

## Usage examples

See Python scripts in the directory `test`.

## License

This is free software under the MIT License.

## Reference

[1] Reinhard Caspary: "Applied Rare-Earth Spectroscopy for Fiber Laser Optimization", doctoral dissertation, Shaker,
Aachen, 2002