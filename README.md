# Lanthanide

Python 3 package to calculate the energy levels of multi-electron systems
populating the f-shell, which means the lanthanide or rare-earth ions from
Ce<sup>3+</sup> (4f<sup>1</sup>) to Yb<sup>3+</sup> (4f<sup>13</sup>). The
calculation is based on Racahs single electron
unit tensor operators. The states are transformed from determinantal product
states to SLJM states. All six perturbation Hamiltonians known from the
literature are included. For a given set of radial integral values, the
total perturbation Hamiltonian can be diagonalised to intermediate coupling
to get the energy levels. For lanthanides in glasses the calculation speed
is improved by using reduced SLJ states. All reduced matrix elements of the
electric and magnetic dipole operators in intermediate coupling are provided
for Judd-Ofelt fits. For given Judd-Ofelt parameters the radiative line strength
of each transition can be obtained.

This package is the successor of the Python package I developed for my PhD
thesis until 2002. A copy of the thesis is in the folder `docs`. 

## Installation

This package is work-in-progress yet. Everything is available but has not
settled to its final shape yet. Expect interfaces to be changed.

To build and install the package, download the files from GitHub and run
the command

```
python -m pip install .
```

## Usage examples

See Python scripts in the directory `test`.

## License

This is free software under the MIT License.