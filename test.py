##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import time
import numpy as np

from lanthanide import Lanthanide, Coupling, RADIAL, reduced_matrix


def show_ion(ion):
    print()
    print("Ion:", ion)
    print(f"  electrons: {ion.num}")
    print(f"  states: {len(ion.product_states)}")
    for num, group in enumerate(ion.single):
        print(f"  {num + 1}-electron elements: {len(group['elements'])} ({len(group['indices'])})")


def show_symmetry(ion):
    print("Symmetry quantum numbers:")
    t = time.time()
    states = ion.states(Coupling.SLJM)
    for name in ("S2", "GR/7", "GG/2", "L2", "tau", "J2", "Jz"):
        sym_list = states[name]
        print(f"  {sym_list.object_class.symbol}: {sym_list}")
    t = time.time() - t
    print(f"  / quantum duration: {t:.1f} s /")


def show_states(ion, coupling):
    t = time.time()
    print("SLJ states:")
    for state in ion.states(coupling):
        print(f"  {state}")
    t = time.time() - t
    print(f"  / states duration: {t:.1f} s /")


def show_hamiltonians(ion):
    t = time.time()
    print("Hamiltonians:")
    for values in [("H1", 2, 4, 6), ("H2", None), ("H3", 0, 1, 2), ("H4", 2, 3, 4, 6, 7, 8), ("H5", 0, 2, 4),
                   ("H6", 2, 4, 6)]:
        operator, args = values[0], values[1:]

        results = []
        for arg in args:
            if arg is None:
                name = operator
            else:
                name = f"{operator}/{arg}"
            matrix = ion.matrix(name, Coupling.SLJ)
            zero = "---" if np.all(np.abs(matrix.array) < 1e-5) else "###"
            if arg is None:
                results.append(f"  {zero}")
            else:
                results.append(f"{arg}/{zero}")
        print(f"  {operator}: {' '.join(results)}")
    t = time.time() - t
    print(f"  / hamiltonians duration: {t:.1f} s /")


def show_levels(ion, min_weight=0.0):
    print("Energy levels:")
    for state in ion.intermediate.states:
        print(f"  {state.energy:6.0f} cm-1  | {state.long(min_weight)} >")


def show_reduced(ion):
    print("Reduced states:")
    states = ion.intermediate.states
    initial = f"{states[0].short():10s}"
    print(f"  {initial} |  <U2>^2  |  <U4>^2  |  <U6>^2  |  <MD>^2")
    print(57 * "-")
    reduced = np.array([ion.reduced()[key][:, 0] for key in ("U2", "U4", "U6", "LS")], dtype=float).T
    for i in range(1, reduced.shape[0]):
        u2, u4, u6, ls = reduced[i, :]
        print(f"  {states[i].short():10s} | {u2:7.4f}  | {u4:7.4f}  | {u6:7.4f}  | {ls:7.4f}")


if __name__ == "__main__":
    with Lanthanide(12, radial=RADIAL["Tm3+/ZBLAN"]) as ion:
        show_ion(ion)
        # show_symmetry(ion)
        # show_states(ion, Coupling.SLJ)
        # show_hamiltonians(ion)
        show_levels(ion, 0.05)
        # show_reduced(ion)
    print("Done.")
