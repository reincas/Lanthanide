##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import time
import numpy as np

from lanthanide import Lanthanide, Coupling

def show_ion(ion, t):
    print()
    print("Ion:", ion)
    print(f"  electrons: {ion.num}")
    print(f"  states: {len(ion.product_states)}")
    for num, group in enumerate(ion.single):
        print(f"  {num + 1}-electron elements: {len(group['elements'])} ({len(group['indices'])})")
    print(f"  / ion duration: {t:.1f} s /")

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


if __name__ == "__main__":
    t = time.time()
    with Lanthanide(2) as ion:
        t = time.time() - t
        #print(ion)

        show_ion(ion, t)
        #show_symmetry(ion)
        #show_states(ion, Coupling.SLJ)
        show_hamiltonians(ion)

    print("Done.")
