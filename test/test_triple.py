##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# Compare 2st order Coulomb interaction matrix elements for three
# electrons (H4) from the literature for the f3 configuration with
# results from the Lanthanide package.
#
##########################################################################

import itertools
import pytest

from lanthanide import Lanthanide, Coupling
from data_triple import SOURCES, STATES, TRIPLE


@pytest.mark.parametrize("data_key", TRIPLE.keys())
def test_triple(data_key):
    # Select data set
    assert data_key in TRIPLE
    data = TRIPLE[data_key]

    # Test source link
    assert "source" in data
    assert data["source"] in SOURCES

    # Number of f electrons
    assert "num" in data
    num = data["num"]

    # Name of tensor operator
    assert "name" in data
    name = data["name"]

    # Common factor of all matrix elements
    assert "factor" in data
    factor = float(data["factor"])

    # SL matrix elements
    assert "elements" in data
    elements = data["elements"]

    # Compare tensor operator matrix to the calculation
    success = True
    with Lanthanide(num) as ion:
        states = ion.states(Coupling.SLJ)
        array = ion.matrix(name, Coupling.SLJ).array

        # Initialize set of SL states and list of phase signs
        sl_states = set()
        phases = []

        # Compare every matrix element
        for i in range(array.shape[0]):
            for j in range(i + 1):

                # Test for zero if final and initial states differ in quantum number J
                Ja = states[i]["J2"].J
                Jb = states[j]["J2"].J
                if Ja != Jb:
                    assert abs(array[i, j]) < 1e-9
                    continue

                # Get respective reduced SL matrix element
                LSa = STATES[num][f"{states[i]["S2"]}{states[i]["L2"]}{states[i]["GR/7"]}{states[i]["GG/2"]}"]
                LSb = STATES[num][f"{states[j]["S2"]}{states[j]["L2"]}{states[j]["GR/7"]}{states[j]["GG/2"]}"]
                if f"{LSa} {LSb}" in elements:
                    element = elements[f"{LSa} {LSb}"]
                    element_str = f"< {LSa} | {LSb} >"
                elif f"{LSb} {LSa}" in elements:
                    element = elements[f"{LSb} {LSa}"]
                    element_str = f"< {LSb} | {LSa} >"
                else:
                    if not abs(array[i, j]) < 1e-9:
                        success = False
                        print(f"ERROR: unknown SL element < {LSa} | {LSb} >!")
                    continue
                value = factor * element

                # SL diagonal element, sign always +1
                if LSa == LSb:
                    if not abs(value - array[i, j]) < 1e-9:
                        success = False
                        print(f"ERROR: {name}[{i},{j}]: {element_str} {value} != {array[i, j]}")
                    continue

                # Both SL states with same sign
                if abs(value - array[i, j]) < 1e-9:
                    sl_states.add(states[i].long())
                    sl_states.add(states[j].long())
                    phases.append((states[i].long(), states[j].long(), 1))
                    continue

                # Both SL states with opposite sign
                if abs(value + array[i, j]) < 1e-9:
                    sl_states.add(states[i].long())
                    sl_states.add(states[j].long())
                    phases.append((states[i].long(), states[j].long(), -1))
                    continue

                # Different magnitude of given and calculated matrix element
                success = False
                print(f"ERROR: {name}[{i},{j}]: {element_str} {value} != {array[i, j]}")

        # Stop if one element test failed
        assert success

        # Test for state phases which match the detected matrix element signs
        sl_states = tuple(sl_states)
        success = False
        for signs in itertools.product((+1, -1), repeat=len(sl_states)):
            if all(signs[sl_states.index(LSa)] * signs[sl_states.index(LSb)] == sign for LSa, LSb, sign in phases):
                success = True
                break
        if not success:
            print(f"ERROR: sign adjustment failed for {data_key}!")
        assert success
