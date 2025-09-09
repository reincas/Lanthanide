##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# Compare calculated matrix elements of "H1", "H2", and "H3" and squared
# reduced matrix elements of the unit tensor operator of GSA transitions
# with measured and calculated spectroscopic data on all lanthanide ions
# with 2-12 electrons from the Carnall 1968 paper series "Electronic
# Energy Levels in the Trivalent Lanthanide Aquo Ions".
#
# Some corrections had to be applied to the published data to correct
# typos or incorrect results.
#
##########################################################################

import math
import numpy as np
import pytest

from lanthanide import Lanthanide, Coupling, HalfInt
from basic import SOURCES, RADIAL


def correct(name, data, corrections):
    """ Apply corrections with given name to the given dataset in place. """

    for key, index, old_value, new_value in corrections:
        if key != name:
            continue
        assert data[index] == old_value
        data[index] = new_value


@pytest.mark.parametrize("name", [key for key in RADIAL if "blocking" not in RADIAL[key]])
def test_basic(name):
    # Select data set
    assert name in RADIAL
    data = RADIAL[name]

    # Test source link
    assert "source" in data
    assert data["source"] in SOURCES

    # Data correction list
    corrections = data.get("correct", [])

    # Number of f electrons
    assert "num" in data
    num = data["num"]

    # Get, correct, and sort energy levels
    assert "energies" in data
    assert isinstance(data["energies"], list)
    energies = list(data["energies"])
    correct("energies", energies, corrections)
    indices = np.argsort(energies)
    energies = sorted(map(float, energies))

    # Get and correct radial integrals
    assert "radial" in data
    assert isinstance(data["radial"], dict)
    radial = dict(data["radial"])
    correct("radial", radial, corrections)
    if not "base" in radial:
        radial["base"] = energies[0]

    # Get, correct, and order values of the J quantum number
    assert "J" in data
    assert isinstance(data["J"], list)
    assert len(data["J"]) == len(energies)
    J = list(data["J"])
    correct("J", J, corrections)
    J = [J[i] for i in indices]
    if num % 2 != 0:
        J = [HalfInt(j) for j in J]

    # Get, correct, and order the optional squared matrix elements of unit tensor operators
    if "U2" in data:
        U = {}
        for key in ("U2", "U4", "U6"):
            assert key in data
            assert isinstance(data[key], list)
            assert len(data[key]) == len(energies) - 1
            U[key] = list(data[key])
            correct(key, U[key], corrections)
            U[key] = [U[key][i - 1] for i in indices[1:]]
        U2 = U["U2"]
        U4 = U["U4"]
        U6 = U["U6"]
    else:
        U2 = U4 = U6 = None

    # Compare given data to the calculation
    success = True
    with Lanthanide(num, coupling=Coupling.SLJ, radial=radial) as ion:

        # Compare given energy levels with calculation and determine mean quadratic deviation
        mean = 0.0
        for i in range(len(energies)):
            diff = energies[i] - ion.energies[i]
            if ion.intermediate[i].J != J[i] or abs(diff) > 2.5:
                success = False
                ref = f"J={J[i]}: {energies[i]:.0f}"
                level = ion.intermediate[i].short()
                calc = f"{level}: {ion.energies[i]:.0f}"
                print(f"*** | level {i} | ref {ref} | calc {calc} | diff {diff:.1f} ***")
            mean += pow(diff, 2)
        mean = math.sqrt(mean) / len(energies)
        assert mean < 0.5

        # Compare squared reduced matrix elements with calculation
        if U2 is not None:
            reduced = ion.line_reduced()
            for i in range(1, len(energies)):
                du2 = abs(reduced.U2[0, i] - U2[i - 1])
                du4 = abs(reduced.U4[0, i] - U4[i - 1])
                du6 = abs(reduced.U6[0, i] - U6[i - 1])
                if max(du2, du4, du6) >= 0.0005:
                    success = False
                    ref = f"J={J[i]}: {U2[i - 1]:.4f} {U4[i - 1]:.4f} {U6[i - 1]:.4f}"
                    level = ion.intermediate[i].short()
                    calc = f"{level}: {reduced.U2[0, i]:.4f} {reduced.U4[0, i]:.4f} {reduced.U6[0, i]:.4f}"
                    print(f"*** | level {i} | ref {ref} | calc {calc} ***")

        #print(f"===> {name} mean energy dev. == {mean:.1f} cm^-1")
    assert success
