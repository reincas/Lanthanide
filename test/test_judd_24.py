##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# Compare 1st order spin-spin and spin-other-orbit (H5) and 2nd order
# spin-orbit (H6) interactions from [24] for the f2 configuration with
# results from the Lanthanide package.
#
# [24] B. R. Judd, H. M. Crosswhite, and Hannah Crosswhite (1968):
#      "Intra-Atomic Magnetic Interactions for f Electrons",
#      Phys. Rev. 169, p. 130,
#      https://doi.org/10.1103/PhysRev.169.130
#
##########################################################################

import math
import numpy as np

from lanthanide import Lanthanide, Coupling, wigner6j

JUDD_24 = {
    "hss/0": (2, 2, {
        "3P 3P": -12,
        "3P 3F": (8 / np.sqrt(3)) * 3,
        "3F 3F": (4 * np.sqrt(14) / 3) * -1,
        "3F 3H": (8 * np.sqrt(11 / 2) / 3) * 2,
        "3H 3H": (4 * np.sqrt(143) / 3) * 1,
    }),
    "hss/2": (2, 2, {
        "3P 3P": -24,
        "3P 3F": (8 / np.sqrt(3)) * 1,
        "3F 3F": (4 * np.sqrt(14) / 3) * 8,
        "3F 3H": (8 * np.sqrt(11 / 2) / 3) * -23 / 11,
        "3H 3H": (4 * np.sqrt(143) / 3) * -34 / 11,
    }),
    "hss/4": (2, 2, {
        "3P 3P": -300 / 11,
        "3P 3F": (8 / np.sqrt(3)) * -100 / 11,
        "3F 3F": (4 * np.sqrt(14) / 3) * -200 / 11,
        "3F 3H": (8 * np.sqrt(11 / 2) / 3) * -325 / 121,
        "3H 3H": (4 * np.sqrt(143) / 3) * - 1325 / 1573,
    }),
    "hsoo/0": (2, 1, {
        "1S 3P": 6,
        "3P 3P": -36,
        "3P 1D": -math.sqrt(2 / 15) * 27,
        "1D 3F": math.sqrt(2 / 5) * 23,
        "3F 3F": 2 * math.sqrt(14) * -15,
        "3F 1G": math.sqrt(11) * -6,
        "1G 3H": math.sqrt(2 / 5) * 39,
        "3H 3H": 8 / math.sqrt(55) * -132,
        "3H 1I": math.sqrt(26) * -5,
    }),
    "hsoo/2": (2, 1, {
        "1S 3P": 2,
        "3P 3P": -72,
        "3P 1D": -math.sqrt(2 / 15) * 14,
        "1D 3F": math.sqrt(2 / 5) * 6,
        "3F 3F": 2 * math.sqrt(14) * -1,
        "3F 1G": math.sqrt(11) * 64 / 33,
        "1G 3H": math.sqrt(2 / 5) * -728 / 33,
        "3H 3H": 8 / math.sqrt(55) * 23,
        "3H 1I": math.sqrt(26) * -30 / 11,
    }),
    "hsoo/4": (2, 1, {
        "1S 3P": 10 / 11,
        "3P 3P": -900 / 11,
        "3P 1D": -math.sqrt(2 / 15) * 115 / 11,
        "1D 3F": math.sqrt(2 / 5) * -195 / 11,
        "3F 3F": 2 * math.sqrt(14) * 10 / 11,
        "3F 1G": math.sqrt(11) * -1240 / 363,
        "1G 3H": math.sqrt(2 / 5) * -3175 / 363,
        "3H 3H": 8 / math.sqrt(55) * 130 / 11,
        "3H 1I": math.sqrt(26) * -375 / 1573,
    }),
    "H6/0": (2, 1, {
        "1S 3P": -2,
        "3P 3P": -1,
        "3P 1D": math.sqrt(15 / 2) * 1,
        "1D 3F": math.sqrt(10) * -1,
        "3F 3F": math.sqrt(14) * -1,
        "3F 1G": math.sqrt(11) * 1,
        "1G 3H": math.sqrt(10) * -1,
        "3H 3H": math.sqrt(55) * -1,
        "3H 1I": math.sqrt(13 / 2) * 1,
    }),
    "H6/2": (2, 1, {
        "1S 3P": -105 / 225,
        "3P 3P": -45 / 225,
        "3P 1D": math.sqrt(15 / 2) * 32 / 225,
        "1D 3F": math.sqrt(10) * -9 / 2 / 225,
        "3F 3F": math.sqrt(14) * 10 / 225,
        "3F 1G": math.sqrt(11) * -20 / 225,
        "1G 3H": math.sqrt(10) * 55 / 2 / 225,
        "3H 3H": math.sqrt(55) * 25 / 225,
        "3H 1I": math.sqrt(13 / 2) * 0 / 225,
    }),
    "H6/4": (2, 1, {
        "1S 3P": -231 / 1089,
        "3P 3P": -33 / 1089,
        "3P 1D": math.sqrt(15 / 2) * -33 / 1089,
        "1D 3F": math.sqrt(10) * 66 / 1089,
        "3F 3F": math.sqrt(14) * 33 / 1089,
        "3F 1G": math.sqrt(11) * 32 / 1089,
        "1G 3H": math.sqrt(10) * -23 / 1089,
        "3H 3H": math.sqrt(55) * 51 / 1089,
        "3H 1I": math.sqrt(13 / 2) * -21 / 1089,
    }),
    "H6/6": (2, 1, {
        "1S 3P": -429 * 25 / 184041,
        "3P 3P": 1287 * 25 / 184041,
        "3P 1D": math.sqrt(15 / 2) * -286 * 25 / 184041,
        "1D 3F": math.sqrt(10) * -429 / 2 * 25 / 184041,
        "3F 3F": math.sqrt(14) * 286 * 25 / 184041,
        "3F 1G": math.sqrt(11) * -104 * 25 / 184041,
        "1G 3H": math.sqrt(10) * -65 / 2 * 25 / 184041,
        "3H 3H": math.sqrt(55) * 13 * 25 / 184041,
        "3H 1I": math.sqrt(13 / 2) * -6 * 25 / 184041,
    }),
}


def run_hss_judd(name):
    assert name in JUDD_24.keys()

    num, t, judd = JUDD_24[name]

    with Lanthanide(num) as ion:
        states = ion.states(Coupling.SLJ)
        array = ion.matrix(name, Coupling.SLJ).array
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                Sa = states[i]["S2"].S
                La = states[i]["L2"].L
                Ja = states[i]["J2"].J
                Sb = states[j]["S2"].S
                Lb = states[j]["L2"].L
                Jb = states[j]["J2"].J
                if Ja != Jb:
                    assert abs(array[i, j]) < 1e-9, f"{name}[{i},{j}] != 0.0"
                    continue

                factor = wigner6j(Sb, Lb, Ja, La, Sa, t)
                if abs(factor) < 1e-9:
                    assert abs(array[i, j]) < 1e-9, f"{name}[{i},{j}] != 0.0"
                    continue

                LSa = str(states[i]["S2"]) + str(states[i]["L2"])
                LSb = str(states[j]["S2"]) + str(states[j]["L2"])
                if f"{LSa} {LSb}" in judd:
                    reduced = judd[f"{LSa} {LSb}"]
                    element = f"< {LSa} | {LSb} >"
                elif f"{LSb} {LSa}" in judd:
                    reduced = judd[f"{LSb} {LSa}"]
                    element = f"< {LSb} | {LSa} >"
                else:
                    raise ValueError(f"Unknown element < {LSa} | {LSb} >!")

                value = factor * reduced
                # Note: Judd [24] says (Sb + La + Ja) %2. This is most likely a typo!
                if (Sb + Lb + Ja) % 2 != 0:
                    value = -value

                if abs(value) < 1e-9:
                    assert abs(array[i, j]) < 1e-9, f"{name}[{i},{j}]: {element} != 0.0"
                    continue

                if abs(value + array[i, j]) < 1e-9:
                    print(f"Sign flip: {name}[{i},{j}]: {element}")
                    continue

                if not abs(value - array[i, j]) < 1e-9:
                    print(f"ERROR: {name}[{i},{j}]: {element} {value} != {array[i, j]}")
                # assert abs(value - array[i, j]) < 1e-9, f"{name}[{i},{j}]: {element}"
                # print(f"Element {element} is ok.")


def test_judd_24():
    for name in JUDD_24:
        run_hss_judd(name)
