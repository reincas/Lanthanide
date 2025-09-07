##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# Compare 1st order spin-spin and spin-other-orbit (H5) and 2nd order
# spin-orbit (H6) interactions from [24] and [25] for the f12
# configuration with results from the Lanthanide package.
#
# [24] B. R. Judd, H. M. Crosswhite, and Hannah Crosswhite (1968):
#      "Intra-Atomic Magnetic Interactions for f Electrons",
#      Phys. Rev. 169, p. 130,
#      https://doi.org/10.1103/PhysRev.169.130
#
# [25] W. T. Carnall, P. R. Fields, J. Morrison, R. Sarup (1970):
#      "Absorption Spectrum of Tm3+:LaF3",
#      J. Chem. Phys. 52, pp. 4054–4059,
#      https://doi.org/10.1063/1.1673608
#
##########################################################################

import math
import numpy as np

from lanthanide import Lanthanide, Coupling, wigner6j, build_hamilton

REFERENCE = {
    "hss/0": (12, 2, {  # Table I in [24]
        "3P 3P": -12,
        "3P 3F": (8 / np.sqrt(3)) * 3,
        "3F 3F": (4 * np.sqrt(14) / 3) * -1,
        "3F 3H": (8 * np.sqrt(11 / 2) / 3) * 2,
        "3H 3H": (4 * np.sqrt(143) / 3) * 1,
    }),
    "hss/2": (12, 2, {  # Table I in [24]
        "3P 3P": -24,
        "3P 3F": (8 / np.sqrt(3)) * 1,
        "3F 3F": (4 * np.sqrt(14) / 3) * 8,
        "3F 3H": (8 * np.sqrt(11 / 2) / 3) * -23 / 11,
        "3H 3H": (4 * np.sqrt(143) / 3) * -34 / 11,
    }),
    "hss/4": (12, 2, {  # Table I in [24]
        "3P 3P": -300 / 11,
        "3P 3F": (8 / np.sqrt(3)) * -100 / 11,
        "3F 3F": (4 * np.sqrt(14) / 3) * -200 / 11,
        "3F 3H": (8 * np.sqrt(11 / 2) / 3) * -325 / 121,
        "3H 3H": (4 * np.sqrt(143) / 3) * - 1325 / 1573,
    }),
    "hsoo/0": (12, 1, {  # Table III in [25]
        "3P 3P": 30,
        "3P 1S": 138,
        "3P 1D": -1 / math.sqrt(30) * 1044,
        "3F 3F": math.sqrt(14) * 36,
        "3F 1D": math.sqrt(10) * 353 / 5,
        "3F 1G": math.sqrt(11) * -72,
        "3H 3H": 1 / math.sqrt(55) * 2574,
        "3H 1G": 1 / math.sqrt(10) * 738,
        "3H 1I": math.sqrt(26) * -38,
    }),
    "hsoo/2": (12, 1, {  # Table III in [25]
        "3P 3P": -78,
        "3P 1S": -10,
        "3P 1D": -1 / math.sqrt(30) * -62,
        "3F 3F": math.sqrt(14) * -8,
        "3F 1D": math.sqrt(10) * -24 / 5,
        "3F 1G": math.sqrt(11) * 262 / 33,
        "3H 3H": 1 / math.sqrt(55) * -146,
        "3H 1G": 1 / math.sqrt(10) * -3436 / 33,
        "3H 1I": math.sqrt(26) * 3 / 11,
    }),
    "hsoo/4": (12, 1, {  # Table III in [25]
        "3P 3P": -930 / 11,
        "3P 1S": -50 / 11,
        "3P 1D": -1 / math.sqrt(30) * -20,
        "3F 3F": math.sqrt(14) * -10 / 11,
        "3F 1D": math.sqrt(10) * -69 / 11,
        "3F 1G": math.sqrt(11) * -250 / 363,
        "3H 3H": 1 / math.sqrt(55) * -610 / 11,
        "3H 1G": 1 / math.sqrt(10) * -16250 / 363,
        "3H 1I": math.sqrt(26) * 1770 / 1573,
    }),
    "H6/2": (12, 1, {  # Table IV in [25]
        "3P 3P": -150 / 225,
        "3P 1S": -315 / 225,
        "3P 1D": math.sqrt(15 / 2) * 137 / 225,
        "3F 3F": -math.sqrt(14) * 95 / 225,
        "3F 1D": -math.sqrt(10) * 219 / 2 / 225,
        "3F 1G": math.sqrt(11) * 85 / 225,
        "3H 3H": -math.sqrt(55) * 80 / 225,
        "3H 1G": -math.sqrt(10) * 155 / 2 / 225,
        "3H 1I": math.sqrt(13 / 2) * 105 / 225,
    }),
    "H6/4": (12, 1, {  # Table IV in [25]
        "3P 3P": -264 / 1089,
        "3P 1S": -693 / 1089,
        "3P 1D": math.sqrt(15 / 2) * 198 / 1089,
        "3F 3F": -math.sqrt(14) * 198 / 1089,
        "3F 1D": -math.sqrt(10) * 165 / 1089,
        "3F 1G": math.sqrt(11) * 263 / 1089,
        "3H 3H": -math.sqrt(55) * 180 / 1089,
        "3H 1G": -math.sqrt(10) * 254 / 1089,
        "3H 1I": math.sqrt(13 / 2) * 210 / 1089,
    }),
    "H6/6": (12, 1, {  # Table IV in [25]
        "3P 3P": 858 * 25 / 184041,
        "3P 1S": -1287 * 25 / 184041,
        "3P 1D": math.sqrt(15 / 2) * 143 * 25 / 184041,
        "3F 3F": -math.sqrt(14) * 143 * 25 / 184041,
        "3F 1D": -math.sqrt(10) * 1287 / 2 * 25 / 184041,
        "3F 1G": math.sqrt(11) * 325 * 25 / 184041,
        "3H 3H": -math.sqrt(55) * 416 * 25 / 184041,
        "3H 1G": -math.sqrt(10) * 923 / 2 * 25 / 184041,
        "3H 1I": math.sqrt(13 / 2) * 423 * 25 / 184041,
    }),
}


def run_carnall(name):
    assert name in REFERENCE.keys()

    num, t, judd = REFERENCE[name]

    with Lanthanide(num) as ion:
        states = ion.states(Coupling.SLJ)
        array = ion.matrix(name, Coupling.SLJ).array
        for i in range(array.shape[0]):
            for j in range(i + 1):
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


def show_levels(ion, min_weight=0.0):
    print("Energy levels:")
    for state in ion.intermediate.states:
        print(f"  {state.energy:6.0f} cm-1  | {state.long(min_weight)} >")


def normalise_radial(radial):
    keys = list(radial.keys())
    new_radial = {}
    if "base" in keys:
        new_radial["base"] = radial["base"]
        keys.remove("base")
    for key in keys:
        if key[:2] in ("H1", "H2", "H3", "H4", "H5", "H6"):
            new_radial[key] = radial[key]
            keys.remove(key)
    for key in keys:
        if key[:3] in ("F_0", "F^0", "F^2", "F^4", "F^6"):
            new_radial[f"H1/{key[-1:]}"] = radial[key]
        if key == "F_2":
            new_radial[f"H1/2"] = radial[key] * 225
        if key == "F_4":
            new_radial[f"H1/4"] = radial[key] * 1089
        if key == "F_6":
            new_radial[f"H1/6"] = radial[key] * 184041 / 25
        if key == "P_2":
            new_radial[f"H6/2"] = radial[key] * 225
        if key == "P_4":
            new_radial[f"H6/4"] = radial[key] * 1089
        if key == "P_6":
            new_radial[f"H6/6"] = radial[key] * 184041 / 25
    if "E^1" in keys:
        A = np.array([[1, 9 / 7, 0, 0],
                      [0, 1 / 42, 143 / 42, 11 / 42],
                      [0, 1 / 77, -130 / 77, 4 / 77],
                      [0, 1 / 462, 5 / 66, -1 / 66]])
        A[1, :] *= 225
        A[2, :] *= 1089
        A[3, :] *= 184041 / 25
        if "E^0" in keys:
            F0, F2, F4, F6 = A @ np.array([radial[f"E^{i}"] for i in range(4)])
            for i in range(4):
                keys.remove(f"E^{i}")
        else:
            F0 = None
            A = A[1:, 1:]
            F2, F4, F6 = A @ np.array([radial[f"E^{i}"] for i in range(1, 4)])
            for i in range(1, 4):
                keys.remove(f"E^{i}")
        if F0 is not None:
            new_radial[f"H1/0"] = F0
        new_radial[f"H1/2"] = F2
        new_radial[f"H1/4"] = F4
        new_radial[f"H1/6"] = F6
    if len(keys) != 0:
        raise ValueError(f"Unknown radial integrals: {", ".join(keys)}!")
    return new_radial


def my_carnall_25():
    for name in REFERENCE:
        run_carnall(name)


my_carnall_25()

Ex = np.array([12005, 6911.8, 33.728, 675.28])
E = np.array([11492, 6737.5, 33.643, 681.22])
A = np.array(
    [[1, 9 / 7, 0, 0], [0, 1 / 42, 143 / 42, 11 / 42], [0, 1 / 77, -130 / 77, 4 / 77], [0, 1 / 462, 5 / 66, -1 / 66]])
A[1, :] *= 225
A[2, :] *= 1089
A[3, :] *= 184041 / 25
F0x, F2x, F4x, F6x = A @ Ex
F0, F2, F4, F6 = A @ E
RADIALx = {"base": 100 - 0.16, "H1/2": F2x, "H1/4": F4x, "H1/6": F6x,
           "H2": 2593.9,
           "H3/0": 9.475, "H3/1": -601.09, "H3/2": 1395.1,
           "H5/0": 5.002, "H5/2": 2.801, "H5/4": 1.901,
           "H6/2": 3.915 * 225, "H6/4": 0.090 * 1089, "H6/6": 0.070 * 184041 / 25}
RADIAL = {"base": 119, "H1/2": F2, "H1/4": F4, "H1/6": F6,
          "H2": 2633.0,
          "H3/0": 13.124, "H3/1": -743.02, "H3/2": 1992.2}
# ENERGIES = np.array(
#     [119.0, 5835.540684891865, 8319.907212940801, 12700.947751885513, 14586.165212939843, 15149.755938613787,
#      21365.2145353792, 28097.684808539343, 34865.37575921367, 35602.582634112245, 36563.675212941016, 38314.83222500194,
#      74406.57979176793])
MEAS = np.array([100, 5858, 8336, 12711, 14559, 15173, 21352, 28061, 34886, 35571.5, 36559, 38344, 74450])
CALC = np.array([119, 5835, 8320, 12701, 14586, 15150, 21366, 28098, 34866, 35604, 36562, 38315, 74450])
CALCx = MEAS - [0.16, -0.08, -2.1, 1.4, -0.8, -3.4, 4.4, 9.2, -4.2, 1.8, -6.7, 0.25, 0.0]
print()
with (Lanthanide(12, radial=RADIALx) as ion):
    print(ion)
    show_levels(ion, 0.0)
    print([f"{energy:.2f}" for energy in CALCx - ion.energies])
    ion.set_radial(RADIAL)
    print([f"{energy:.2f}" for energy in CALC - ion.energies])
