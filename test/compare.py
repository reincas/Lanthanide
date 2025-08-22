##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# Compare matrices with literature values.
#
##########################################################################

import math
import numpy as np

from lanthanide import Lanthanide, Coupling, RADIAL, reduced_matrix


def judd_hss(k, names):
    values = {
        0: {
            "3P 3P": -12,
            "3P 3F": (8 / np.sqrt(3)) * 3,
            "3F 3F": (4 * np.sqrt(14) / 3) * -1,
            "3F 3H": (8 * np.sqrt(11 / 2) / 3) * 2,
            "3H 3H": (4 * np.sqrt(143) / 3) * 1,
        },
        2: {
            "3P 3P": -24,
            "3P 3F": (8 / np.sqrt(3)) * 1,
            "3F 3F": (4 * np.sqrt(14) / 3) * 8,
            "3F 3H": (8 * np.sqrt(11 / 2) / 3) * -23 / 11,
            "3H 3H": (4 * np.sqrt(143) / 3) * 34 / 11,
        },
        4: {
            "3P 3P": -300 / 11,
            "3P 3F": (8 / np.sqrt(3)) * -100 / 11,
            "3F 3F": (4 * np.sqrt(14) / 3) * -200 / 11,
            "3F 3H": (8 * np.sqrt(11 / 2) / 3) * -325 / 121,
            "3H 3H": (4 * np.sqrt(143) / 3) * - 1325 / 1573,
        },
    }
    values = values[k]
    print(" | ".join([f"{key}: {value / math.sqrt(5):.3f}" for key, value in values.items()]))
    s5 = math.sqrt(5)
    elements = {}
    for key in values:
        value = values[key] / s5
        elements[key] = value
        key1, key2 = key.split()
        elements[f"{key2} {key1}"] = value

    result = np.zeros((len(names), len(names)), dtype=float)
    for i in range(len(names)):
        for j in range(len(names)):
            key = f"{names[i][:-1]} {names[j][:-1]}"
            if key in elements:
                result[i, j] = elements[key]
    return result


if __name__ == "__main__":

    with Lanthanide(2) as ion:
        print(ion)

    print("Done.")
