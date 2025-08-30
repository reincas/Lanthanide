##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import math
import numpy as np

from lanthanide import product_states, init_single, calc_unit, SymmetryList, ORBITAL

class DummyLanthanide:
    def __init__(self, num: int):
        self.num = num
        self.product = product_states(num)
        self.single = init_single(None, None, self.product)


def test_symmetry():
    ion = DummyLanthanide(3)

    array_s2 = 1.5 * (calc_unit(ion, "TT/a/1") + 2 * calc_unit(ion, "TT/b/1"))
    values, vectors = np.linalg.eigh(array_s2)
    sym = SymmetryList(values, "S2")
    assert sym.count() == {'2': 224, '4': 140}

    l = ORBITAL
    factor = l * (l + 1) * (2 * l + 1)
    array_l2 = factor * (calc_unit(ion, "UU/a/1") + 2 * calc_unit(ion, "UU/b/1"))
    values, vectors = np.linalg.eigh(array_l2)
    sym = SymmetryList(values, "L2")
    assert sym.count() == {'S': 4, 'P': 6, 'D': 40, 'F': 56, 'G': 72, 'H': 44, 'I': 78, 'K': 30, 'L': 34}

    factor = math.sqrt(1.5 * l * (l + 1) * (2 * l + 1))
    array_ls = factor * (calc_unit(ion, "UT/a/1") + 2 * calc_unit(ion, "UT/b/1"))
    array_j2 = array_l2 + 2 * array_ls + array_s2
    values, vectors = np.linalg.eigh(array_j2)
    sym = SymmetryList(values, "J2")
    assert sym.count() == {'1/2': 4, '3/2': 24, '5/2': 42, '7/2': 56, '9/2': 70, '11/2': 60,
                           '13/2': 42, '15/2': 48, '17/2': 18}
