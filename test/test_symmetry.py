##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import math
import numpy as np

from lanthanide import product_states, single_elements, SingleElements, calc_unit, SymmetryList, ORBITAL

class DummyLanthanide:
    def __init__(self, num: int):
        self.num = num
        self.product_states = states = product_states(num)
        one, two, three = single_elements(states)
        group = {
            "one": {"indices": one[0], "elements": one[1]},
            "two": {"indices": two[0], "elements": two[1]},
            "three": {"indices": three[0], "elements": three[1]},
        }
        self.single = SingleElements(group, states)


def test_symmetry():
    ion = DummyLanthanide(3)

    array_s2 = 1.5 * (calc_unit(ion, "TT/a/1") + 2 * calc_unit(ion, "TT/b/1"))
    values, vectors = np.linalg.eigh(array_s2)
    sym = SymmetryList(values, "S2")
    assert sym.count() == {2: 224, 4: 140}

    l = ORBITAL
    factor = l * (l + 1) * (2 * l + 1)
    array_l2 = factor * (calc_unit(ion, "UU/a/1") + 2 * calc_unit(ion, "UU/b/1"))
    values, vectors = np.linalg.eigh(array_l2)
    sym = SymmetryList(values, "L2")
    assert sym.count() == {0: 4, 1: 6, 2: 40, 3: 56, 4: 72, 5: 44, 6: 78, 7: 30, 8: 34}

    factor = math.sqrt(1.5 * l * (l + 1) * (2 * l + 1))
    array_ls = factor * (calc_unit(ion, "UT/a/1") + 2 * calc_unit(ion, "UT/b/1"))
    array_j2 = array_l2 + 2 * array_ls + array_s2
    values, vectors = np.linalg.eigh(array_j2)
    sym = SymmetryList(values, "J2")
    assert sym.count() == {1: 4, 3: 24, 5: 42, 7: 56, 9: 70, 11: 60, 13: 42, 15: 48, 17: 18}
