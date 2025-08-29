##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

from lanthanide import product_states, init_single, get_matrix, ORBITAL, SPIN, build_SLJM, SymmetryList, \
    SYM_CHAIN_SLJM, StateListSLJM

SYMS = {
    1: {
        "S2": {2: 14},
        "GR/7": {3: 14},
        "GG/2": {6: 14},
        "L2": {3: 14},
        "tau": {0: 14},
        "J2": {5: 6, 7: 8},
        "num": {0: 14},
        "Jz": {1: 2, 3: 2, 5: 2, 7: 1, -7: 1, -5: 2, -3: 2, -1: 2},
    },
    2: {
        "S2": {1: 28, 3: 63},
        "GR/7": {0: 1, 5: 63, 7: 27},
        "GG/2": {0: 1, 12: 42, 14: 27, 6: 21},
        "L2": {0: 1, 1: 9, 2: 5, 3: 21, 4: 9, 5: 33, 6: 13},
        "tau": {0: 91},
        "J2": {0: 2, 2: 3, 4: 15, 6: 7, 8: 27, 10: 11, 12: 26},
        "num": {0: 91},
        "Jz": {0: 13, 2: 11, 4: 10, 6: 7, 8: 6, 10: 3, 12: 2, -12: 2, -10: 3, -8: 6, -6: 7, -4: 10, -2: 11},
    },
    12: {
        "S2": {1: 28, 3: 63},
        "GR/7": {0: 1, 5: 63, 7: 27},
        "GG/2": {0: 1, 12: 42, 14: 27, 6: 21},
        "L2": {0: 1, 1: 9, 2: 5, 3: 21, 4: 9, 5: 33, 6: 13},
        "tau": {0: 91},
        "J2": {0: 2, 2: 3, 4: 15, 6: 7, 8: 27, 10: 11, 12: 26},
        "num": {0: 91},
        "Jz": {0: 13, 2: 11, 4: 10, 6: 7, 8: 6, 10: 3, 12: 2, -12: 2, -10: 3, -8: 6, -6: 7, -4: 10, -2: 11},
    },
    13: {
        "S2": {2: 14},
        "GR/7": {3: 14},
        "GG/2": {6: 14},
        "L2": {3: 14},
        "tau": {0: 14},
        "J2": {5: 6, 7: 8},
        "num": {0: 14},
        "Jz": {1: 2, 3: 2, 5: 2, 7: 1, -7: 1, -5: 2, -3: 2, -1: 2},
    },
}


class DummyLanthanide:
    def __init__(self, num: int):
        self.num = num
        self.l = ORBITAL
        self.s = SPIN
        self.product_states = states = product_states(num)
        self.single = init_single(None, None, states)

    def matrix(self, name):
        return get_matrix(self, name)


def run_state(num, num_sljm, num_slj):
    ion = DummyLanthanide(num)
    values, transform = build_SLJM(ion)
    for i, name in enumerate(SYM_CHAIN_SLJM):
        result = SymmetryList(values[:, i], name).count()
        # print(f'"{name}": {result},')
        assert result == SYMS[num][name]

    states_sljm = StateListSLJM(values, transform)
    assert len(states_sljm) == num_sljm
    assert type(states_sljm[1]).__name__ == "StateSLJM"

    states_slj = states_sljm.to_SLJ()
    assert len(states_slj) == num_slj
    assert type(states_slj[1]).__name__ == "StateSLJ"


def test_state_Ce():
    run_state(1, 14, 2)


def test_state_Pr():
    run_state(2, 91, 13)


def test_state_Tm():
    run_state(12, 91, 13)


def test_state_Yb():
    run_state(13, 14, 2)
