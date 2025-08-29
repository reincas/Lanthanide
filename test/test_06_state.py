##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import math
import numpy as np

from lanthanide import product_states, init_single, get_matrix, ORBITAL, SPIN, build_SLJM


class DummyLanthanide:
    def __init__(self, num: int):
        self.num = num
        self.l = ORBITAL
        self.s = SPIN
        self.product_states = states = product_states(num)
        self.single = init_single(None, None, states)

    def matrix(self, name):
        return get_matrix(self, name)

def test_state():

    ion = DummyLanthanide(2)
    values, transform = build_SLJM(ion)