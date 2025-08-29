##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import math

from lanthanide import LEN_SHELL, product_states, single_elements, SingleElements

SIZES = {
    1: [105, 105, 0, 0, 0, 0],
    2: [1183, 1274, 4186, 16744, 0, 0],
    3: [6370, 7098, 36400, 172536, 66430, 2391480],
    4: [21021, 24024, 156156, 804804, 396396, 20684664],
    5: [47047, 55055, 407407, 2242240, 1248247, 79639560],
    6: [75075, 90090, 705705, 4144140, 2387385, 179459280],
    7: [87516, 108108, 844272, 5333328, 2946372, 261621360],
    8: [75075, 96096, 705705, 4876872, 2387385, 257297040],
    9: [47047, 63063, 407407, 3171168, 1248247, 172540368],
    10: [21021, 30030, 156156, 1441440, 396396, 77837760],
    11: [6370, 10010, 36400, 440440, 66430, 22702680],
    12: [1183, 2184, 4186, 84084, 4186, 3963960],
    13: [105, 273, 105, 8736, 105, 360360],
}

def run_single(num: int):
    states = product_states(num)
    assert len(states) == math.comb(LEN_SHELL, num)

    result = single_elements(states)
    assert len(result) == 3
    assert all(len(sub) == 2 for sub in result)

    size = [len(x) for sub in result for x in sub]
    assert size == SIZES[num]

    one, two, three = result
    group = {
        "one": {"indices": one[0], "elements": one[1]},
        "two": {"indices": two[0], "elements": two[1]},
        "three": {"indices": three[0], "elements": three[1]},
    }
    single = SingleElements(group, states)
    assert len(single) == num
    assert single[1] == group["one"]
    if num >= 2:
        assert single[2] == group["two"]
    if num >= 3:
        assert single[3] == group["three"]

def test_single_Ce():
    run_single(1)

def test_single_Pr():
    run_single(2)

def test_single_Er():
    run_single(3)

def test_single_Yb():
    run_single(13)