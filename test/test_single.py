import math
import pytest

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

import sys

@pytest.mark.parametrize("num", [1, 2])
def test_single(num: int):
    states = product_states(num)
    assert len(states) == math.comb(LEN_SHELL, num)

    result = single_elements(states)
    assert len(result) == 3
    assert all(len(sub) == 2 for sub in result)
    sys.stdout.flush()
    return

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
