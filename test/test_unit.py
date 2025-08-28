import pytest

from lanthanide import product_states, single_elements, SingleElements, calc_unit


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


def get_nz(array):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if abs(array[i, j]) < 1e-7:
                continue
            yield i, j, float(array[i, j])


def test_unit():
    ion = DummyLanthanide(2)

    array = calc_unit(ion, "UU/b/2")
    assert len(list(get_nz(array))) == 231
    assert pytest.approx(array[1, 1], rel=1e-12) == -0.05952380952380953
    assert pytest.approx(array[13, 2], rel=1e-12) == -0.05952380952380953
    assert pytest.approx(array[17, 36], rel=1e-12) == 0.016835875742536845
    assert pytest.approx(array[47, 9], rel=1e-12) == 0.03367175148507368
    assert pytest.approx(array[48, 10], rel=1e-12) == 0.03367175148507368
    assert pytest.approx(array[90, 90], rel=1e-12) == 0.05952380952380953

    array = calc_unit(ion, "UUTT/b/4,6,1,1,2")
    assert len(list(get_nz(array))) == 716
    assert pytest.approx(array[2, 3], rel=1e-12) == -0.0003505766263598649
    assert pytest.approx(array[6, 17], rel=1e-12) == 0.0010793363150885641
    assert pytest.approx(array[8, 38], rel=1e-12) == 4.141428056635748e-05
    assert pytest.approx(array[40, 31], rel=1e-12) == 7.11190595213394e-05
    assert pytest.approx(array[56, 64], rel=1e-12) == -0.0027053132445372286
    assert pytest.approx(array[89, 89], rel=1e-12) == -0.0003695402112383326

    #print(len(list(get_nz(array))))
    #for line in get_nz(array):
    #    print(line)
