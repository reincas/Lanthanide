##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import math
import pytest

from lanthanide import HalfInt, wigner3j, wigner6j

SPIN = HalfInt(1)


def test_wigner3j():
    assert pytest.approx(wigner3j(3, 2, 3, 0, 0, 0), rel=1e-12) == 2 / math.sqrt(105)
    assert pytest.approx(wigner3j(3, 2, 3, -2, -1, 3), rel=1e-12) == -math.sqrt(5) / (2 * math.sqrt(21))
    assert pytest.approx(wigner3j(1, 1, 1, 0, 0, 0), rel=1e-12) == 0.0
    assert pytest.approx(wigner3j(1, 1, 1, 1, -1, 0), rel=1e-12) == 1 / math.sqrt(6)
    assert pytest.approx(wigner3j(2, 1, 1, 1, 0, -1), rel=1e-12) == -1 / math.sqrt(10)
    assert pytest.approx(wigner3j(2, 3 * SPIN, 3 * SPIN, 0, -SPIN, SPIN), rel=1e-12) == -1 / (2 * math.sqrt(5))
    assert pytest.approx(wigner3j(SPIN, SPIN, 1, -SPIN, SPIN, 0), rel=1e-12) == 1 / math.sqrt(6)
    assert pytest.approx(wigner3j(2, 2, 2, 0, 0, 0), rel=1e-12) == -math.sqrt(2 / 35)


def test_wigner6j():
    assert pytest.approx(wigner6j(0, 0, 0, 0, 0, 0), rel=1e-12) == 1.0
    assert pytest.approx(wigner6j(1, 2, 3, 2, 1, 2), rel=1e-12) == 1 / (5 * math.sqrt(21))
    assert pytest.approx(wigner6j(1, 1, 1, 1, 1, 1), rel=1e-12) == 1 / 6
    assert pytest.approx(wigner6j(1, 1, 1, 2, 2, 5), rel=1e-12) == 0.0
    assert pytest.approx(wigner6j(0, 2, 2, 3, 3, 3), rel=1e-12) == 1 / math.sqrt(35)
    assert pytest.approx(wigner6j(SPIN, 1, SPIN, SPIN, 1, SPIN), rel=1e-12) == 1 / 6
    assert pytest.approx(wigner6j(1, SPIN, 3 * SPIN, 1, 3 * SPIN, SPIN), rel=1e-12) == math.sqrt(5) / (6 * math.sqrt(2))
    assert pytest.approx(wigner6j(3 * SPIN, 1, SPIN, 3 * SPIN, 1, SPIN), rel=1e-12) == -1 / 12
