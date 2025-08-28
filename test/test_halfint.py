##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

from lanthanide import HalfInt

def test_halfint():
    assert -HalfInt(1) == HalfInt(-1)
    assert HalfInt(2) + HalfInt(3) == HalfInt(2 + 3)
    assert HalfInt(1) + HalfInt(3) == 2
    assert HalfInt(1) - HalfInt(3) == -1
    assert HalfInt(1) + 2 == HalfInt(5)
    assert 2 - HalfInt(3) == HalfInt(1)
    assert 2 * HalfInt(3) == 3
    assert HalfInt(1) * 3 == HalfInt(3)
    assert HalfInt(1) < HalfInt(3)
    assert HalfInt(1) <= 2
    assert -1 <= HalfInt(3)
    assert HalfInt(1) != HalfInt(3)
    assert HalfInt(1) * 3 == HalfInt(3)
    assert str(-HalfInt(5) + 1) == "-3/2"
