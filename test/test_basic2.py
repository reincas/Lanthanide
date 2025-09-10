##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# Compare calculated energy levels in Carnalls papers from 1969 and 1970
# on Pr3+:LaF3 and Tm3+:LaF3 with results from the Lanthanide package.
#
# Several typos in paper [25] had to be corrected.
#
##########################################################################

import pytest

from test_basic import run_basic
from data_basic2 import SOURCES, RADIAL


@pytest.mark.parametrize("name", [key for key in RADIAL if "blocking" not in RADIAL[key]])
def test_basic2(name):
    # Select data set
    assert name in RADIAL
    data = RADIAL[name]

    # Test source link
    assert "source" in data
    assert data["source"] in SOURCES

    # Run test
    run_basic(data)
