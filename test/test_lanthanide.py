##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import numpy as np

from lanthanide import Lanthanide, RADIAL, Coupling, StateListJ, StateJ

F02_ENERGIES = [205.0, 2305.84812, 4482.86283, 5113.96719, 6507.57988, 6935.88967, 9949.42237,
                17040.04537, 20881.93048, 21514.06056, 21516.9519, 22660.35176, 46901.03677]

with Lanthanide(2) as ion:
    assert str(ion) == "Pr3+ (4f2)"
    assert list(np.round(ion.energies, 5)) == F02_ENERGIES
    assert isinstance(ion.coupling, Coupling)
    assert len(ion.intermediate.states) == 13
    assert isinstance(ion.intermediate, StateListJ)
    assert all(isinstance(state, StateJ) for state in ion.intermediate.states)
    state = ion.intermediate.states[7]
    assert round(state.energy, 5) == 17040.04537
    assert list(np.round(state.weights, 5)) ==[0.89133, 0.02248, 0.08618]

    #print(", ".join(map(str, np.round(state.weights, 5))))
    print(state.weights)

