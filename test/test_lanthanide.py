##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import numpy as np

from lanthanide import Lanthanide, RADIAL, Coupling, StateListJ, StateJ

F02_ENERGIES = [327.39, 2363.85298, 4498.00172, 5106.88044, 6463.20358, 6954.38691, 9882.96568,
                17022.73705, 20856.47111, 21471.52564, 21510.80436, 22646.25844, 46461.4324]
F02_REDUCED = [0.20208, 0.03274, 0.40418, 0.34733, 0.05105, 0.00609,
               0.0162, 0.17265, 0.17099, 0.0499, 0.03641, 0.00663]

with Lanthanide(2) as ion:
    ion.set_radial(RADIAL["Pr3+/ZBLAN"])
    reduced = ion.reduced().U4[1:, 0]
    assert list(np.round(reduced, 5)) == F02_REDUCED
    assert str(ion) == "Pr3+ (4f2)"
    assert list(np.round(ion.energies, 5)) == F02_ENERGIES
    assert isinstance(ion.coupling, Coupling)
    assert len(ion.intermediate.states) == 13
    assert isinstance(ion.intermediate, StateListJ)
    assert all(isinstance(state, StateJ) for state in ion.intermediate.states)
    state = ion.intermediate.states[7]
    assert round(state.energy, 5) == 17022.73705
    assert list(np.round(state.weights, 5)) ==[0.8986, 0.02108, 0.08032]

    judd_ofelt = {"JO/2": 1.981, "JO/4": 4.645, "JO/6": 6.972}
    strength = ion.line_strengths(judd_ofelt)
    sed = strength.Sed[1:, 0]
    smd = strength.Smd[1:, 0]

    print(np.min(strength.Sed), np.max(strength.Sed))
    print(np.min(strength.Smd), np.max(strength.Smd))
    print(", ".join(map(str, np.round(sed, 5))))
    print(", ".join(map(str, np.round(smd, 5))))

