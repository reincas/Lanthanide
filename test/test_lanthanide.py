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
F02_SED = [0.46281, 0.09618, 0.31718, 0.56533, 0.31429, 0.01498, 0.03703, 0.06853, 0.06787, 0.03506, 0.0956, 0.00263]
F02_SMD = [0.03363, 0.0, 0.0, 2e-05, 0.00049, 0.0002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

def test_lanthanide():
    with Lanthanide(2) as ion:
        ion.set_radial(RADIAL["Pr3+/ZBLAN"])
        reduced = ion.reduced().U4[1:, 0]
        assert list(np.round(reduced, 5)) == F02_REDUCED
        assert str(ion) == "Pr3+ (4f2)"
        assert isinstance(ion.coupling, Coupling)

        assert list(np.round(ion.energies, 5)) == F02_ENERGIES
        assert isinstance(ion.intermediate, StateListJ)
        assert len(ion.intermediate.states) == 13

        assert all(isinstance(state, StateJ) for state in ion.intermediate.states)
        state = ion.intermediate.states[7]
        assert round(state.energy, 5) == 17022.73705
        assert list(np.round(state.weights, 5)) ==[0.8986, 0.02108, 0.08032]

        judd_ofelt = {"JO/2": 1.981, "JO/4": 4.645, "JO/6": 6.972}
        strength = ion.line_strengths(judd_ofelt)
        sed = strength.Sed[1:, 0] * 1e52
        smd = strength.Smd[1:, 0] * 1e52
        assert list(np.round(sed, 5)) == F02_SED
        assert list(np.round(smd, 5)) == F02_SMD

        #print(", ".join(map(str, np.round(smd, 5))))

