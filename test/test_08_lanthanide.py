##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import pytest
import numpy as np

from lanthanide import Lanthanide, RADIAL, Coupling, StateListJ, StateJ

F02_ENERGIES = [327.39, 2363.85298, 4498.00172, 5106.88044, 6463.20358, 6954.38691, 9882.96568,
                17022.73705, 20856.47111, 21471.52564, 21510.80436, 22646.25844, 46461.4324]
F02_REDUCED = [0.20207968, 0.03274454, 0.40418374, 0.34733364, 0.05105489, 0.00608966,
               0.01620277, 0.17264825, 0.17099050, 0.04989722, 0.03641452, 0.00662584]
F02_SED = [0.46280620, 0.09617662, 0.31717911, 0.56532627, 0.31428973, 0.01498091,
           0.03703203, 0.06852648, 0.06786849, 0.03506035, 0.09560437, 0.00262989]
F02_SMD = [3.36312633e-02, 0.0, 0.0, 1.90612714e-05, 4.89040561e-04, 1.97265651e-04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

def test_lanthanide():
    with Lanthanide(2) as ion:
        ion.set_radial(RADIAL["Pr3+/ZBLAN"])
        reduced = ion.reduced().U4[1:, 0]
        assert pytest.approx(reduced, rel=1e-6) == F02_REDUCED
        assert str(ion) == "Pr3+ (4f2)"
        assert isinstance(ion.coupling, Coupling)

        assert pytest.approx(ion.energies, rel=1e-7) == F02_ENERGIES
        assert isinstance(ion.intermediate, StateListJ)
        assert len(ion.intermediate.states) == 13
        assert ion.num_states(Coupling.Product) == 91
        assert ion.num_states(Coupling.SLJM) == 91
        assert ion.num_states(Coupling.SLJ) == 13
        assert ion.num_states(Coupling.J) == 13

        assert all(isinstance(state, StateJ) for state in ion.intermediate.states)
        state = ion.intermediate.states[7]
        assert pytest.approx(state.energy, rel=1e-7) == 17022.73705
        assert list(np.round(state.weights, 5)) ==[0.8986, 0.02108, 0.08032]

        judd_ofelt = {"JO/2": 1.981, "JO/4": 4.645, "JO/6": 6.972}
        strength = ion.line_strengths(judd_ofelt)
        sed = strength.Sed[1:, 0] * 1e52
        smd = strength.Smd[1:, 0] * 1e52
        assert pytest.approx(sed, abs=1e-6) == F02_SED
        assert pytest.approx(smd, abs=1e-9) == F02_SMD

