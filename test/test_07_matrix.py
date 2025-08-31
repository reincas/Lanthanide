##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import numpy as np
import pytest

from lanthanide import RADIAL, product_states, init_single, Matrix, get_matrix, ORBITAL, SPIN, Coupling, \
    build_hamilton, reduced_matrix, init_states, LANTHANIDES

EIGEN = np.array([2, 3, 5, 6, 9])
MATRIX = np.array([[-0.07344058, 0.31934316, 0.88450433, 0.22878318, 0.24070059],
                   [-0.45118128, -0.64774046, 0.03943607, -0.03655011, 0.61153629],
                   [-0.18617929, -0.00317505, -0.24141063, 0.94993278, -0.06837989],
                   [0.37062937, 0.4274932, -0.33870052, 0.04202489, 0.75059929],
                   [0.78677384, -0.54377544, 0.2076048, 0.20538735, 0.00338833]])
ENERGIES_F02 = [-11247.06110968, -9146.21298849, -6969.19827869, -6338.09392371, -4944.48122688,
                -4516.17143759, -1502.63873643, 5587.9842584, 9429.86936611, 10061.99944569,
                10064.89079064, 11208.29065387, 35448.97566233]
WEIGHT4 = [0.033471471373506444, 0.23274398860708642, 0.7107571845637194, 0.016418996863721762, 0.3182046719237578,
           0.6168446733433915, 0.2530001541532816, 0.06965943708332314, 0.0, 0.0, 0.013512482962126204,
           0.4963453057444576, 0.0]
WEIGHT6 = [0.11737443386221942, 0.6591978972155589, 0.303903979802215, 0.0, 0.0, 0.0909492368332769,
           0.005887991636483049, 0.0, 0.0, 0.0, 0.03237457133704324, 0.0, 0.0]
REDH12 = [0, 0, 0, -0.09256622271389522, 0, 0, 0, -0.04378764103673054, 0, 0, 0, 0.005384598920204938, 0]
REDH16 = [-0.0007494882534019258, 0, 0, 0, 0, -0.0658613894531618, 0.06990903249054214, 0, 0, 0, 0, 0, 0]
REDH2 = [0, 0, -4.882742931038221, 0, 0, 0, 0, 0, 0, 0, 0.5105255184602158, 0, 0]
REDH3 = [0, 0, 0, 0, 0, -9.878179134281773e-33, -2.2997047323199953e-33, 1.4404590304834598e-31,
         -2.5445423023613995e-18, -1.0147245635276235e-32, -5.315068581563589e-16, -7.420369617991846e-16,
         -5.729396997429248e-16, 2.2360679774997894, -2.7005322956408437e-16, 3.471125772152828e-17,
         7.919381992920967e-17, 3.6040284143277693e-17, 1.3233533810636782e-16, 3.6971448061602215e-16,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0]

class DummyMatrix:
    def __init__(self, array):
        self.array = array


class DummyLanthanide:
    def __init__(self, num: int):
        self.num = num
        self.l = ORBITAL
        self.s = SPIN
        self.product = product_states(num)
        self.single = init_single(None, None, self.product)
        self._state_dict_ = init_states(None, None, self)

    def matrix(self, name):
        return get_matrix(self, name)

    def states(self, coupling):
        return self._state_dict_[coupling.name]


def run_matrix(num):
    ion = DummyLanthanide(num)

    radial = RADIAL[f"{LANTHANIDES[num]}3+"]
    H = build_hamilton(ion, radial, Coupling.SLJ)
    energies, transform = H.fast_diagonalise()

    states = ion.states(Coupling.SLJ)
    intermediate = states.to_J(energies, transform)
    ion._state_dict_[Coupling.J.name] = intermediate

    return ion, energies, transform


def test_matrix():
    ion, energies, transform = run_matrix(2)

    array = MATRIX @ np.diag(EIGEN) @ MATRIX.T
    matrix = Matrix(ion, array, "test")
    eigen, vectors = matrix.diagonalise()
    assert pytest.approx(eigen, abs=1e-6) == EIGEN
    assert pytest.approx(np.abs(vectors), abs=1e-6) == np.abs(MATRIX)

    array = ((5 * matrix - matrix) / 2).array
    matrix = Matrix(ion, array)
    eigen, vectors = matrix.diagonalise()
    assert pytest.approx(eigen, abs=1e-6) == 2 * EIGEN
    assert pytest.approx(np.abs(vectors), abs=1e-6) == np.abs(MATRIX)


def test_hamilton():
    ion, energies, transform = run_matrix(2)
    assert pytest.approx(energies, abs=1e-5) == ENERGIES_F02


def test_reduced():
    ion, energies, transform = run_matrix(2)
    U4 = np.power(reduced_matrix(ion, "ED/4,{q}", Coupling.J), 2)
    assert pytest.approx(U4[:, 2], rel=1e-9) == WEIGHT4
    U6 = np.power(reduced_matrix(ion, "ED/6,{q}", Coupling.J), 2)
    assert pytest.approx(U6[:, 3], rel=1e-9) == WEIGHT6
    red = reduced_matrix(ion, "H1/2", Coupling.J)
    assert pytest.approx(red[3, :], rel=1e-9) == REDH12
    red = reduced_matrix(ion, "H1/6", Coupling.J)
    assert pytest.approx(red[5, :], rel=1e-9) == REDH16
    red = reduced_matrix(ion, "H2", Coupling.J)
    assert pytest.approx(red[10, :], rel=1e-9) == REDH2
    red = reduced_matrix(ion, "H3/2", Coupling.SLJM)
    assert pytest.approx(red[:, 13], rel=1e-9) == REDH3
    #print(", ".join(map(str, red[:, 13])))
