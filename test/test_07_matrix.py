##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import numpy as np
import pytest

from lanthanide import RADIAL, product_states, init_single, Matrix, get_matrix, Coupling, ORBITAL

EIGEN = np.array([2, 3, 5, 6, 9])
MATRIX = np.array([[-0.07344058, 0.31934316, 0.88450433, 0.22878318, 0.24070059],
                   [-0.45118128, -0.64774046, 0.03943607, -0.03655011, 0.61153629],
                   [-0.18617929, -0.00317505, -0.24141063, 0.94993278, -0.06837989],
                   [0.37062937, 0.4274932, -0.33870052, 0.04202489, 0.75059929],
                   [0.78677384, -0.54377544, 0.2076048, 0.20538735, 0.00338833]])


class DummyMatrix:
    def __init__(self, array):
        self.array = array


class DummyLanthanide:
    def __init__(self, num: int):
        self.num = num
        self.l = ORBITAL
        self.product_states = states = product_states(num)
        self.single = init_single(None, None, states)

        def matrix(self, name, coupling=None):
            assert coupling is None or coupling == Coupling.Product
            return get_matrix(self, name)


def test_matrix():
    ion = DummyLanthanide(2)

    # name = "UT/b/2"
    # array = calc_unit(ion, name)
    # matrix = Matrix(ion, array, name)

    array = MATRIX @ np.diag(EIGEN) @ MATRIX.T
    matrix = Matrix(ion, array, "test")
    eigen, vectors = matrix.diagonalize()
    assert pytest.approx(eigen, abs=1e-6) == EIGEN
    assert pytest.approx(np.abs(vectors), abs=1e-6) == np.abs(MATRIX)

    array = ((5 * matrix - matrix) / 2).array
    matrix = Matrix(ion, array)
    eigen, vectors = matrix.diagonalize()
    assert pytest.approx(eigen, abs=1e-6) == 2 * EIGEN
    assert pytest.approx(np.abs(vectors), abs=1e-6) == np.abs(MATRIX)

    radial = RADIAL["Pr3+"]
    # H = build_hamilton(ion, radial, Coupling.SLJ)
