##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

from .halfint import HalfInt
from .wigner import wigner3j
from .state import Coupling, StateListProduct, StateListSLJM, StateListSLJ, StateListJ, \
    StateProduct, StateSLJM, StateSLJ, StateJ
from .matrix import MATRIX, reduced_matrix
from .lanthanide import RADIAL, JUDD_OFELT, Lanthanide


def create_all():
    for num in (1, 13, 2, 12, 3, 11, 4, 10, 5, 9, 6, 8, 7):
        with Lanthanide(num) as ion:
            print(ion)
            ion.reduced()
