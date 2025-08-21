##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import numpy as np

from lanthanide import Lanthanide, Coupling, RADIAL

if __name__ == "__main__":
    for num in range(1, 4):
        with Lanthanide(num) as ion:
            print(ion)
            ion.reduced()

