##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

SOURCES = {
    "[25]": {
        "authors": ["W. T. Carnall", "P. R. Fields", "J. Morrison", "R. Sarup"],
        "title": "Absorption Spectrum of Tm3+:LaF3",
        "journal": "Journal of Chemical Physics",
        "volume": "52",
        "pages": "4054-4059",
        "year": "1970",
        "doi": "https://doi.org/10.1063/1.1673608",
    },
    "[68]": {
        "authors": ["W. T. Carnall", "P. R. Fields", "R. Sarup"],
        "title": "1S Level of Pr3+ in Crystal Matrices and Energy‐Level Parameters for the 4f2 Configuration of Pr3+ in LaF3",
        "journal": "Journal of Chemical Physics",
        "volume": "51",
        "pages": "2587-2591",
        "year": "1969",
        "doi": "https://doi.org/10.1063/1.1672382",
        "remark": "[68] is ref 4 in [25]",
    },
}

RADIAL = {
    "Pr3+:LaF3": {
        "num": 2,
        "radial":
            {"E^1": 4559.0, "E^2": 21.954, "E^3": 467.75,
             "H2": 744.44, "H3/0": 15.294, "H3/1": -669.02, "H3/2": 1411.8},
        "J":
            [4, 5, 6, 2, 3, 4, 4, 2, 0, 1, 6, 2, 0],
        "energies":
            [232, 2317, 4502, 5190, 6586, 7045, 9992, 17047, 20905, 21534, 21514, 22748, 46986],
        "source": "[68]",
        "table": "II"
    },
    "Pr3+:LaF3/ext": {
        "num": 2,
        "radial":
            {"E^1": 4567.2, "E^2": 21.954, "E^3": 467.75,
             "H2": 721.90, "H3/0": 15.294, "H3/1": -669.02, "H3/2": 1413.7,
             "H5/0": 1.231, "H5/2": 0.690, "H5/4": 0.468, "P_2": 2.697, "P_4": 0.343, "P_6": 0.324},
        "J":
            [4, 5, 6, 2, 3, 4, 4, 2, 0, 1, 6, 2, 0],
        "energies/meas":
            [200, 2363, 4487, 5215, 6568, 7031, 10001, 17047, 20927, 21514, 21514, 22746, 46986],
        "energies/meas-calc":
            [-21, -54, -6.3, 24, -6.6, -7.3, -5.6, 13, 7.6, -3.6, -5.2, 4.4, 5.6],
        "correct":{
            ("energies/meas-calc", 1, -54, 54),
            ("energies/meas-calc", 7, 13, 4.4),  # 1D2 <-> 3P2
            ("energies/meas-calc", 11, 4.4, 13), # 1D2 <-> 3P2
        },
        "source": "[25]",
        "table": "II"
    },
    "Tm3+:LaF3": {
        "num": 12,
        "radial":
            {"E^1": 6737.5, "E^2": 33.643, "E^3": 681.22,
             "H2": 2633.0, "H3/0": 13.124, "H3/1": -743.02, "H3/2": 1992.2},
        "J":
            [6, 4, 5, 4, 3, 2, 4, 2, 6, 0, 1, 2],
        "energies":
            [119, 5835, 8320, 12701, 14586, 15150, 21366, 28098, 34866, 35604, 36562, 38315],
        "source": "[25]",
        "table": "I"
    },
    "Tm3+:LaF3/ext": {
        "num": 12,
        "radial":
            {"E^1": 6911.8, "E^2": 33.728, "E^3": 675.28,
             "H2": 2593.9, "H3/0": 9.475, "H3/1": -601.09, "H3/2": 1395.1,
             "H5/0": 5.002, "H5/2": 2.801, "H5/4": 1.901, "P_2": 3.915, "P_4": 0.090, "P_6": 0.070},
        "J":
            [6, 4, 5, 4, 3, 2, 4, 2, 6, 0, 1, 2],
        "energies/meas":
            [100, 5858, 8336, 12711, 14559, 15173, 21352, 28061, 34886, 35572, 36559, 38344],
        "energies/meas-calc":
            [-6.7, 9.2, 1.8, 4.4, -3.4, -2.1, -4.2, -0.8, 0.25, 0.16, -0.08, 1.4],
        "correct":{
            ("energies/meas", 8, 34886, 34866),
            ("energies/meas", 9, 35572, 35604),  # calc instead of meas (wrong column)
            ("energies/meas-calc", 1, 9.2, 4.4), # Classic Tm3+ problem 3F4 <-> 3H4
            ("energies/meas-calc", 3, 4.4, 9.2), # Classic Tm3+ problem 3F4 <-> 3H4
        },
        "source": "[25]",
        "table": "II"
    },
}