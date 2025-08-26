##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import math
import numpy as np

from .halfint import HalfInt
from .wigner import wigner3j

UNIT_VERSION = 1
ORBITAL = 3
SPIN = HalfInt(1)
MAGNETIC = [(ml, ms) for ml in range(ORBITAL, -ORBITAL - 1, -1) for ms in (SPIN, -SPIN)]
LEN_SHELL = len(MAGNETIC)


def product_states(num):
    """ Use own combinations algorithm instead of itertools.combinations to a defined fixed order. """

    def build(start, current_state):
        if len(current_state) == num:
            states.append(current_state)
        else:
            for i in range(start, LEN_SHELL):
                build(i + 1, current_state + (i,))

    states = []
    build(0, ())
    return states


##########################################################################
# One-electron elementary unit tensor operators
##########################################################################

class UnitUa():

    def __init__(self, k: int, q: int):
        assert isinstance(k, int)
        assert isinstance(q, int)
        assert 0 <= k <= 2 * ORBITAL
        assert -k <= q <= k
        self.order = 1
        self.symmetric = False
        self.k = k
        self.q = q
        self.name = f"U/a/{k},{q}"

    def element(self, l, s, mla, msa, mlb, msb):
        if msa != msb or self.q != mla - mlb:
            return 0.0

        result = wigner3j(l, self.k, l, -mla, self.q, mlb)

        if (l - mla) % 2:
            result = -result
        return result


class UnitTa():

    def __init__(self, k: int, q: int):
        assert isinstance(k, int)
        assert isinstance(q, int)
        assert 0 <= k <= 1
        assert -k <= q <= k
        self.order = 1
        self.symmetric = False
        self.k = k
        self.q = q
        self.name = f"T/a/{k},{q}"

    def element(self, l, s, mla, msa, mlb, msb):
        if mla != mlb or self.q != msa - msb:
            return 0.0

        result = wigner3j(s, self.k, s, -msa, self.q, msb)

        if (s - msa) % 2:
            result = -result
        return result


class UnitUUa():

    def __init__(self, k: int):
        assert isinstance(k, int)
        assert 0 <= k <= 2 * ORBITAL
        self.order = 1
        self.symmetric = True
        self.k = k
        self.name = f"UU/a/{k}"

    def element(self, l, s, mla, msa, mlb, msb):
        if msa != msb or mla != mlb:
            return 0.0

        return 1 / (2 * l + 1)


class UnitTTa():

    def __init__(self, k: int):
        assert isinstance(k, int)
        assert 0 <= k <= 1
        self.order = 1
        self.symmetric = True
        self.k = k
        self.name = f"TT/a/{k}"

    def element(self, l, s, mla, msa, mlb, msb):
        if msa != msb or mla != mlb:
            return 0.0

        return 0.5


class UnitUTa():

    def __init__(self, k: int):
        assert isinstance(k, int)
        assert 0 <= k <= 1
        self.order = 1
        self.symmetric = True
        self.k = k
        self.name = f"UT/a/{k}"

    def element(self, l, s, mla, msa, mlb, msb):
        if mla + msa != mlb + msb:
            return 0.0

        result = wigner3j(l, self.k, l, -mla, mla - mlb, mlb)
        result *= wigner3j(s, self.k, s, -msa, int(msa - msb), msb)

        if (l + s - mlb - msa) % 2:
            result = -result
        return result


##########################################################################
# Two-electron elementary unit tensor operators
##########################################################################

class UnitUUb():

    def __init__(self, k: int):
        assert isinstance(k, int)
        assert 0 <= k <= 2 * ORBITAL
        self.order = 2
        self.symmetric = True
        self.k = k
        self.name = f"UU/b/{k}"

    def element(self, l, s, mla, msa, mlb, msb, mlc, msc, mld, msd):
        if msa != msc or msb != msd or mla + mlb != mlc + mld:
            return 0.0

        result = wigner3j(l, self.k, l, -mla, mla - mlc, mlc)
        result *= wigner3j(l, self.k, l, -mlb, mlb - mld, mld)

        if (2 * l - mlb - mlc) % 2:
            result = -result
        return result


class UnitTTb():

    def __init__(self, k: int):
        assert isinstance(k, int)
        assert 0 <= k <= 1
        self.order = 2
        self.symmetric = True
        self.k = k
        self.name = f"TT/b/{k}"

    def element(self, l, s, mla, msa, mlb, msb, mlc, msc, mld, msd):
        if mla != mlc or mlb != mld or msa + msb != msc + msd:
            return 0.0

        result = wigner3j(s, self.k, s, -msa, msa - msc, msc)
        result *= wigner3j(s, self.k, s, -msb, msb - msd, msd)

        if (2 * s - msb - msc) % 2:
            result = -result
        return result


class UnitUTb():

    def __init__(self, k: int):
        assert isinstance(k, int)
        assert 0 <= k <= 1
        self.order = 2
        self.symmetric = True
        self.k = k
        self.name = f"UT/b/{k}"

    def element(self, l, s, mla, msa, mlb, msb, mlc, msc, mld, msd):
        if msa != msc or mlb != mld or mla + msb != mlc + msd:
            return 0.0

        result = wigner3j(l, self.k, l, -mla, mla - mlc, mlc)
        result *= wigner3j(s, self.k, s, -msb, msb - msd, msd)

        if (l + s - msb - mlc) % 2:
            result = -result
        return result


class UnitUUTTb():

    def __init__(self, ku1: int, ku2: int, kt1: int, kt2: int, k: int):
        assert isinstance(k, int)
        assert isinstance(ku1, int)
        assert isinstance(ku2, int)
        assert isinstance(kt1, int)
        assert isinstance(kt2, int)
        assert abs(ku1 - ku2) <= k <= ku1 + ku2
        assert abs(kt1 - kt2) <= k <= kt1 + kt2
        assert 0 <= ku1 <= 2 * ORBITAL
        assert 0 <= ku2 <= 2 * ORBITAL
        assert 0 <= kt1 <= 1
        assert 0 <= kt2 <= 1
        self.order = 2
        self.symmetric = True
        self.k = k
        self.ku1 = ku1
        self.ku2 = ku2
        self.kt1 = kt1
        self.kt2 = kt2
        self.name = f"UUTT/b/{ku1},{ku2},{kt1},{kt2},{k}"

    def element(self, l, s, mla, msa, mlb, msb, mlc, msc, mld, msd):
        if mla + mlb + msa + msb != mlc + mld + msc + msd:
            return 0.0

        result = 2 * self.k + 1
        result *= wigner3j(self.ku1, self.k, self.ku2, mla - mlc, mld + mlc - mlb - mla, mlb - mld)
        result *= wigner3j(self.kt1, self.k, self.kt2, msa - msc, msd + msc - msb - msa, msb - msd)
        result *= wigner3j(l, self.ku1, l, -mla, mla - mlc, mlc)
        result *= wigner3j(l, self.ku2, l, -mlb, mlb - mld, mld)
        result *= wigner3j(s, self.kt1, s, -msa, msa - msc, msc)
        result *= wigner3j(s, self.kt2, s, -msb, msb - msd, msd)

        if (2 * l + 2 * s - msa - msb - mlc - mld) % 2:
            result = -result
        return result


##########################################################################
# Three-electron elementary unit tensor operators
##########################################################################

class UnitUUUc():

    def __init__(self, k1: int, k2: int, k3: int):
        assert isinstance(k1, int)
        assert isinstance(k2, int)
        assert isinstance(k3, int)
        assert 0 <= k1 <= 2 * ORBITAL
        assert 0 <= k2 <= 2 * ORBITAL
        assert 0 <= k3 <= 2 * ORBITAL
        self.order = 3
        self.symmetric = True
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.name = f"UUU/c/{k1},{k2},{k3}"

    def element(self, l, s, mla, msa, mlb, msb, mlc, msc, mld, msd, mle, mse, mlf, msf):
        if msa != msd or msb != mse or msc != msf:
            return 0.0

        result = wigner3j(self.k1, self.k2, self.k3, mla - mld, mlb - mle, mlc - mlf)
        result *= wigner3j(l, self.k1, l, -mla, mla - mld, mld)
        result *= wigner3j(l, self.k2, l, -mlb, mlb - mle, mle)
        result *= wigner3j(l, self.k3, l, -mlc, mlc - mlf, mlf)

        if (3 * l - mla - mlb - mlc) % 2:
            result = -result
        return result


##########################################################################
# Matrix elements of unit tensor operators in product space
##########################################################################

UNITS = {
    "Ua": UnitUa,
    "Ta": UnitTa,
    "UUa": UnitUUa,
    "TTa": UnitTTa,
    "UTa": UnitUTa,
    "UUb": UnitUUb,
    "TTb": UnitTTb,
    "UTb": UnitUTb,
    "UUTTb": UnitUUTTb,
    "UUUc": UnitUUUc,
}


def matrix_element(ion, unit, single, stats, keys):
    value = 0.0
    for key, parity in keys:
        # initial, final = ion.single.index_pair(key, unit.order)
        # quant = sum([MAGNETIC[i] for i in final + initial], (ORBITAL, SPIN))
        # single_value = unit.element(*quant)
        # stats["single"] += 1
        # if single_value:
        #     stats["nz_single"] += 1

        if key in single:
            single_value = single[key]
        else:
            initial, final = ion.single.index_pair(key, unit.order)
            quant = sum([MAGNETIC[i] for i in final + initial], (ORBITAL, SPIN))
            single_value = unit.element(*quant)
            single[key] = single_value
            stats["single"] += 1
            if single_value:
                stats["nz_single"] += 1

        if parity:
            value -= single_value
        else:
            value += single_value
    return value


def matrix_elements(ion, unit, single, stats):
    div = math.factorial(unit.order)
    stats["elements"] = 0
    stats["nz_elements"] = 0
    stats["single"] = 0
    stats["nz_single"] = 0
    for initial_index, final_index, key_slice in ion.single.elements(unit.order):
        keys = ion.single.lower_keys(key_slice, unit.order)
        value = matrix_element(ion, unit, single, stats, keys) / div
        stats["elements"] += 1
        if value:
            stats["nz_elements"] += 1
            yield initial_index, final_index, value
        if unit.symmetric or initial_index == final_index:
            continue
        keys = ion.single.upper_keys(key_slice, unit.order)
        value = matrix_element(ion, unit, single, stats, keys) / div
        stats["elements"] += 1
        if value:
            stats["nz_elements"] += 1
            yield final_index, initial_index, value


def unit_matrix(ion, unit_name):
    unit, order, params = unit_name.split("/")
    unit = UNITS[unit + order]
    params = map(int, params.split(","))
    unit = unit(*params)

    N = len(ion.product_states)
    matrix = np.zeros((N, N), dtype=float)
    single = {}
    stats = {}
    if ion.num >= unit.order:
        for i, f, value in matrix_elements(ion, unit, single, stats):
            matrix[f, i] = value

    if unit.symmetric:
        lower_tri_indices = np.tril_indices_from(matrix, k=-1)
        upper_tri_indices = (lower_tri_indices[1], lower_tri_indices[0])
        matrix[upper_tri_indices] = matrix[lower_tri_indices]

    stats["name"] = unit.name
    stats["unique"] = len(single)
    return matrix, stats


##########################################################################
# HDF5 cache interface
##########################################################################

def get_unit(ion, name):
    if name not in ion.unit_vault:
        print(f"Create unit matrix {name} ... ", end="")
        matrix = unit_matrix(ion, name)[0]
        ion.unit_vault.create_dataset(name, data=matrix, compression="gzip", compression_opts=9)
        ion.vault.flush()
        print("done.")
    else:
        pass
        #print(f"Unit matrix {name}: load")
    return np.array(ion.unit_vault[name])


def init_unit(vault, group_name: str):
    if group_name in vault:
        if not vault.attrs["valid"] or "version" not in vault[group_name].attrs or vault[group_name].attrs["version"] != UNIT_VERSION:
            del vault[group_name]
        vault.flush()

    if group_name not in vault:
        vault.attrs["valid"] = False
        vault.create_group(group_name)
        vault[group_name].attrs["version"] = UNIT_VERSION
        vault.flush()

    return vault[group_name]
