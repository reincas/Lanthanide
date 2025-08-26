##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import math
import numpy as np

from .halfint import HalfInt
from .wigner import wigner3j

# Version of the elementary unit tensor algorithms. If the precomputed unit tensor matrices in the file cache come
# with another version number, they will be recomputed. This will also render all other elements following in the
# chain of dependent cache elements invalid. These elements are the modules 'states' and 'matrix', see the init
# function of the Lanthanide class.
UNIT_VERSION = 1

# Orbital angular momentum quantum number of electrons in the f shell
ORBITAL = 3

# Spin quantum number of an electron
SPIN = HalfInt(1)

# Magnetic quantum numbers ml and ms of electrons in the f shell in standard order
MAGNETIC = [(ml, ms) for ml in range(ORBITAL, -ORBITAL - 1, -1) for ms in (SPIN, -SPIN)]

# Number of different electrons in 14 for the f configurations of lanthanide ions
LEN_SHELL = len(MAGNETIC)


def product_states(num):
    """ The electron product states in a configuration with num electrons is given by all num-combinations of the
    available electrons. We could thus use itertools.combinations(range(LEN_SHELL), num) to get the list of states.
    However, the Lanthanide package uses its own algorithm for the calculation of these combinations to ensure a
    defined and fixed order of the states. This keeps consistency within the file cache. """

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
    """ This class provides the evaluation of the q component u(k)_q of the elementary one-electron unit tensor
    operator of rank k in the orbital angular momentum space for the final electron a and initial electron b.
    Note: For q != 0 this operator is *not* symmetric with respect to the exchange of the two electrons. """

    def __init__(self, k: int, q: int):
        # assert isinstance(k, int)
        # assert isinstance(q, int)
        # assert 0 <= k <= 2 * ORBITAL
        # assert -k <= q <= k

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
    """ This class provides the evaluation of the q component t(k)_q of the elementary one-electron unit tensor
    operator of rank k in the spin angular momentum space for the final electron a and initial electron b.
    Note: For q != 0 this operator is *not* symmetric with respect to the exchange of the two electrons. """

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
    """ This class provides the evaluation of the scalar product u(k)·u(k) of the elementary one-electron unit tensor
    operator of rank k in the orbital angular momentum space for the final electron a and initial electron b. This
    scalar one-electron operator is symmetric with respect to the exchange of the two electrons. """

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
    """ This class provides the evaluation of the scalar product t(k)·t(k) of the elementary one-electron unit tensor
    operator of rank k in the spin angular momentum space for the final electron a and initial electron b. This
    scalar one-electron operator is symmetric with respect to the exchange of the two electrons. """

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
    """ This class provides the evaluation of the scalar product u(k)·t(k) of the elementary one-electron unit tensor
    operators of rank k in the orbital (u) and the spin (t) angular momentum space for the final electron a and
    initial electron b. This scalar one-electron operator is symmetric with respect to the exchange of the two
    electrons. """

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
    """ This class provides the evaluation of the scalar product u1(k)·u2(k) of the elementary one-electron unit
    tensor operators u1(k) and u2(k) of rank k in the orbital angular momentum space for the final electrons a, b
    and initial electrons c, d. This scalar two-electron operator is symmetric with respect to the exchange of the
    two electron pairs a, b and c, d. """

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
    """ This class provides the evaluation of the scalar product t1(k)·t2(k) of the elementary one-electron unit
    tensor operators t1(k) and t2(k) of rank k in the spin angular momentum space for the final electrons a, b
    and initial electrons c, d. This scalar two-electron operator is symmetric with respect to the exchange of the
    two electron pairs a, b and c, d. """

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
    """ This class provides the evaluation of the scalar product u1(k)·t2(k) of the elementary one-electron unit
    tensor operators u1(k) and t2(k) of rank k in the orbital (u1) and the spin (t2) angular momentum space for the
    final electrons a, b and initial electrons c, d. This scalar two-electron operator is symmetric with respect to
    the exchange of the two electron pairs a, b and c, d. """

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
    """ This class provides the evaluation of the scalar product {u1(ku1) x u2(ku2)}(k)·{t1(kt1) x t2(kt2)}(k) of the
    elementary one-electron mixed unit tensor operators {u1(ku1) x u2(ku2)}(k) and {t1(kt1) x t2(kt2)}(k) of rank k
    in the orbital (u1u2) and the spin (t1t2) angular momentum space for the final electrons a, b and initial
    electrons c, d. This scalar two-electron operator is symmetric with respect to the exchange of the two electron
    pairs a, b and c, d. """

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
    """ This class provides the evaluation of the triple scalar product u1(k1)·u2(k2)·u2(k3) of the elementary
    one-electron unit tensor operators u1(k1), u2(k2), and u3(k3) of rank k1, k2, and k3 in the orbital angular
    momentum space for the final electrons a, b, c and initial electrons d, e, f. This scalar three-electron
    operator is symmetric with respect to the exchange of the two electron triples a, b, c and d, e, f. """

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

def matrix_element(ion, operator, keys, cache):
    """ Calculate the matrix element of a unit tensor operator using the given elementary tensor operator object and
    the list of binary bra-ket keys (see module single) for which the elementary operator must be evaluated.
    The values of elementary tensor operators are cached. """

    # Calculate sum of elementary tensor operator values with parity
    value = 0.0
    for key, parity in keys:

        # Take the value of the elementary tensor operator from the cache
        if key in cache:
            single_value = cache[key]

        # Calculate the value of the elementary tensor operator and store in the cache
        else:
            initial, final = ion.single.index_pair(key, operator.order)
            quant = sum([MAGNETIC[i] for i in final + initial], (ORBITAL, SPIN))
            single_value = operator.element(*quant)
            cache[key] = single_value

        # Add to sum with parity
        if parity:
            value -= single_value
        else:
            value += single_value

    # Return value of the matrix element
    return value


def matrix_elements(ion, operator, cache):
    """ Generate all non-zero matrix elements of a unit tensor operator using the given elementary tensor
    operator object. """

    # Matrix element must be divided by the factorial of the number of electrons, the operator is acting on
    div = math.factorial(operator.order)

    # Calculate matrix elements based on the list of potentially non-zero matrix elements (see module single)
    for initial_index, final_index, key_slice in ion.single.elements(operator.order):

        # Calculate lower triangle matrix element and yield if non-zero
        keys = ion.single.lower_keys(key_slice, operator.order)
        value = matrix_element(ion, operator, keys, cache) / div
        if value:
            yield initial_index, final_index, value

        # Calculate only lower triangle matrix of symmetric tensor operators. Diagonal elements also require no
        # additional calculation
        if operator.symmetric or initial_index == final_index:
            continue

        # Calculate upper triangle matrix element and yield if non-zero
        keys = ion.single.upper_keys(key_slice, operator.order)
        value = matrix_element(ion, operator, keys, cache) / div
        if value:
            yield final_index, initial_index, value


# This dictionary is used to determine the respective elementary tensor operator class required for the calculation
# of a unit tensor operator with given name. The final letter in the name indicates if the operator should act on
# one ("a"), two ("b"), or three ("c") electrons.
OPERATORS = {
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


def unit_matrix(ion, operator_name):
    operator, num, params = operator_name.split("/")
    operator = OPERATORS[operator + num]
    params = map(int, params.split(","))
    operator = operator(*params)

    N = len(ion.product_states)
    matrix = np.zeros((N, N), dtype=float)
    cache = {}
    if ion.num >= operator.order:
        for i, f, value in matrix_elements(ion, operator, cache):
            matrix[f, i] = value

    if operator.symmetric:
        lower_tri_indices = np.tril_indices_from(matrix, k=-1)
        upper_tri_indices = (lower_tri_indices[1], lower_tri_indices[0])
        matrix[upper_tri_indices] = matrix[lower_tri_indices]

    return matrix


##########################################################################
# HDF5 cache interface
##########################################################################

def get_unit(ion, name):
    if name not in ion.unit_vault:
        print(f"Create unit matrix {name} ... ", end="")
        matrix = unit_matrix(ion, name)
        ion.unit_vault.create_dataset(name, data=matrix, compression="gzip", compression_opts=9)
        ion.vault.flush()
        print("done.")
    else:
        pass
        # print(f"Unit matrix {name}: load")
    return np.array(ion.unit_vault[name])


def init_unit(vault, group_name: str):
    if group_name in vault:
        if not vault.attrs["valid"] or "version" not in vault[group_name].attrs or vault[group_name].attrs[
            "version"] != UNIT_VERSION:
            del vault[group_name]
        vault.flush()

    if group_name not in vault:
        vault.attrs["valid"] = False
        vault.create_group(group_name)
        vault[group_name].attrs["version"] = UNIT_VERSION
        vault.flush()

    return vault[group_name]
