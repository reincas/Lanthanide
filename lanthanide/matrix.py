##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import math
import numpy as np

from .wigner import wigner3j
from .unit import get_unit
from .state import Coupling

MATRIX_VERSION = 1
JUDD_TABLE = [[
    (2, 2, 2, 1),
    -math.sqrt(11 / 1134), math.sqrt(605 / 5292), math.sqrt(32761 / 889056),
    math.sqrt(3575 / 889056), -math.sqrt(17303 / 396900), -math.sqrt(1573 / 8232),
    math.sqrt(264407 / 823200), math.sqrt(21879 / 274400), -math.sqrt(46189 / 231525),
], [
    (2, 2, 4, 3),
    math.sqrt(4 / 189), -math.sqrt(6760 / 43659), math.sqrt(33 / 1372),
    -math.sqrt(325 / 37044), math.sqrt(416 / 33075), -math.sqrt(15028 / 305613),
    math.sqrt(28717 / 2778300), -math.sqrt(37349 / 926100), -math.sqrt(8398 / 694575),
], [
    (2, 4, 4, 3),
    math.sqrt(1 / 847), -math.sqrt(1805 / 391314), -math.sqrt(4 / 33957),
    -math.sqrt(54925 / 373527), -math.sqrt(117 / 296450), math.sqrt(4693 / 12326391),
    -math.sqrt(1273597 / 28014525), math.sqrt(849524 / 9338175), -math.sqrt(134368 / 3112725),
], [
    (2, 4, 6, 6),
    math.sqrt(26 / 3267), -math.sqrt(4160 / 754677), -math.sqrt(13 / 264),
    math.sqrt(625 / 26136), math.sqrt(256 / 571725), math.sqrt(1568 / 107811),
    math.sqrt(841 / 1960200), -math.sqrt(17 / 653400), -math.sqrt(15827 / 245025),
], [
    (4, 4, 4, 1),
    -math.sqrt(6877 / 139755), math.sqrt(55016 / 717409), math.sqrt(49972 / 622545),
    math.sqrt(92480 / 1369599), math.sqrt(178802 / 978285), -math.sqrt(297680 / 5021863),
    -math.sqrt(719104 / 2282665), -math.sqrt(73644 / 2282665), -math.sqrt(2584 / 18865),
], [
    (4, 4, 6, 3),
    math.sqrt(117 / 1331), -math.sqrt(195 / 204974), math.sqrt(52 / 1089),
    math.sqrt(529 / 11979), -math.sqrt(2025 / 18634), -math.sqrt(49 / 395307),
    -math.sqrt(1369 / 35937), math.sqrt(68 / 11979), 0,
], [
    (2, 6, 6, 3),
    math.sqrt(2275 / 19602), math.sqrt(1625 / 143748), math.sqrt(325 / 199584),
    math.sqrt(6889 / 2195424), 71 / 198, -math.sqrt(1 / 223608),
    math.sqrt(625 / 81312), math.sqrt(1377 / 27104), math.sqrt(323 / 22869),
], [
    (4, 6, 6, 3),
    math.sqrt(12376 / 179685), math.sqrt(88400 / 1185921), -math.sqrt(442 / 12705),
    -math.sqrt(10880 / 251559), -math.sqrt(1088 / 179685), -math.sqrt(174080 / 8301447),
    -math.sqrt(8704 / 3773385), -math.sqrt(103058 / 1257795), -math.sqrt(19 / 31185),
], [
    (6, 6, 6, 1),
    math.sqrt(4199 / 539055), math.sqrt(29393 / 790614), math.sqrt(205751 / 784080),
    -math.sqrt(79135 / 1724976), math.sqrt(2261 / 1078110), math.sqrt(79135 / 175692),
    math.sqrt(15827 / 319440), -math.sqrt(8379 / 106480), -math.sqrt(98 / 1485),
]]


def judd_factor(i: int, c: int):
    assert 0 <= i < len(JUDD_TABLE)
    assert 1 <= c <= 9
    k1, k2, k3, mult = JUDD_TABLE[i][0]
    return k1, k2, k3, mult * JUDD_TABLE[i][c]


def matrix_UU(ion, k: int):
    assert 0 <= k <= 2 * ion.l
    return ion.matrix(f"UU/a/{k}") + 2 * ion.matrix(f"UU/b/{k}")


def matrix_TT(ion, k: int):
    assert 0 <= k <= 2 * ion.s
    return ion.matrix(f"TT/a/{k}") + 2 * ion.matrix(f"TT/b/{k}")


def matrix_UT(ion, k: int):
    assert 0 <= k <= 2 * ion.s
    return ion.matrix(f"UT/a/{k}") + 2 * ion.matrix(f"UT/b/{k}")


def matrix_L(ion, q: int):
    return math.sqrt(ion.l * (ion.l + 1) * (2 * ion.l + 1)) * ion.matrix(f"U/a/1,{q}")


def matrix_S(ion, q: int):
    return math.sqrt(1.5) * ion.matrix(f"T/a/1,{q}")


def matrix_J(ion, q: int):
    return ion.matrix(f"L/{q}") + ion.matrix(f"S/{q}")


def matrix_ED(ion, k: int, q: int):
    return ion.matrix(f"U/a/{k},{q}")


def matrix_MD(ion, q: int):
    return ion.matrix(f"L/{q}") + 2.00231924 * ion.matrix(f"S/{q}")


def matrix_Lz(ion):
    return ion.matrix("L/0")


def matrix_Sz(ion):
    return ion.matrix("S/0")


def matrix_Jz(ion):
    return ion.matrix("Lz") + ion.matrix("Sz")


def matrix_L2(ion):
    return ion.l * (ion.l + 1) * (2 * ion.l + 1) * ion.matrix("UU/1")


def matrix_S2(ion):
    return 1.5 * ion.matrix("TT/1")


def matrix_LS(ion):
    return math.sqrt(1.5 * ion.l * (ion.l + 1) * (2 * ion.l + 1)) * ion.matrix("UT/1")


def matrix_J2(ion):
    return ion.matrix("L2") + 2 * ion.matrix("LS") + ion.matrix("S2")


def matrix_GR(ion, d: int):
    assert d in (3, 5, 7)
    sum = 0.0
    for k in range(1, d, 2):
        sum += (2 * k + 1) * ion.matrix(f"UU/{k}")
    sum /= d - 2
    return sum


def matrix_GG(ion, d: int):
    assert d == 2
    return (3 * ion.matrix("UU/1") + 11 * ion.matrix("UU/5")) / 4


def matrix_H1(ion, k: int):
    l = ion.l
    assert 0 <= k <= 2 * l
    assert k % 2 == 0

    factor = (2 * l - 1) * wigner3j(l, k, l, 0, 0, 0)
    return factor * factor * ion.matrix(f"UU/{k}")


def matrix_H2(ion):
    l = ion.l
    factor = math.sqrt(1.5 * l * (l + 1) * (2 * l + 1))
    return factor * ion.matrix("UT/a/1")


def matrix_H3(ion, i: int):
    assert i in (0, 1, 2)
    if i == 0:
        matrix = ion.matrix("L2")
    elif i == 1:
        matrix = ion.matrix("GG/2")
    else:
        matrix = ion.matrix("GR/7")
    return matrix


def matrix_H4(ion, c: int):
    assert 1 <= c <= 9

    matrix = 0.0
    for i in range(len(JUDD_TABLE)):
        k1, k2, k3, factor = judd_factor(i, c)
        matrix += factor * 6 * math.sqrt((2 * k1 + 1) * (2 * k2 + 1) * (2 * k3 + 1)) \
                  * ion.matrix(f"UUU/c/{k1},{k2},{k3}")
    return matrix


def matrix_ss(ion, k: int):
    l = ion.l
    assert 0 <= k < 2 * l
    assert k % 2 == 0

    ck0 = -(2 * l + 1) * wigner3j(l, k, l, 0, 0, 0)
    ck2 = -(2 * l + 1) * wigner3j(l, k + 2, l, 0, 0, 0)
    factor = -12 * ck0 * ck2 * math.sqrt((k + 1) * (k + 2) * (2 * k + 1) * (2 * k + 3) * (2 * k + 5) / 5)
    return factor * ion.matrix(f"UUTT/b/{k},{k + 2},1,1,2")


def matrix_soo(ion, k: int):
    l = ion.l
    assert 0 <= k < 2 * l
    assert k % 2 == 0

    ck0 = -(2 * l + 1) * wigner3j(l, k, l, 0, 0, 0)
    factor0 = -ck0 * ck0 * math.sqrt((2 * l + k + 2) * (2 * l - k) * (k + 1) * (2 * k + 1) * (2 * k + 3))
    matrix0 = factor0 * (ion.matrix(f"UUTT/b/{k},{k + 1},1,0,1") + 2 * ion.matrix(f"UUTT/b/{k + 1},{k},0,1,1"))

    ck2 = -(2 * l + 1) * wigner3j(l, k + 2, l, 0, 0, 0)
    factor2 = -ck2 * ck2 * math.sqrt((2 * l + k + 3) * (2 * l - k - 1) * (k + 2) * (2 * k + 3) * (2 * k + 5))
    matrix2 = factor2 * (ion.matrix(f"UUTT/b/{k + 1},{k + 2},1,0,1") + 2 * ion.matrix(f"UUTT/b/{k + 1},{k + 2},0,1,1"))

    return 2 * (matrix0 + matrix2)


def matrix_H5(ion, k: int):
    return ion.matrix(f"ss/{k}") + ion.matrix(f"soo/{k}")


def matrix_H5fix(ion):
    return ion.matrix("H5/0") + 0.56 * ion.matrix("H5/2") + 0.38 * ion.matrix("H5/4")


def matrix_H6(ion, k: int):
    l = ion.l
    assert 0 < k <= 2 * l
    assert k % 2 == 0

    matrix = 0.0
    if k > 0:
        factor = math.sqrt((2 * l + k + 1) * (2 * l - k + 1) * k * (2 * k - 1) / (2 * k + 1))
        matrix += factor * ion.matrix(f"UUTT/b/{k},{k - 1},0,1,1")
    if k < 2 * l:
        factor = -math.sqrt((2 * l + k + 2) * (2 * l - k) * (k + 1) * (2 * k + 3) / (2 * k + 1))
        matrix += factor * ion.matrix(f"UUTT/b/{k},{k + 1},0,1,1")

    ck = -(2 * l + 1) * wigner3j(l, k, l, 0, 0, 0)
    return (2 * ck * ck / 6) * matrix


def matrix_H6fix(ion):
    return ion.matrix("H6/2") + 0.75 * ion.matrix("H6/4") + 0.50 * ion.matrix("H6/6")


MATRIX = {
    "UU": matrix_UU,
    "TT": matrix_TT,
    "UT": matrix_UT,
    "L": matrix_L,
    "S": matrix_S,
    "J": matrix_J,
    "ED": matrix_ED,
    "MD": matrix_MD,
    "Lz": matrix_Lz,
    "Sz": matrix_Sz,
    "Jz": matrix_Jz,
    "L2": matrix_L2,
    "S2": matrix_S2,
    "LS": matrix_LS,
    "J2": matrix_J2,
    "GR": matrix_GR,
    "GG": matrix_GG,
    "H1": matrix_H1,
    "H2": matrix_H2,
    "H3": matrix_H3,
    "H4": matrix_H4,
    "ss": matrix_ss,
    "soo": matrix_soo,
    "H5": matrix_H5,
    "H5fix": matrix_H5fix,
    "H6": matrix_H6,
    "H6fix": matrix_H6fix,
}


class Matrix:

    def __init__(self, ion, array, name=None, coupling=None):
        assert name is None or isinstance(name, str)
        assert coupling is None or isinstance(coupling, Coupling)

        self.ion = ion
        self.name = name
        self.coupling = coupling if coupling else Coupling.Product
        self.states = self.ion.states(coupling)
        self.array = array
        self.shape = self.array.shape

    def __neg__(self):
        return Matrix(self.ion, -self.array)

    def __add__(self, other):
        if not other:
            return self
        elif isinstance(other, Matrix):
            if self.coupling != other.coupling:
                raise ValueError("Cannot add matrices with different couplings!")
            return Matrix(self.ion, self.array + other.array)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Matrix):
            if self.coupling != other.coupling:
                raise ValueError("Cannot subtract matrices with different couplings!")
            return Matrix(self.ion, self.array - other.array)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, Matrix):
            if self.coupling != other.coupling:
                raise ValueError("Cannot subtract matrices with different couplings!")
            return Matrix(self.ion, other.array - self.array)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return Matrix(self.ion, self.array * other)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (float, int)):
            return Matrix(self.ion, self.array / other)
        return NotImplemented

    def diagonalize(self):
        values, vectors = np.linalg.eigh(self.array)
        return values, vectors

    def fast_diagonalize(self):
        """ Fast diagonalization algorithm acting inside J spaces. """
        if self.coupling not in (Coupling.SLJM, Coupling.SLJ):
            raise RuntimeError("Fasr diagonalization is only available for SLJM or SLJ coupling!")

        # Initialize eigenvalues and eigenvectors
        num_states = len(self.states)
        values = np.zeros(num_states, dtype=float)
        vectors = np.zeros((num_states, num_states), dtype=float)

        # Diagonalize hamiltonian in each J sub-space
        for i, j in self.states.J_slices:
            if j - i == 1:
                values[i] = self.array[i, i]
                vectors[i, i] = 1.0
            else:
                a, b = np.linalg.eigh(self.array[i:j, i:j])
                values[i:j] = a
                vectors[i:j, i:j] = b

        # Sort results for increasing order of eigenvalues
        indices = np.argsort(values)
        values = values[indices]
        vectors = vectors[:, indices]

        # Return eigenvalues and eigenvectors
        return values, vectors

    def transform(self, coupling):
        # Coupling matches already
        if coupling == self.coupling:
            return self

        # Back-transform to product states
        if self.coupling == Coupling.Product:
            array = self.array
        else:
            transform = self.ion.states(self.coupling).transform
            array = transform @ self.array @ transform.T

        # Transform to desired coupling
        if coupling != Coupling.Product:
            transform = self.ion.states(coupling).transform
            array = transform.T @ array @ transform

        # Return matrix in new state
        return Matrix(self.ion, array, self.name, coupling)


def reduced_matrix(ion, k: int, J: np.ndarray, transform: np.ndarray) -> np.ndarray:
    assert k is None or (0 <= k <= 2 * ion.l)
    assert isinstance(J, np.ndarray)
    assert isinstance(transform, np.ndarray)
    assert len(J.shape) == 1
    assert len(transform.shape) == 2
    assert transform.shape[0] == transform.shape[1] == J.shape[0]

    if k is None:
        k = 1
        name = "MD/%d"
    else:
        name = f"ED/{k},%d"
    num_states = len(J)
    hyper = np.zeros((2 * k + 1, num_states, num_states), dtype=float)
    for q in range(-k, k + 1):
        hyper[q + k, :, :] = transform.T @ ion.cached_matrix(name % q, Coupling.SLJ).array @ transform

    def value(i: int, j: int):
        Ja = J[i]
        Jb = J[j]
        q = Ja - Jb
        if q < -k or q > k:
            return 0.0
        factor = wigner3j(Ja, k, Jb, -Ja, q, Jb)
        if factor == 0.0:
            return 0.0
        return hyper[q + k, i, j] / factor

    return np.array([[value(i, j) for j in range(num_states)] for i in range(num_states)], dtype=float)


def build_hamilton(ion, radial, coupling):
    assert coupling in (Coupling.SLJM, Coupling.SLJ)
    assert isinstance(radial, dict)

    num_states = len(ion.states(coupling))
    array = np.zeros((num_states, num_states), dtype=float)
    for name in radial:
        if name == "base":
            continue
        array += radial[name] * ion.cached_matrix(name, coupling).array
    return Matrix(ion, array, "H", coupling)


##################################################
# HDF5 interface
##################################################

STORE = ["H1", "H2", "H3", "H4", "H5", "H6"]

def get_matrix(ion, name, coupling=None):
    assert isinstance(name, str)
    assert coupling is None or isinstance(coupling, Coupling)

    coupling = coupling or Coupling.Product

    # Matrix of unit tensor operator
    if name.count("/") == 2:
        array = get_unit(ion, name)
        return Matrix(ion, array, name).transform(coupling)

    # Get name and arguments for high order operator
    if "/" in name:
        main, args = name.split("/")
    else:
        main, args = name, ""
    if main not in MATRIX:
        raise ValueError(f"Unknown matrix: {name}")
    args = map(int, args.split(",")) if args else ()

    # Build matrix from scratch
    if main not in STORE or coupling not in (Coupling.SLJM, Coupling.SLJ):
        return MATRIX[main](ion, *args).transform(coupling)

    # Get SLJM matrix from HDF5 vault
    if coupling == Coupling.SLJM:
        group = ion.matrix_vault[Coupling.SLJM.name]
        if name in group:
            matrix = Matrix(ion, np.array(group[name]), name, Coupling.SLJM)
        else:
            print(f"SLJM matrix {name}: generate")
            matrix = MATRIX[main](ion, *args).transform(Coupling.SLJM)
            group.create_dataset(name, data=matrix.array, compression="gzip", compression_opts=9)
            ion.vault.flush()

    # Get SLJ matrix from HDF5 vault
    elif coupling == Coupling.SLJ:
        group = ion.matrix_vault[Coupling.SLJ.name]
        if name in group:
            matrix = Matrix(ion, np.array(group[name]), name, Coupling.SLJ)
        else:
            matrix = get_matrix(ion, name, Coupling.SLJM).transform(Coupling.SLJ)
            group.create_dataset(name, data=matrix.array, compression="gzip", compression_opts=9)
            ion.vault.flush()

    # Other couplings are not supported (unit tensor matrices use their own vault)
    else:
        print(name, coupling)
        raise RuntimeError(f"Matrix storage is supported only for SLJM and SLJ matrices.")

    return matrix


def init_matrix(vault, group_name: str):
    if group_name in vault:
        if not vault.attrs["valid"] or "version" not in vault[group_name].attrs or vault[group_name].attrs["version"] != MATRIX_VERSION:
            del vault[group_name]
            vault.flush()

    if group_name not in vault:
        vault.attrs["valid"] = False
        group = vault.create_group(group_name)
        vault[group_name].attrs["version"] = MATRIX_VERSION
        group.create_group(Coupling.SLJM.name)
        group.create_group(Coupling.SLJ.name)
        vault.flush()

    return vault[group_name]


if __name__ == "__main__":
    from lanthanide import Lanthanide

    with Lanthanide(1) as ion:
        print(ion)
        states = ion.states(Coupling.SLJM)
        H = ion.matrix("H1/4", Coupling.SLJM)
