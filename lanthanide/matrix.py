##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This module provides the calculation of matrices of higher-order tensor
# operators in the determinantal product space of a given electron
# configuration based on unit tensor operators from the module "unit".
#
# The function get_matrix() returns a Matrix object containing a given
# tensor operator matrix in a given coupling scheme. The matrices of
# certain tensor operators, namely the perturbation hamiltonians are
# eventually stored in the HDF5 file cache in SLJM and SLJ coupling.
#
# Matrix objects support diagonalisation and elementary arithmetic
# operations to build for linear combinations of these matrices.
#
##########################################################################

from functools import lru_cache
import math
import numpy as np

from .wigner import wigner3j
from .unit import get_unit
from .state import Coupling

# Version of the algorithms for the calculation of high-order tensor operators in this module. If the precomputed
# matrices in the file cache come with another version number, they will be recomputed.
MATRIX_VERSION = 2


##########################################################################
# Unit tensor operators
##########################################################################

def matrix_UU(ion, k: int):
    """ Return the matrix of the scalar product of two unit tensor operators of rank k in the orbital angular momentum
    space in determinantal product state coupling. """

    assert 0 <= k <= 2 * ion.l
    return get_matrix(ion, f"UU/a/{k}") + 2 * get_matrix(ion, f"UU/b/{k}")


def matrix_TT(ion, k: int):
    """ Return the matrix of the scalar product of two unit tensor operators of rank k in the spin angular momentum
    space in determinantal product state coupling. """

    assert 0 <= k <= 2 * ion.s
    return get_matrix(ion, f"TT/a/{k}") + 2 * get_matrix(ion, f"TT/b/{k}")


def matrix_UT(ion, k: int):
    """ Return the matrix of the scalar product of two unit tensor operators of rank k in the orbital and spin angular
    momentum space in determinantal product state coupling. """

    assert 0 <= k <= 2 * ion.s
    return get_matrix(ion, f"UT/a/{k}") + 2 * get_matrix(ion, f"UT/b/{k}")


##########################################################################
# Angular momentum tensor operators
##########################################################################

def matrix_L(ion, q: int):
    """ Return the matrix of the component q of the tensor operator of the total orbital angular momentum in
    determinantal product state coupling. """

    return math.sqrt(ion.l * (ion.l + 1) * (2 * ion.l + 1)) * get_matrix(ion, f"U/a/1,{q}")


def matrix_S(ion, q: int):
    """ Return the matrix of the component q of the tensor operator of the total spin angular momentum in
    determinantal product state coupling. """

    return math.sqrt(1.5) * get_matrix(ion, f"T/a/1,{q}")


def matrix_J(ion, q: int):
    """ Return the matrix of the component q of the tensor operator of the total angular momentum in determinantal
    product state coupling. """

    return get_matrix(ion, f"L/{q}") + get_matrix(ion, f"S/{q}")


def matrix_Lz(ion):
    """ Return the matrix of the z component of the tensor operator of the total orbital angular momentum in
    determinantal product state coupling. """

    return get_matrix(ion, "L/0")


def matrix_Sz(ion):
    """ Return the matrix of the z component of the tensor operator of the total spin angular momentum in
    determinantal product state coupling. """

    return get_matrix(ion, "S/0")


def matrix_Jz(ion):
    """ Return the matrix of the z component of the tensor operator of the total angular momentum in determinantal
    product state coupling. """

    return get_matrix(ion, "Lz") + get_matrix(ion, "Sz")


def matrix_L2(ion):
    """ Return the matrix of the squared tensor operator of the total orbital angular momentum in determinantal
    product state coupling. """

    return ion.l * (ion.l + 1) * (2 * ion.l + 1) * get_matrix(ion, "UU/1")


def matrix_S2(ion):
    """ Return the matrix of the squared tensor operator of the total spin angular momentum in determinantal
    product state coupling. """

    return 1.5 * get_matrix(ion, "TT/1")


def matrix_LS(ion):
    """ Return the matrix of the scalar product of the tensor operators of the total orbital angular momentum and
    total spin angular momentum in determinantal product state coupling. """

    return math.sqrt(1.5 * ion.l * (ion.l + 1) * (2 * ion.l + 1)) * get_matrix(ion, "UT/1")


def matrix_J2(ion):
    """ Return the matrix of the squared tensor operator of the total angular momentum in determinantal product
    state coupling. """

    return get_matrix(ion, "L2") + 2 * get_matrix(ion, "LS") + get_matrix(ion, "S2")


##########################################################################
# Radiative transition tensor operators
##########################################################################

def matrix_ED(ion, k: int, q: int):
    """ Return the matrix of the component q of the unit tensor operator of rank k in determinantal product state
    coupling as required for the calculation of electric dipole transitions. Note that for each matrix element only
    one component q may be non-zero depending on the initial and final states. """

    return get_matrix(ion, f"U/a/{k},{q}")


def matrix_MD(ion, q: int):
    """ Return the matrix of the component q of the tensor operator of the total orbital angular momentum plus the
    tensor operator of the total spin angular momentum times the spin-g-factor in determinantal product state coupling
    as required for the calculation of magnetic dipole transitions. Note that for each matrix element only one
    component q may be non-zero depending on the initial and final states. """

    return get_matrix(ion, f"L/{q}") + 2.00231924 * get_matrix(ion, f"S/{q}")


##########################################################################
# Tensor operator of the perturbation hamiltonian H1
##########################################################################

def matrix_H1(ion, k: int):
    """ Return the matrix of the perturbation hamiltonian H1/k in determinantal product state coupling. """

    assert 0 <= k <= 2 * ion.l
    assert k % 2 == 0

    l = ion.l
    factor = (2 * l + 1) * wigner3j(l, k, l, 0, 0, 0)
    return factor * factor * get_matrix(ion, f"UU/b/{k}")


##########################################################################
# Tensor operator of the perturbation hamiltonian H2
##########################################################################

def matrix_H2(ion):
    """ Return the matrix of the perturbation hamiltonian H2 in determinantal product state coupling. """

    l = ion.l
    factor = math.sqrt(1.5 * l * (l + 1) * (2 * l + 1))
    return factor * get_matrix(ion, "UT/a/1")


##########################################################################
# Tensor operator of the perturbation hamiltonian H3
##########################################################################

def matrix_GR(ion, d: int):
    """ Return the matrix of the Casimir operator G(Rd) of the rotational group in d dimensions in determinantal
    product state coupling. """

    assert d in (3, 5, 7)

    sum = 0.0
    for k in range(1, d, 2):
        sum += (2 * k + 1) * get_matrix(ion, f"UU/{k}")
    sum /= d - 2
    return sum


def matrix_GG(ion, d: int):
    """ Return the matrix of the Casimir operator G(G2) of the special group G2 in determinantal product state
    coupling. """

    assert d == 2

    return (3 * get_matrix(ion, "UU/1") + 11 * get_matrix(ion, "UU/5")) / 4


def matrix_H3(ion, i: int):
    """ Return the matrix of the perturbation hamiltonian H2/i in determinantal product state coupling. """

    assert i in (0, 1, 2)

    if i == 0:
        matrix = get_matrix(ion, "L2")
    elif i == 1:
        matrix = get_matrix(ion, "GG/2")
    else:
        matrix = get_matrix(ion, "GR/7")
    return matrix


##########################################################################
# Tensor operator of the perturbation hamiltonian H4
##########################################################################

# These data tables are required for the calculation of the perturbation hamiltonian H4 of intra-configuration
# interactions which is based on effective three-electron scalar products
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
    """ Return the ranks k1, k2, and k3 of the three factors of the triple scalar product for the summand i for the
    perturbation hamiltonian H4/{c}. """

    assert 0 <= i < len(JUDD_TABLE)
    assert 1 <= c <= 9

    k1, k2, k3, mult = JUDD_TABLE[i][0]
    return k1, k2, k3, mult * JUDD_TABLE[i][c]


def matrix_H4(ion, c: int):
    """ Return the matrix of the perturbation hamiltonian H4/c in determinantal product state coupling. """

    assert 1 <= c <= 9

    matrix = 0.0
    for i in range(len(JUDD_TABLE)):
        k1, k2, k3, factor = judd_factor(i, c)
        matrix += factor * 6 * math.sqrt((2 * k1 + 1) * (2 * k2 + 1) * (2 * k3 + 1)) \
                  * get_matrix(ion, f"UUU/c/{k1},{k2},{k3}")
    return matrix


##########################################################################
# Tensor operator of the perturbation hamiltonian H5
##########################################################################

def matrix_ss(ion, k: int):
    """ Return the matrix of the spin-spin interaction in determinantal product state coupling. """

    assert 0 <= k < 2 * ion.l
    assert k % 2 == 0

    l = ion.l
    ck0 = -(2 * l + 1) * wigner3j(l, k, l, 0, 0, 0)
    ck2 = -(2 * l + 1) * wigner3j(l, k + 2, l, 0, 0, 0)
    factor = -12 * ck0 * ck2 * math.sqrt((k + 1) * (k + 2) * (2 * k + 1) * (2 * k + 3) * (2 * k + 5) / 5)
    return factor * get_matrix(ion, f"UUTT/b/{k},{k + 2},1,1,2")


def matrix_soo(ion, k: int):
    """ Return the matrix of the spin-other-orbit interaction in determinantal product state coupling. """

    assert 0 <= k < 2 * ion.l
    assert k % 2 == 0

    l = ion.l

    ck0 = -(2 * l + 1) * wigner3j(l, k, l, 0, 0, 0)
    factor0 = -ck0 * ck0 * math.sqrt((2 * l + k + 2) * (2 * l - k) * (k + 1) * (2 * k + 1) * (2 * k + 3))
    matrix0 = factor0 * (get_matrix(ion, f"UUTT/b/{k},{k + 1},0,1,1") \
                         + 2 * get_matrix(ion, f"UUTT/b/{k + 1},{k},0,1,1"))

    ck2 = -(2 * l + 1) * wigner3j(l, k + 2, l, 0, 0, 0)
    factor2 = -ck2 * ck2 * math.sqrt((2 * l + k + 3) * (2 * l - k - 1) * (k + 2) * (2 * k + 3) * (2 * k + 5))
    matrix2 = factor2 * (get_matrix(ion, f"UUTT/b/{k + 2},{k + 1},0,1,1") \
                         + 2 * get_matrix(ion, f"UUTT/b/{k + 1},{k + 2},0,1,1"))

    return 2 * (matrix0 + matrix2)


def matrix_H5(ion, k: int):
    """ Return the matrix of the perturbation hamiltonian H5/k in determinantal product state coupling. """

    return get_matrix(ion, f"hss/{k}") + get_matrix(ion, f"hsoo/{k}")


def matrix_H5fix(ion):
    """ Return the linear combination H5/0 + 0.56 * H5/2 + 0.38 * H5/4 of the total perturbation hamiltonian H5 in
    determinantal product state coupling. """

    return get_matrix(ion, "H5/0") \
        + 0.56 * get_matrix(ion, "H5/2") \
        + 0.38 * get_matrix(ion, "H5/4")


##########################################################################
# Tensor operator of the perturbation hamiltonian H6
##########################################################################

def matrix_H6(ion, k: int):
    """ Return the matrix of the perturbation hamiltonian H6/k in determinantal product state coupling. """

    assert 0 < k <= 2 * ion.l
    assert k % 2 == 0

    l = ion.l

    matrix = 0.0
    if k > 0:
        factor = math.sqrt((2 * l + k + 1) * (2 * l - k + 1) * k * (2 * k - 1) / (2 * k + 1))
        matrix += factor * get_matrix(ion, f"UUTT/b/{k},{k - 1},0,1,1")
    if k < 2 * l:
        factor = -math.sqrt((2 * l + k + 2) * (2 * l - k) * (k + 1) * (2 * k + 3) / (2 * k + 1))
        matrix += factor * get_matrix(ion, f"UUTT/b/{k},{k + 1},0,1,1")

    ck = -(2 * l + 1) * wigner3j(l, k, l, 0, 0, 0)
    return (2 * ck * ck / 6) * matrix


def matrix_H6fix(ion):
    """ Return the linear combination H6/2 + 0.75 * H6/4 + 0.50 * H6/6 of the total perturbation hamiltonian H6 in
    determinantal product state coupling. """

    return get_matrix(ion, "H6/2") \
        + 0.75 * get_matrix(ion, "H6/4") \
        + 0.50 * get_matrix(ion, "H6/6")


##########################################################################
# Module interface
##########################################################################

# This dictionary is used to determine the respective function for the calculation of a tensor operator matrix
# identified by name string.
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
    "hss": matrix_ss,
    "hsoo": matrix_soo,
    "H5": matrix_H5,
    "H5fix": matrix_H5fix,
    "H6": matrix_H6,
    "H6fix": matrix_H6fix,
}


class Matrix:
    """ This class holds the matrix of a tensor operator in arbitrary coupling. It supports basic arithmetic
    operations to allow for linear combinations of several Matrix objects. It also supports diagonalisation and
    transformation to another coupling scheme. """

    def __init__(self, ion, array: np.ndarray, name=None, coupling=None):
        """ Initialize a new tensor operator matrix for the given lanthanide ion by the given numpy array in the
        given coupling scheme (default Coupling.Product) with an optional name string. """

        assert isinstance(array, np.ndarray)
        assert name is None or isinstance(name, str)
        assert coupling is None or isinstance(coupling, Coupling)

        self.ion = ion
        self.name = name
        self.coupling = coupling or Coupling.Product
        self.array = array
        self.shape = self.array.shape

    def __neg__(self):
        """ Return this matrix multiplied by -1. """

        return Matrix(self.ion, -self.array)

    def __add__(self, other):
        """ Return this + other matrix. Return this matrix if other is zero. """

        if not other:
            return self
        elif isinstance(other, Matrix):
            if self.ion != other.ion:
                raise ValueError("Cannot add matrices of different ions!")
            if self.coupling != other.coupling:
                raise ValueError("Cannot add matrices with different couplings!")
            return Matrix(self.ion, self.array + other.array)
        return NotImplemented

    def __radd__(self, other):
        """ Return other + this matrix. """

        return self.__add__(other)

    def __sub__(self, other):
        """ Return this - other matrix. """

        if isinstance(other, Matrix):
            if self.ion != other.ion:
                raise ValueError("Cannot add matrices of different ions!")
            if self.coupling != other.coupling:
                raise ValueError("Cannot subtract matrices with different couplings!")
            return Matrix(self.ion, self.array - other.array)
        return NotImplemented

    def __rsub__(self, other):
        """ Return other - this matrix. """

        if isinstance(other, Matrix):
            if self.ion != other.ion:
                raise ValueError("Cannot add matrices of different ions!")
            if self.coupling != other.coupling:
                raise ValueError("Cannot subtract matrices with different couplings!")
            return Matrix(self.ion, other.array - self.array)
        return NotImplemented

    def __mul__(self, other):
        """ Return this matrix multiplied by a real factor. """

        if isinstance(other, (float, int)):
            return Matrix(self.ion, self.array * other)
        return NotImplemented

    def __rmul__(self, other):
        """ Return this matrix multiplied by a real factor. """

        return self.__mul__(other)

    def __truediv__(self, other):
        """ Return this matrix divided by a real factor. """

        if isinstance(other, (float, int)):
            return Matrix(self.ion, self.array / other)
        return NotImplemented

    def diagonalise(self):
        """ Standard diagonalisation algorithm acting on the whole matrix. Return the ordered numpy vector of
        eigenvalues and the 2D numpy array of eigenvectors (columns). """

        values, vectors = np.linalg.eigh(self.array)
        return values, vectors

    def fast_diagonalise(self):
        """ Fast diagonalisation algorithm acting inside J spaces only. Return the ordered numpy vector of
        eigenvalues and the 2D numpy array of eigenvectors (columns). """

        if self.coupling not in (Coupling.SLJM, Coupling.SLJ):
            raise RuntimeError("Fast diagonalisation is only available for SLJM or SLJ coupling!")

        # Initialize eigenvalues and eigenvectors
        states = self.ion.states(self.coupling)
        num_states = len(states)
        values = np.zeros(num_states, dtype=float)
        vectors = np.zeros((num_states, num_states), dtype=float)

        # Diagonalize hamiltonian in each J sub-space
        for i, j in states.J_slices:
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

    def transform(self, coupling: Coupling):
        """ Return this matrix transformed to the given coupling scheme. """

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

        # Return matrix in new coupling scheme
        return Matrix(self.ion, array, self.name, coupling)


def build_hamilton(ion, radial: dict, coupling: Coupling):
    """ Build and return the matrix of a perturbation hamiltonian operator as linear combination of the interaction
    hamiltonians and factors specified in the dictionary radial in the given coupling scheme."""

    assert coupling in (Coupling.SLJM, Coupling.SLJ)
    assert isinstance(radial, dict)

    # Initialize empty hamiltonian matrix
    num_states = len(ion.states(coupling))
    array = np.zeros((num_states, num_states), dtype=float)

    # Build linear combination specified in the radial dictionary
    for name in radial:
        if name == "base":
            continue
        if name[:1] != "H":
            continue
        array += radial[name] * get_matrix(ion, name, coupling).array

    # Return the total hamiltonian as Matrix object in the given coupling scheme
    return Matrix(ion, array, "H", coupling)


def reduced_matrix(ion, name: str, k: int, J: list, transform=None) -> np.ndarray:
    """ Return a matrix of reduced matrix elements derived from the components of the operator with given name and
    rank k. The parameter is a list of quantum numbers J of the total angular momentum for all states. If a
    transformation matrix is given, it is used to transform the operator to an intermediate SLJ coupling. Note
    that the intermediate coupling must preserve the quantum number J of each state. """

    assert 0 <= k <= 2 * ion.l
    assert isinstance(J, list)
    if transform is not None:
        assert isinstance(transform, np.ndarray)
        assert len(transform.shape) == 2
        assert transform.shape[0] == transform.shape[1] == len(J)

    # Matrix of the potentially non-zero components or the tensor operator of rank k
    num_states = len(J)
    if k == 0:
        array = get_matrix(ion, name.format(q=0), Coupling.SLJ).array
    else:
        array = np.zeros((num_states, num_states), dtype=float)
        for q in range(-k, k + 1):
            array += get_matrix(ion, name.format(q=q), Coupling.SLJ).array

    # Transform to intermediate coupling
    if transform is not None:
        array = transform.T @ array @ transform

    def value(i: int, j: int):
        """ Apply the Wigner-Eckart theorem to the given matrix element of array. """

        Ja = J[i]
        Jb = J[j]
        q = Ja - Jb
        if q < -k or q > k:
            return 0.0
        factor = wigner3j(Ja, k, Jb, -Ja, q, Jb)
        if factor == 0.0:
            return 0.0
        return array[i, j] / factor

    # Apply the Wigner-Eckart theorem to every matrix element and return the matrix of reduced tensor matrix elements
    return np.array([[value(i, j) for j in range(num_states)] for i in range(num_states)], dtype=float)


##########################################################################
# HDF5 cache interface
##########################################################################

# List of tensor operators which are eventually stored in the file cache
STORE = ["H1", "H2", "H3", "H4", "H5", "H6"]


@lru_cache(maxsize=None)
def get_matrix(ion, name, coupling=None):
    """ Return a Matrix object of the matrix of the tensor operator with given name and in the given coupling
    scheme (default: Coupling.Product). The matrices of the tensor operators in the list STORE are stored in the
    HDF5 file cache in SLJM and SLJ coupling. """

    assert isinstance(name, str)
    assert coupling is None or isinstance(coupling, Coupling)

    # Coupling scheme of the returned Matrix object
    coupling = coupling or Coupling.Product

    # Tensor name
    tensor = name if "/" not in name else name.split("/")[0]

    # Build matrix
    if tensor not in STORE or coupling not in (Coupling.SLJM, Coupling.SLJ) or not hasattr(ion, "vault"):
        if name.count("/") == 2:
            array = get_unit(ion, name)
        else:
            if "/" in name:
                main, args = name.split("/")
            else:
                main, args = name, ""
            if main not in MATRIX:
                raise ValueError(f"Unknown matrix: {name}")
            args = map(int, args.split(",")) if args else ()
            array = MATRIX[main](ion, *args).array

        return Matrix(ion, array, name).transform(coupling)

    # Get SLJM matrix from HDF5 vault
    if coupling == Coupling.SLJM:
        group = ion.matrix_vault[Coupling.SLJM.name]
        if name in group:
            matrix = Matrix(ion, np.array(group[name]), name, Coupling.SLJM)
        else:
            print(f"Create SLJM matrix {name} ...")
            matrix = get_matrix(ion, name).transform(Coupling.SLJM)
            group.create_dataset(name, data=matrix.array, compression="gzip", compression_opts=9)
            ion.vault.flush()
            print(f"SLJM matrix {name} done.")

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
    """ Initialize and return the cache for the storage of higher order tensor operator matrices in SLJM and SLJ
    coupling in the HDF5 group with given name in the given HDF5 file vault. """

    # Delete the group in the HDF5 file, if the cache is marked as invalid or its version number does not match
    if group_name in vault:
        if not vault.attrs["valid"] or "version" not in vault[group_name].attrs or vault[group_name].attrs[
            "version"] != MATRIX_VERSION:
            del vault[group_name]
            vault.flush()

    # Create a new HDF5 group with current version number if it is missing
    if group_name not in vault:
        vault.attrs["valid"] = False
        group = vault.create_group(group_name)
        vault[group_name].attrs["version"] = MATRIX_VERSION
        group.create_group(Coupling.SLJM.name)
        group.create_group(Coupling.SLJ.name)
        vault.flush()

    # Return the HDF5 group of the higher-order tensor matrix cache
    return vault[group_name]
