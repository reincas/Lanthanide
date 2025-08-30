##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This module provides classes for the representation of single states
# and lists of all states of a configuration in the determinantal product
# state space and in SLJM, SLJ and intermediate coupling. The main tasks
# of these objects is the generation of short and long string
# representations of the states and the provision of transformation
# matrices between the different coupling schemes.
#
##########################################################################

from enum import Enum
import numpy as np

from .wigner import wigner3j
from .unit import ORBITAL, SPIN, MAGNETIC
from .symmetry import CHR_ORBITAL, SYMMETRY, SymmetryList

# Version of the algorithms for coupled electron states in this module. If the precomputed states and transformation
# matrix in the file cache come with another version number, they will be recomputed. This will also render all other
# elements following in the chain of dependent cache elements invalid, which is the module 'matrix', see the init
# function of the Lanthanide class.
TERM_VERSION = 2

# Chain of symmetry group operators used to classify the SLJM and SLJ states. The order in these tuples relates to
# the columns of the eigenvalue matrices.
SYM_CHAIN_SLJM = ("S2", "GR/7", "GG/2", "L2", "tau", "J2", "num", "Jz")
SYM_CHAIN_SLJ = SYM_CHAIN_SLJM[:-1]


class Coupling(Enum):
    """ This enumeration class is used to mark the four coupling schemes used in the Lanthanide package: determinantal
    product state coupling, SLJM coupling, SLJ coupling, and intermediate coupling. """

    Product = 0
    SLJM = 1
    SLJ = 2
    J = 3


def val2key(values, sym_chain=SYM_CHAIN_SLJM):
    """ Convert a float array of symmetry values to an integer symmetry key array. The rows of the arrays are
    referring to the electron states and the columns refer to the chain of symmetry operators of which the names
    are given in the parameter sym_chain. """

    assert isinstance(values, np.ndarray)
    assert len(values.shape) == 2
    assert values.dtype == float
    assert values.shape[1] == len(sym_chain)

    return np.array([SymmetryList(values[:, i], name).keys for i, name in enumerate(SYM_CHAIN_SLJM)], dtype=int).T


class StateList:
    """ Abstract class for a list of electron states in a certain coupling scheme. The abstract defines some common
    methods acting on the common attribute states, which contains the list of State objects. """

    states = []

    def __len__(self):
        """ Returns the number of states. """

        return len(self.states)

    def __iter__(self):
        """ Generate each state. """

        for state in self.states:
            yield state

    def __getitem__(self, item: int):
        """ Return the state with the given index. """

        return self.states[item]

    def short(self):
        """ Return a list containing the short names of all states. """

        return [state.short() for state in self.states]

    def long(self):
        """ Return a list containing the long names of all states. """

        return [state.long() for state in self.states]


##########################################################################
# Product states
##########################################################################

class StateProduct:
    """ Class for a determinantal product state. """

    def __init__(self, values):
        """ Store the index values of the electrons and all of their quantum numbers. """

        # Common electron quantum numbers of the orbital angular momentum and the spin
        self.l = ORBITAL
        self.s = SPIN

        # List of all magnetic quantum numbers ml, ms for electrons in the configuration
        self.m = MAGNETIC

        # Index values and quantum numbers of all electrons of the state
        self.values = list(values)
        self.quantum = [(self.l, self.m[i][0], self.s, self.m[i][1]) for i in self.values]

    def short(self):
        """ Return a short string representation of the state. """

        quantum = [(ml, "du"[(2 * ms + 1) // 2]) for l, ml, s, ms in self.quantum]
        return " ".join(f"{ml:+d}{ms}" for ml, ms in quantum)

    def long(self):
        """ Return a long string representation of the state. """

        quantum = [(CHR_ORBITAL[l].lower(), ml, 2 * s, 2 * ms) for l, ml, s, ms in self.quantum]
        return " ".join(f"{l},{ml:+d},{s}/2,{ms:+d}/2" for l, ml, s, ms in quantum)

    def __getitem__(self, item: int):
        """ Return the quantum numbers l, s, ml, and ms of the state with the given index. """

        return self.quantum[item]

    def __str__(self):
        """ Return a long string representation of the state. """

        return self.long()


class StateListProduct(StateList):
    """ Class containing a list of StateProduct objects representing an electron configuration. """

    def __init__(self, values):
        """ Store the given array of electron indices and build the list of StateProduct objects. """

        assert len(set(len(state) for state in values)) == 1

        # Array of electron indices. Rows represent states, columns individual electrons in each state.
        self.values = np.array(values)

        # List of all StateProduct objects
        self.states = [StateProduct(v) for v in values]

        # No transformation matrix
        self.transform = None

    def to_SLJM(self, ion):
        """ Build and return a StateListSLJM object using the function ion.matrix(name) to get the matrix of
         a symmetry operator of given name in determinantal product state coupling. """

        get_array = lambda name: ion.matrix(name).array
        values, transform = build_SLJM(ORBITAL, len(self), get_array)
        return StateListSLJM(values, transform)

    def __str__(self):
        """ Return a string representation of the list of states. """

        return f"<List of {len(self)} product states>"


##########################################################################
# SLJM states
##########################################################################

class StateSLJM:
    """ Class for an electron state in SLJM coupling following the chain of symmetry operators in SYM_CHAIN_SLJM. """

    def __init__(self, values):
        """ Store the eigenvalues of the state and build a dictionary containing a Symmetry object for each link in
        the chain of symmetry operators in SYM_CHAIN_SLJM. """

        self.sym_chain = SYM_CHAIN_SLJM
        assert len(values) == len(self.sym_chain)
        self.values = list(values)
        self.symmetries = dict((name, SYMMETRY[name](value)) for name, value in zip(SYM_CHAIN_SLJM, self.values))

    def keys(self):
        """ Generate the symmetry keys of the state in the chain order. """

        for sym in self.sym_chain:
            yield sym

    def short(self):
        """ Return a short string representation of the state. """

        num = f"({self['num']})" if self["num"].key > 0 else ""
        return f"{self['S2']}{self['L2']}{self['J2']}{num} {self['Jz']}"

    def long(self):
        """ Return a long string representation of the state. """

        tau = f"{self['tau']}" if self["tau"].value > 0 else ""
        num = f"({self['num']})" if self["num"].key > 0 else ""
        return f"{self['S2']}{self['L2']} {self['GR/7']} {self['GG/2']} {self['J2']}{tau}{num} {self['Jz']}"

    # def key(self, sym_names=None):
    #     if sym_names is None:
    #         sym_names = self.sym_chain
    #     return " ".join([str(self[sym]) for sym in sym_names])

    def __getitem__(self, sym_name: str):
        """ Return the Symmetry object for the given operator name. """

        if sym_name not in self.symmetries:
            raise KeyError(f"Unknown symmetry {sym_name}!")
        return self.symmetries[sym_name]

    def __str__(self):
        """ Return a long string representation of the state. """

        return self.long()


class StateListSLJM(StateList):
    """ Class containing a list of StateSLJM objects representing an electron configuration. """

    def __init__(self, values, transform):
        """ Store the given eigenvalue matrix, as well as the transformation matrix from the determinantal product
        state space to SLJM coupling, and build the list of StateSLJM objects. """

        assert len(values.shape) == 2 and values.shape[1] == len(SYM_CHAIN_SLJM)
        assert len(transform.shape) == 2 and transform.shape[0] == transform.shape[1]
        assert values.shape[0] == transform.shape[0]

        # Store the chain of symmetry operators and the eigenvalue and transformation matrices
        self.sym_chain = SYM_CHAIN_SLJM
        self.values = np.array(values)
        self.transform = np.array(transform)

        # List of StateSLJM objects
        self.states = [StateSLJM(v) for v in self.values]

        # List of slices for all states with different J quantum number. This is used for the calculation of
        # energy levels from perturbation hamiltonians.
        self.J_slices = []
        i = 0
        for j in range(1, len(self) + 1):
            if j == len(self) or self[j]["J2"].key != self[i]["J2"].key:
                self.J_slices.append((i, j))
                i = j

    def to_SLJ(self):
        """ Pick all stretched states with M = J and return the respective StateListSLJ object. """

        # Indices of all stretched states
        state_indices = [i for i, state in enumerate(self.states) if state["J2"].key == state["Jz"].key]

        # Indices of all symmetry operators except Jz in the symmetry chain
        sym_indices = [j for j in range(len(SYM_CHAIN_SLJM)) if SYM_CHAIN_SLJM[j] != "Jz"]

        # Extract the eigenvalues and transformation vectors of the stretched states
        values = self.values[state_indices, :][:, sym_indices]
        transform = self.transform[:, state_indices]

        # Return a StateListSLJ object of the stretched states
        return StateListSLJ(values, transform)

    def __str__(self):
        """ Return a string representation of the list of states. """

        return f"<List of {len(self)} SLJM states>"


##########################################################################
# SLJ states
##########################################################################

class StateSLJ:
    """ Class for an electron state in SLJ coupling following the chain of symmetry operators in SYM_CHAIN_SLJ. """

    def __init__(self, values):
        """ Store the eigenvalues of the state and build a dictionary containing a Symmetry object for each link in
        the chain of symmetry operators in SYM_CHAIN_SLJ. """

        self.sym_chain = SYM_CHAIN_SLJ
        assert len(values) == len(self.sym_chain)
        self.values = list(values)
        self.symmetries = dict((name, SYMMETRY[name](value)) for name, value in zip(self.sym_chain, self.values))

    def keys(self):
        """ Generate the symmetry keys of the state in the chain order. """

        for sym in self.sym_chain:
            yield sym

    def short(self):
        """ Return a short string representation of the state. """

        num = f"({self['num']})" if self["num"].key > 0 else ""
        return f"{self['S2']}{self['L2']}{self['J2']}{num}"

    def long(self):
        """ Return a long string representation of the state. """

        tau = f"{self['tau']}" if self["tau"].value > 0 else ""
        num = f"({self['num']})" if self["num"].key > 0 else ""
        return f"{self['S2']}{self['L2']} {self['GR/7']} {self['GG/2']} {self['J2']}{tau}{num}"

    def __getitem__(self, sym_name: str):
        """ Return the Symmetry object for the given operator name. """

        if sym_name not in self.symmetries:
            raise KeyError(f"Unknown symmetry {sym_name}!")
        return self.symmetries[sym_name]

    def __str__(self):
        """ Return a long string representation of the state. """

        return self.long()


class StateListSLJ(StateList):
    """ Class containing a list of StateSLJ objects representing an electron configuration. """

    def __init__(self, values, transform):
        """ Store the given eigenvalue matrix, as well as the transformation matrix from the determinantal product
        state space to SLJ coupling, and build the list of StateSLJ objects. """

        assert len(values.shape) == 2 and values.shape[1] == len(SYM_CHAIN_SLJ)
        assert len(transform.shape) == 2
        assert values.shape[0] == transform.shape[1]

        # Store the chain of symmetry operators and the eigenvalue and transformation matrices
        self.sym_chain = SYM_CHAIN_SLJ
        self.values = np.array(values)
        self.transform = np.array(transform)

        # List of StateSLJ objects
        self.states = [StateSLJ(v) for v in self.values]

        # List of slices for all states with different J quantum number. This is used for the calculation of
        # energy levels from perturbation hamiltonians.
        self.J_slices = []
        i = 0
        for j in range(1, len(self) + 1):
            if j == len(self) or self[j]["J2"].key != self[i]["J2"].key:
                self.J_slices.append((i, j))
                i = j

    def to_J(self, energies, transform):
        """ Return a StateListJ object representing an intermediate coupling of the SLJ states. """

        return StateListJ(self, energies, transform)

    def __str__(self):
        """ Return a string representation of the list of states. """

        return f"<List of {len(self)} SLJ states>"


##########################################################################
# Intermediate states
##########################################################################

class StateJ:
    """ Class for an electron state in an intermediate coupling of SLJ states. """

    def __init__(self, energy, values, states):
        """ Store the energy level of the state and the vector (values) for the linear combination of the given
        SLJ states. """

        assert isinstance(energy, float)
        assert isinstance(values, np.ndarray)
        assert isinstance(states, list)
        assert len(states) == len(values)
        assert all(state["J2"].J == states[0]["J2"].J for state in states[1:])

        # Energy level of the state
        self.energy = energy

        # Linear combination vector and weight factor vector
        self.values = values
        self.weights = values * values

        # The state in intermediate coupling is a linear combination of this list of related SLJ states
        self.states = states

        # Common quantum number J of the total angular momentum of all related SLJ states
        self.J = self.states[0]["J2"].J

    def short(self):
        """ Return a short string representation of the state. """

        return self.states[np.argmax(self.weights)].short()

    def long(self, min_weight=0.0):
        """ Return a long string representation of the state. """

        indices = reversed(np.argsort(self.weights))
        return " + ".join(
            [f"{self.weights[i]:.2f} {self.states[i].short()}" for i in indices if self.weights[i] > min_weight])

    def __str__(self):
        """ Return a long string representation of the state. """

        return self.long()


class StateListJ(StateList):
    """ Class containing a list of StateJ objects representing an electron state in an intermediate coupling
    of SLJ states. """

    def __init__(self, slj_states, energies, transform):
        """ Store the given StateListSLJ object with the list of related SLJ states, the energies of all states and
        the transformation matrix representing the linear combination of the SLJ states. """

        assert isinstance(slj_states, StateListSLJ)
        assert len(transform.shape) == 2 and transform.shape[0] == transform.shape[1]
        assert len(energies) == len(slj_states) == transform.shape[0]

        # Store the SLJ states, energy levels and the transformation matrix from SLJ to intermediate coupling
        self.slj_states = slj_states
        self.energies = energies
        self.transform = transform

        # J quantum numbers of all SLJ states
        term_J = np.array([state["J2"].J for state in self.slj_states])

        # Matrix of weight factors for the SLJ components or each state in intermediate coupling
        weight = np.power(self.transform, 2)

        # J quantum number of each state in intermediate coupling is taken from its main SLJ component
        self.J = [term_J[i] for i in np.argmax(weight, axis=0)]

        # Build the list of StateJ objects
        self.states = []
        for i in range(len(self.J)):

            # Indices of all SLJ states with the same quantum number J as the current state
            indices = np.array(np.argwhere(term_J == self.J[i]).flat)

            # Combination vector and SLJ states for the linear combination of the current state
            values = self.transform[indices, i]
            slj_states = [self.slj_states[i] for i in indices]

            # Add StateJ object representing the current state in intermediate coupling
            self.states.append(StateJ(energies[i], values, slj_states))

    def __str__(self):
        """ Return a string representation of the list of states. """

        return f"<List of {len(self)} intermediate states>"


##########################################################################
# Build SLJM states
##########################################################################

class ReducedMatrixUk:
    """ SLJM matrix holding reduced matrix elements <J'||U(k)||J> of the unit tensor U(k) of rank k in the
    orbital space or NAN, if the Wigner-Eckart theorem is not applicable for the respective matrix element.
    As long as the phases of the transformation vectors from product to SLJM space are not adjusted, only the
    diagonal elements of this matrix contain reduced matrix elements with correct sign. Therefore, the
    non-diagonal elements are used to fix the signs of the transformation vectors by the function
    phase_SLJM(). """

    def __init__(self, l, get_array, transform, J, M, k: int):
        """ Calculate and store the non-zero components of the unit tensor operator U(k) in the SLJM space. Use
        the given transformation matrix and the vectors of J and M quantum numbers of every state. """

        assert 0 <= k <= 2 * l
        assert len(J) == len(M)

        # Store the J and M values of all states as well as the tensor rank
        self.J = J
        self.M = M
        self.k = k

        # Store the non-zero component of the unit tensor operator U(k) for each matrix element
        self.hyper = np.zeros((2 * k + 1, len(J), len(J)), dtype=float)
        for q in range(-k, k + 1):
            self.hyper[q + k, :, :] = transform.T @ get_array(f"U/a/{k},{q}") @ transform

    def __getitem__(self, item):
        """ Use the Wigner-Eckart theorem to calculate the reduced matrix element from <J'M'|U(k)_q|JM>. """

        i, j = item
        Ja = self.J[i]
        Jb = self.J[j]
        Ma = self.M[i]
        Mb = self.M[j]
        q = Ma - Mb

        # Wigner-Eckart theorem is only applicable if the respective tensor component q = M'-M is not zero
        if q < -self.k or q > self.k:
            return np.nan
        result = self.hyper[q + self.k, i, j]
        if result == 0.0:
            return np.nan

        # Wigner-Eckart theorem is only applicable if the Wigner3j factor is not zero
        factor = wigner3j(Ja, self.k, Jb, -Ma, q, Mb)
        if factor == 0.0:
            return np.nan

        # Return a valid reduced matrix element <J'||U(k)||J> of the unit tensor operator U(k)
        if (Ja - Ma) % 2 != 0:
            factor = -factor
        return result / factor

    def sub_matrix(self, i_min, i_max):
        """ Return sub-matrix on the diagonal for the investigation of a J sub-space. """

        n = i_max - i_min
        elements = [self[i, j] for i in range(i_min, i_max) for j in range(i_min, i_max)]
        return np.array(elements, dtype=float).reshape(n, n)


def phase_SLJM(l, get_array, values, transform):
    """ Adjust the signs of the transformation vectors from product to SLJM space. This adjustment allows to calculate
    reduced matrix elements in the SLJ space. """

    assert isinstance(values, np.ndarray)
    assert isinstance(transform, np.ndarray)
    assert len(values.shape) == 2
    assert len(transform.shape) == 2
    assert values.shape[1] == len(SYM_CHAIN_SLJM)
    assert values.shape[0] == transform.shape[0] == transform.shape[1]

    # Specific keys for J-subspaces
    keys = val2key(values, SYM_CHAIN_SLJM)
    sym_indices = [SYM_CHAIN_SLJM.index(name) for name in ("S2", "GR/7", "GG/2", "L2", "tau", "J2")]
    state_keys = [" ".join(map(str, row)) for row in keys[:, sym_indices]]
    num_states = len(state_keys)

    # No phase fixed yet
    unknown = np.ones(num_states, dtype=bool)

    # Signs of column vectors
    signs = np.ones(num_states, dtype=int)

    # First index of first J space
    i = 0

    # Find first index of next J space
    slices = []
    for j in range(num_states + 1):
        if j < num_states and state_keys[j] == state_keys[i]:
            continue

        # Stretched state used as reference
        unknown[i] = False

        # Index slice of the current J space
        if j - i > 1:
            slices.append((i, j))

        # First index of next J space
        i = j

    # Get J and M quantum number of all states
    J = [s.J for s in SymmetryList(values[:, SYM_CHAIN_SLJM.index("J2")], "J2")]
    M = [s.M for s in SymmetryList(values[:, SYM_CHAIN_SLJM.index("Jz")], "Jz")]

    # Increase unit tensor rank until all phases are fixed. Rank 0 is completely diagonal. We thus start with k=1.
    for k in range(1, 2 * l + 1):
        if not np.any(unknown):
            break

        # Prepare unit tensor matrix with Wigner-Eckart theorem applied
        # print(f"unit({k}): {sum(unknown)}/{num_states}")
        reduced_Uk = ReducedMatrixUk(l, get_array, transform, J, M, k)

        # Use every J space with unknown phases
        for i_min, i_max in slices:
            if not np.any(unknown[i_min:i_max]):
                continue

            # Get the diagonal sub-matrix containing the reduced matrix element <J||U(k)||J> calculated from each
            # <JM'|U(k)_q|JM> element in the current J sub-space. Only the diagonal elements always show the correct
            # sign. A NAN value indicates that this <JM'|JM> element is not suitable to calculate the reduced matrix
            # element. All valid elements share the same absolute value.
            sub_matrix = reduced_Uk.sub_matrix(i_min, i_max)

            # Stretched state is reference. Use other rank k, if it is not valid. This actually never happens. There is
            # just one special case for the Nd3+ state | 4S (111) [00] 3/2 > with almost zero matrix elements in all
            # <JM'|U(k)_q|JM> tensor components for every rank k.
            ref_element = sub_matrix[0, 0]
            if np.isnan(ref_element):
                continue

            # Find column with unknown phase
            for col in range(i_min + 1, i_max):
                if not unknown[col]:
                    continue

                # Find a valid element in the column
                for row in range(i_min, col):
                    element = sub_matrix[col - i_min, row - i_min]
                    if np.isnan(element):
                        continue

                    # Fix phase of this column vector. Magnitude of reduced matrix elements should be independent
                    # of M. An exception is made for very small elements (just | 4S (111) [00] 3/2 > in Nd3+).
                    sign = element * signs[row] / ref_element
                    assert abs(ref_element) < 1e-25 or (abs(sign) - 1) < 1e-5
                    if abs(ref_element) < 1e-25 or (abs(sign) - 1) >= 1e-5:
                        state = StateSLJ(values[col, :-1])
                        str_element = f"element({row},{col}) = {element}"
                        str_ref = f"ref({i_min},{i_min}) = {ref_element}"
                        print(f"Warning (phase_SLJM) for state | {state} >: {str_element} / {str_ref} / sign = {sign}")
                    if sign < 0:
                        signs[col] = -1
                    unknown[col] = False
                    break

    # Correct all column vectors
    assert not np.any(unknown)
    # print(f"{np.sum(signs < 0)} signs flipped.")
    return transform * signs


def build_tau(syms, states):
    """ The lanthanide configurations f5 - f9 contain pairs of SLJM states, which match in all quantum numbers
    in the chain of operators S2, G(R7), G(G2), L2, and J2. These states get an artificial tau value of 1 and 2
    assigned ad-hoc. The tau value of unique states is 0. Return an SymmetryList object containing the tau values
    of all states. """

    # Build a dictionary which maps every distinct state to the respective state indices
    names = {}
    for i in range(states):
        key = f"{syms['S2'][i]} {syms['GR/7'][i]} {syms['GG/2'][i]} {syms['L2'][i]} {syms['J2'][i]} {syms['Jz'][i]}"
        if key not in names:
            names[key] = [i]
        else:
            names[key].append(i)

    # Assign tau values 1 and 2 to matching pairs and 0 to states with unique quantum numbers
    tau_values = states * [0]
    for key in names:
        if len(names[key]) > 1:
            if len(names[key]) > 2:
                raise RuntimeError("More than 2 equal SLJM states found!")
            for num, i in enumerate(names[key]):
                tau_values[i] = num + 1

    # Return an SymmetryList object containing the tau values of all states
    return SymmetryList(tau_values, "tau")


def build_num(syms, states):
    """ Assign individual integer numbers to states, which match in the quantum numbers for the operators
    S2, L2, and J2 as a short-cut. The num value of unique states is 0. Return an SymmetryList object containing
    the num values of all states."""

    # Build a dictionary which maps every distinct quantum number combination S, L, J to the respective state indices
    names = {}
    for i in range(states):

        # This is the reference SLJ key
        key1 = f"{syms['S2'][i]} {syms['L2'][i]} {syms['J2'][i]}"
        if key1 not in names:
            names[key1] = {}
            names[key1]["keys"] = []

        # This key distinguishes the SLJ states with the same SLJ values
        key2 = f"{syms['GR/7'][i]} {syms['GG/2'][i]} {syms['tau'][i]}"
        if key2 not in names[key1]:
            names[key1][key2] = [i]
            names[key1]["keys"].append(key2)
        else:
            names[key1][key2].append(i)

    # Assign integer num values to matching states and 0 to states with unique quantum numbers
    num_values = states * [0]
    for key1 in names:
        if len(names[key1]) > 2:
            num_value = 1
            for key2 in names[key1]["keys"]:
                for i in names[key1][key2]:
                    num_values[i] = num_value
                num_value += 1

    # Return an SymmetryList object containing the num values of all states
    return SymmetryList(num_values, "num")


def sort_states(values: np.ndarray, transform: np.ndarray, sym_order: tuple) -> (np.ndarray, np.ndarray):
    """ Lexicographical sort of states based on their given eigenvalue matrix according to the chain of symmetry
    groups given in SYM_CHAIN_SLJM. Return new eigenvalue and transformation matrices corresponding to the new order
    of states. """

    # Sorting is based on the integer keys. The eigenvalue matrix
    keys = val2key(values, SYM_CHAIN_SLJM)

    # Map given symmetry chain to SYM_CHAIN_SLJM
    sym_indices = [SYM_CHAIN_SLJM.index(name) for name in sym_order]

    # Determine state indices resembling lexicographically sorted integer eigenvalue keys
    state_indices = np.lexsort(keys[:, sym_indices].T)

    # Use sorted state indices to build new eigenvalue and transformation matrices
    if np.any(state_indices != np.arange(state_indices.shape[0])):
        values = values[state_indices, :]
        transform = transform[:, state_indices]

    # Return new eigenvalue and transformation matrices
    return values, transform


def build_SLJM(l: int, num_states: int, get_array):
    """ Build the transformation matrix from the determinantal product state space to the SLJM space and the matrix
    of all eigenvalues of the symmetry operators in the classification chain. The first argument is the number of
    electron states and the second is a function used to get the matrix of a symmetry operator of given name in
    determinantal product state coupling. """

    # Initialize the eigenvalue matrix, the transformation matrix, and the dictionary of SymmetryList objects
    eigen_vectors = np.zeros((num_states, num_states), dtype=float)
    transform = None
    symmetries = {}

    # Initialize the list of sub-spaces which will be split by the algorithm
    sym_slices = [slice(0, num_states)]

    # Follow the chain or symmetry operators, but skip the pseudo operators "tau" and "num". Build a transformation
    # matrix from product states to SLJM coupling together with the eigenvalues of all symmetry operators.
    for name in SYM_CHAIN_SLJM:
        if name in ("tau", "num"):
            continue

        # Get the matrix of the current symmetry operator in the determinantal product space and apply the current
        # transformation matrix
        array = get_array(name)
        if transform is not None:
            array = transform.T @ array @ transform
        assert isinstance(array, np.ndarray)

        # Initialize eigenvalues and eigenvectors of the current symmetry operator
        eigen_values = []
        eigen_vectors *= 0.0

        # Calculate eigenvalues and eigenvectors of the current symmetry operator by diagonalising its pre-transformed
        # matrix in each of the current sub-spaces
        for sym_slice in sym_slices:
            if sym_slice.stop - sym_slice.start > 1:
                V, U = np.linalg.eigh(array[sym_slice, sym_slice])
            elif sym_slice.stop - sym_slice.start == 1:
                V = array[sym_slice, sym_slice].reshape((1,))
                U = np.ones((1, 1), float)
            else:
                raise RuntimeError("Empty slice!")
            eigen_values += list(V)
            eigen_vectors[sym_slice, sym_slice] = U

        # Store SymmetryList object containing the eigenvalues of the current symmetry operator for all states
        symmetries[name] = SymmetryList(eigen_values, name)

        # Split the sub-spaces in such a way that all states inside the new sub-spaces have the same eigenvalue of
        # the current symmetry operator
        sym_slices = symmetries[name].split_syms(sym_slices)

        # Use the eigenvectors of the current symmetry operator to update the transformation matrix
        if transform is None:
            transform = np.array(eigen_vectors)
        else:
            transform = transform @ eigen_vectors

    # Label the state pairs which cannot be resolved by our classification with the chain of symmetry operators
    symmetries["tau"] = build_tau(symmetries, num_states)

    # Generate short-cut numbers to distinguish states sharing the same SLJ quantum numbers
    symmetries["num"] = build_num(symmetries, num_states)

    # Build a matrix of eigenvalues (and pseudo eigenvalues "tau" and "num"). Rows correspond to SLJM states,
    # columns to operators from SYM_CHAIN_SLJM
    values = np.zeros((num_states, len(SYM_CHAIN_SLJM)), dtype=float)
    for j, name in enumerate(SYM_CHAIN_SLJM):
        eigen_values = symmetries[name]
        for i in range(len(eigen_values)):
            values[i, j] = eigen_values[i].value

    # Adjust phases of the eigenvectors in the transformation matrix. This would not be necessary for the calculation
    # of ordinary matrix elements of tensor operators. However, it is the precondition for the calculation of
    # reduced matrix elements in SLJ coupling.
    sym_order = ("Jz", "J2", "tau", "L2", "GG/2", "GR/7", "S2")
    values, transform = sort_states(values, transform, sym_order)
    transform = phase_SLJM(l, get_array, values, transform)

    # Sort the SLJM states is such a way that all states with the same J quantum number form contiguous groups.
    # In this the matrices of the perturbation hamiltonians can be split into a chain of sub-matrices along the
    # main diagonal. Diagonalisation of these sub-matrices speeds the calculation of energy levels up significantly.
    sym_order = ("Jz", "tau", "L2", "GG/2", "GR/7", "S2", "J2")
    values, transform = sort_states(values, transform, sym_order)

    # Return the eigenvalue and transformation matrices
    return values, transform


##########################################################################
# HDF5 cache interface
##########################################################################

def init_states(vault, group_name, ion):
    """ Initialize the cache for the storage of eigenvalue and transformation matrices for the transformation from
    the determinantal product state space to SLJM coupling in the HDF5 group with given name in the given HDF5
    file vault. Return a dictionary containing StateList objects for product, SLJM, and SLJ coupling. """

    # No file cache
    if not vault:
        num_states = len(ion.product)
        get_array = lambda name: ion.matrix(name).array
        values, transform = build_SLJM(ion.l, num_states, get_array)
        group = {"values": values, "transform": transform}

    # Use file cache
    else:

        # Delete the group in the HDF5 file, if the cache is marked as invalid or its version number does not match
        if group_name in vault:
            if not vault.attrs["valid"] or "version" not in vault[group_name].attrs \
                    or vault[group_name].attrs["version"] != TERM_VERSION:
                del vault[group_name]
                vault.flush()

        # Generate all data, if the HDF5 group is missing
        if group_name not in vault:
            print("Create SLJM states ...")

            # Render all cache structures following in the dependence chain as invalid
            vault.attrs["valid"] = False

            # Create new HDF5 group and store the current version number
            vault.create_group(group_name)
            vault[group_name].attrs["version"] = TERM_VERSION

            # Build the transformation matrix from the determinantal product state space to the SLJM space and the
            # matrix of all eigenvalues of the symmetry operators in the classification chain.
            num_states = len(ion.product)
            get_array = lambda name: ion.matrix(name).array
            values, transform = build_SLJM(ion.l, num_states, get_array)

            # Store eigenvalue and transformation matrices
            group = vault[group_name].create_group(Coupling.SLJM.name)
            group.create_dataset("values", data=values, compression="gzip", compression_opts=9)
            group.create_dataset("transform", data=transform, compression="gzip", compression_opts=9)

            # Flush the cache file
            vault.flush()
            print("SLJM states done.")

        # HDF5 state group
        else:
            group = vault[group_name][Coupling.SLJM.name]

    # StateList object for the determinantal product states
    Product_States = StateListProduct(ion.product)

    # The StateList object for SLJM states is taken from the file cache
    values = np.array(group["values"])
    transform = np.array(group["transform"])
    SLJM_states = StateListSLJM(values, transform)

    # StateList object for SLJ states
    SLJ_states = SLJM_states.to_SLJ()

    # Return StateList dictionary
    return {
        Coupling.Product.name: Product_States,
        Coupling.SLJM.name: SLJM_states,
        Coupling.SLJ.name: SLJ_states,
    }
