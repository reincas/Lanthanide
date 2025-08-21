##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import numpy as np
from enum import Enum

from .wigner import wigner3j
from .unit import ORBITAL, SPIN, MAGNETIC
from .symmetry import CHR_ORBITAL, SYMMETRY, SymmetryList

TERM_VERSION = 2
SYM_CHAIN_SLJM = ("S2", "GR/7", "GG/2", "L2", "tau", "J2", "num", "Jz")
SYM_CHAIN_SLJ = ("S2", "GR/7", "GG/2", "L2", "tau", "J2", "num")


class Coupling(Enum):
    Product = 0
    SLJM = 1
    SLJ = 2
    J = 3


##################################################
# Product states
##################################################

class StateProduct:
    def __init__(self, values):
        self.l = ORBITAL
        self.s = SPIN
        self.m = MAGNETIC
        self.values = list(values)
        self.quantum = [(self.l, self.m[i][0], self.s, self.m[i][1]) for i in self.values]

    def short(self):
        quantum = [(ml, "du"[(2 * ms + 1) // 2]) for l, ml, s, ms in self.quantum]
        return " ".join(f"{ml:+d}{ms}" for ml, ms in quantum)

    def long(self):
        quantum = [(CHR_ORBITAL[l].lower(), ml, 2 * s, 2 * ms) for l, ml, s, ms in self.quantum]
        return " ".join(f"{l},{ml:+d},{s}/2,{ms:+d}/2" for l, ml, s, ms in quantum)

    def __getitem__(self, item):
        return self.quantum[item]

    def __str__(self):
        return self.long()


class StateListProduct:
    def __init__(self, values):
        assert len(set(len(state) for state in values)) == 1

        self.values = np.array(values)
        self.transform = None
        self.states = [StateProduct(v) for v in values]

    def __len__(self):
        return len(self.states)

    def __iter__(self):
        for state in self.states:
            yield state

    def __getitem__(self, item):
        return self.states[item]

    def __str__(self):
        return f"<List of {len(self)} product states>"


##################################################
# SLJM states
##################################################

class StateSLJM:
    def __init__(self, values):
        self.sym_chain = SYM_CHAIN_SLJM
        assert len(values) == len(self.sym_chain)
        self.values = list(values)
        self.symmetries = dict((name, SYMMETRY[name](value)) for name, value in zip(SYM_CHAIN_SLJM, self.values))

    def short(self):
        num = f"({self['num']})" if self["num"].key > 0 else ""
        return f"{self['S2']}{self['L2']}{self['J2']}{num} {self['Jz']}"

    def long(self):
        tau = f"{self['tau']}" if self["tau"].value > 0 else ""
        num = f"({self['num']})" if self["num"].key > 0 else ""
        return f"{self['S2']}{self['L2']} {self['GR/7']} {self['GG/2']} {self['J2']}{tau}{num} {self['Jz']}"

    # def key(self, sym_names=None):
    #     if sym_names is None:
    #         sym_names = self.sym_chain
    #     return " ".join([str(self[sym]) for sym in sym_names])

    def __getitem__(self, key):
        if key not in self.symmetries:
            raise KeyError(f"Unknown symmetry {key}!")
        return self.symmetries[key]

    def __str__(self):
        return self.long()


class StateListSLJM:
    def __init__(self, values, transform):
        assert len(values.shape) == 2 and values.shape[1] == len(SYM_CHAIN_SLJM)
        assert len(transform.shape) == 2 and transform.shape[0] == transform.shape[1]
        assert values.shape[0] == transform.shape[0]

        self.sym_chain = SYM_CHAIN_SLJM
        self.values = np.array(values)
        self.transform = np.array(transform)
        self.states = [StateSLJM(v) for v in self.values]

        self.J_slices = []
        i = 0
        for j in range(1, len(self) + 1):
            if j == len(self) or self[j]["J2"].key != self[i]["J2"].key:
                self.J_slices.append((i, j))
                i = j

    def to_SLJ(self):
        state_indices = [i for i, state in enumerate(self.states) if state["J2"].key == state["Jz"].key]
        sym_indices = [j for j in range(len(SYM_CHAIN_SLJM)) if SYM_CHAIN_SLJM[j] != "Jz"]
        values = self.values[state_indices, :][:, sym_indices]
        transform = self.transform[:, state_indices]
        return StateListSLJ(values, transform)

    def __len__(self):
        return len(self.states)

    def __iter__(self):
        for state in self.states:
            yield state

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.states[item]
        values = self.values[:, SYM_CHAIN_SLJM.index(item)]
        return SymmetryList(values, item)

    def __str__(self):
        return f"<List of {len(self)} SLJM states>"


def val2key(values, sym_chain=SYM_CHAIN_SLJM):
    """ Convert float symmetry value array to int symmetry key array. """

    assert isinstance(values, np.ndarray)
    assert len(values.shape) == 2
    assert values.dtype == float
    assert values.shape[1] == len(sym_chain)

    return np.array([SymmetryList(values[:, i], name).keys for i, name in enumerate(SYM_CHAIN_SLJM)], dtype=int).T


##################################################
# SLJ states
##################################################

class StateSLJ:
    def __init__(self, values):
        self.sym_chain = SYM_CHAIN_SLJ
        assert len(values) == len(self.sym_chain)
        self.values = list(values)
        self.symmetries = dict((name, SYMMETRY[name](value)) for name, value in zip(self.sym_chain, self.values))

    def short(self):
        num = f"({self['num']})" if self["num"].key > 0 else ""
        return f"{self['S2']}{self['L2']}{self['J2']}{num}"

    def long(self):
        tau = f"{self['tau']}" if self["tau"].value > 0 else ""
        num = f"({self['num']})" if self["num"].key > 0 else ""
        return f"{self['S2']}{self['L2']} {self['GR/7']} {self['GG/2']} {self['J2']}{tau}{num}"

    def __getitem__(self, key):
        if key not in self.symmetries:
            raise KeyError(f"Unknown symmetry {key}!")
        return self.symmetries[key]

    def __str__(self):
        return self.long()


class StateListSLJ:
    def __init__(self, values, transform):
        self.sym_chain = SYM_CHAIN_SLJ
        assert len(values.shape) == 2 and values.shape[1] == len(self.sym_chain)
        assert len(transform.shape) == 2
        assert values.shape[0] == transform.shape[1]

        self.values = np.array(values)
        self.transform = np.array(transform)
        self.states = [StateSLJ(v) for v in self.values]

        self.J_slices = []
        i = 0
        for j in range(1, len(self) + 1):
            if j == len(self) or self[j]["J2"].key != self[i]["J2"].key:
                self.J_slices.append((i, j))
                i = j

    def to_J(self, energies, transform):
        return StateListJ(self, energies, transform)

    def __len__(self):
        return len(self.states)

    def __iter__(self):
        for state in self.states:
            yield state

    def __getitem__(self, item):
        return self.states[item]

    def __str__(self):
        return f"<List of {len(self)} SLJ states>"

##################################################
# Intermediate states
##################################################

class StateJ:
    def __init__(self, energy, values, states):
        assert isinstance(energy, float)
        assert isinstance(values, np.ndarray)
        assert isinstance(states, list)
        assert len(states) == len(values)

        self.energy = energy
        self.values = values
        self.weights = values * values
        self.states = states
        self.J = self.states[0]["J2"].J

    def short(self):
        return self.states[np.argmax(self.weights)].short()

    def long(self, min_weight=0.0):
        indices = reversed(np.argsort(self.weights))
        return " + ".join([f"{self.weights[i]:.2f} {self.states[i].short()}" for i in indices if self.weights[i] > min_weight])

    def __str__(self):
        return self.long()

class StateListJ:
    def __init__(self, slj_states, energies, transform):
        assert isinstance(slj_states, StateListSLJ)
        assert len(transform.shape) == 2 and transform.shape[0] == transform.shape[1]
        assert len(energies) == len(slj_states) == transform.shape[0]

        self.slj_states = slj_states
        self.energies = energies
        self.transform = transform

        term_J = np.array([state["J2"].J for state in self.slj_states])
        weight = np.power(self.transform, 2)
        self.J = np.array([term_J[i] for i in np.argmax(weight, axis=0)], dtype=int)
        self.states = []
        for i in range(len(self.J)):
            indices = np.array(np.argwhere(term_J == self.J[i]).flat)
            values = self.transform[indices, i]
            slj_states = [self.slj_states[i] for i in indices]
            self.states.append(StateJ(energies[i], values, slj_states))

    def __len__(self):
        return len(self.states)

    def __iter__(self):
        for state in self.states:
            yield state

    def __getitem__(self, item):
        return self.states[item]

    def __str__(self):
        return f"<List of {len(self)} intermediate states>"


##################################################
# Build SLJM states
##################################################

class ReducedMatrixUk:
    """ SLJM matrix holding reduced matrix elements <J'||U(k)||J> of the unit tensor U(k) of rank k in the
    orbital space or NAN, if the Wigner-Eckart theorem is not applicable for the respective matrix element.
    As long as the phases of the transformation vectors from product to SLJM space are not adjusted, only the
    diagonal elements of this matrix contain reduced matrix elements with correct sign. The non-diagonal
    elements may thus be used to fix the signs of the transformation vectors. """

    def __init__(self, ion, transform, J, M, k: int):
        assert 0 <= k <= 2 * ion.l
        assert len(J) == len(M)

        # Store the J and M values of all states as well as the tensor rank
        self.J = J
        self.M = M
        self.k = k

        # Store all components of the unit tensor operator U(k)
        self.hyper = np.zeros((2 * k + 1, len(J), len(J)), dtype=float)
        for q in range(-k, k + 1):
            self.hyper[q + k, :, :] = transform.T @ ion.matrix(f"U/a/{k},{q}").array @ transform

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


def phase_SLJM(ion, values, transform):
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
    for k in range(1, 2 * ion.l + 1):
        if not np.any(unknown):
            break

        # Prepare unit tensor matrix with Wigner-Eckart theorem applied
        # print(f"unit({k}): {sum(unknown)}/{num_states}")
        reduced_Uk = ReducedMatrixUk(ion, transform, J, M, k)

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
                        state = StateSLJ(values[col,:-1])
                        print(f"Warning (phase_SLJM) for state | {state} >: element({row},{col}) = {element} / ref({i_min},{i_min}) = {ref_element} / sign = {sign}")
                    if sign < 0:
                        signs[col] = -1
                    unknown[col] = False
                    break

    # Correct all column vectors
    assert not np.any(unknown)
    #print(f"{np.sum(signs < 0)} signs flipped.")
    return transform * signs


def build_tau(syms, states):
    names = {}
    for i in range(states):
        key = f"{syms['S2'][i]} {syms['GR/7'][i]} {syms['GG/2'][i]} {syms['L2'][i]} {syms['J2'][i]} {syms['Jz'][i]}"
        if key not in names:
            names[key] = [i]
        else:
            names[key].append(i)

    tau_values = states * [0]
    for key in names:
        if len(names[key]) > 1:
            if len(names[key]) > 2:
                raise RuntimeError("More than 2 equal SLJM states found!")
            for num, i in enumerate(names[key]):
                tau_values[i] = num + 1
    return SymmetryList(tau_values, "tau")


def build_num(syms, states):
    names = {}
    for i in range(states):
        key1 = f"{syms['S2'][i]} {syms['L2'][i]} {syms['J2'][i]}"
        if key1 not in names:
            names[key1] = {}
            names[key1]["keys"] = []

        key2 = f"{syms['GR/7'][i]} {syms['GG/2'][i]} {syms['tau'][i]}"
        if key2 not in names[key1]:
            names[key1][key2] = [i]
            names[key1]["keys"].append(key2)
        else:
            names[key1][key2].append(i)

    num_values = states * [0]
    for key1 in names:
        if len(names[key1]) > 2:
            num_value = 1
            for key2 in names[key1]["keys"]:
                for i in names[key1][key2]:
                    num_values[i] = num_value
                num_value += 1
    return SymmetryList(num_values, "num")


def sort_states(values: np.ndarray, transform: np.ndarray, sym_order: tuple) -> (np.ndarray, np.ndarray):
    """ Lexicographical sort of states based on integer symmetry values. """

    keys = val2key(values, SYM_CHAIN_SLJM)
    sym_indices = [SYM_CHAIN_SLJM.index(name) for name in sym_order]
    state_indices = np.lexsort(keys[:, sym_indices].T)
    if np.any(state_indices != np.arange(state_indices.shape[0])):
        values = values[state_indices, :]
        transform = transform[:, state_indices]

    return values, transform


def build_SLJM(ion):
    print("Create SLJM states ... ", end="")
    states = len(ion.product_states)
    eigen_vectors = np.zeros((states, states), dtype=float)
    transform = None
    symmetries = {}

    sym_slices = [slice(0, states)]
    for name in SYM_CHAIN_SLJM:
        if name in ("tau", "num"):
            continue
        print(f"{name} ... ", end="")

        array = ion.matrix(name).array
        if transform is not None:
            array = transform.T @ array @ transform

        eigen_values = []
        eigen_vectors *= 0.0
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

        eigen_values = SymmetryList(eigen_values, name)
        sym_slices = eigen_values.split_syms(sym_slices)
        symmetries[name] = eigen_values

        if transform is None:
            transform = np.array(eigen_vectors)
        else:
            transform = transform @ eigen_vectors

    print("tau ... ", end="")
    symmetries["tau"] = build_tau(symmetries, states)
    print("num ... ", end="")
    symmetries["num"] = build_num(symmetries, states)

    values = np.zeros((states, len(SYM_CHAIN_SLJM)), dtype=float)
    for j, name in enumerate(SYM_CHAIN_SLJM):
        eigen_values = symmetries[name]
        for i in range(len(eigen_values)):
            values[i, j] = eigen_values[i].value

    print("phase ... ", end="")
    # Adjust phases for reduced matrix elements from SLJ states
    sym_order = ("Jz", "J2", "tau", "L2", "GG/2", "GR/7", "S2")
    values, transform = sort_states(values, transform, sym_order)
    transform = phase_SLJM(ion, values, transform)

    # Sort for J spaces
    sym_order = ("Jz", "tau", "L2", "GG/2", "GR/7", "S2", "J2")
    values, transform = sort_states(values, transform, sym_order)

    print("done.")
    return values, transform


##################################################
# HDF5 interface
##################################################

def init_states(vault, group_name, ion):
    if group_name in vault:
        if not vault.attrs["valid"] or "version" not in vault[group_name].attrs or vault[group_name].attrs["version"] != TERM_VERSION:
            del vault[group_name]
            vault.flush()

    if group_name not in vault:
        vault.attrs["valid"] = False
        vault.create_group(group_name)
        vault[group_name].attrs["version"] = TERM_VERSION

        values, transform = build_SLJM(ion)

        group = vault[group_name].create_group(Coupling.SLJM.name)
        group.create_dataset("values", data=values, compression="gzip", compression_opts=9)
        group.create_dataset("transform", data=transform, compression="gzip", compression_opts=9)
        vault.flush()

    Product_States = StateListProduct(ion.product_states)

    group = vault[group_name][Coupling.SLJM.name]
    values = np.array(group["values"])
    transform = np.array(group["transform"])
    SLJM_states = StateListSLJM(values, transform)

    SLJ_states = SLJM_states.to_SLJ()

    return {
        Coupling.Product.name: Product_States,
        Coupling.SLJM.name: SLJM_states,
        Coupling.SLJ.name: SLJ_states,
    }


if __name__ == "__main__":
    import time
    from lanthanide import Lanthanide

    for num in range(1, 14):
        with Lanthanide(num) as ion:
            print(ion)
            states = ion.states(Coupling.SLJM)
            new_transform = phase_SLJM(ion, states.values, states.transform)

    print("Done.")
