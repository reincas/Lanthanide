##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import math

from .halfint import HalfInt

CHR_ORBITAL = "SPDFGHIKLMNOQRTUVWXYZ"


def casimir_Rk(w):
    """ """
    sum = 0
    for i in range(len(w)):
        sum += w[i] * (w[i] - 1 + 2 * len(w) - 2 * i)
    return sum // 2


def casimir_G2(u):
    return u[0] * u[0] + u[1] * u[1] + u[0] * u[1] + 5 * u[0] + 4 * u[1]


def casimir_dict():
    casimir = {}

    R5 = {}
    for i in range(3):
        for j in range(i + 1):
            w = (i, j)
            R5[casimir_Rk(w)] = w

    R7 = {}
    for i in range(3):
        for j in range(i + 1):
            for k in range(j + 1):
                w = (i, j, k)
                R7[casimir_Rk(w)] = w

    G2 = {}
    for i in range(5):
        for j in range(i + 1):
            u = (i, j)
            G2[casimir_G2(u)] = u

    return {"GR/5": R5, "GR/7": R7, "GG/2": G2}


CASIMIR = casimir_dict()


class Symmetry:
    value: float
    name: str
    key: int
    symbol: str
    str_value: str

    def _to_int_(self, value):
        key = round(value)
        if abs(value - key) > 1e-5:
            raise ValueError(f"{value} is not a legal eigenvalue of operator {self.name}!")
        return key

    def __str__(self):
        return self.str_value

    def __lt__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return self.key < other.key

    def __le__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return self.key <= other.key

    def __gt__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return self.key > other.key

    def __ge__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return self.key >= other.key

    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return self.key == other.key

    def __ne__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return self.key != other.key


class SymmetryS2(Symmetry):
    name = "S2"
    symbol = "2S+1"

    def __init__(self, value):
        self.value = value
        self.key = self._to_int_(math.sqrt(4 * self.value + 1))
        self.str_value = (str(self.key))


class SymmetryGR7(Symmetry):
    name = "GR/7"
    symbol = "W"

    def __init__(self, value):
        self.value = value
        self.key = self._to_int_(5 * self.value)
        self.str_value = "(%d%d%d)" % CASIMIR["GR/7"][self.key]


class SymmetryGR5(Symmetry):
    name = "GR/5"
    symbol = "W"

    def __init__(self, value):
        self.value = value
        self.key = self._to_int_(3 * self.value)
        self.str_value = "(%d%d)" % CASIMIR["GR/5"][self.key]


class SymmetryGG2(Symmetry):
    name = "GG/2"
    symbol = "U"

    def __init__(self, value):
        self.value = value
        self.key = self._to_int_(12 * self.value)
        self.str_value = "[%d%d]" % CASIMIR["GG/2"][self.key]


class SymmetryL2(Symmetry):
    name = "L2"
    symbol = "L"

    def __init__(self, value):
        self.value = value
        self.key = self._to_int_((math.sqrt(4 * self.value + 1) - 1) / 2)
        self.str_value = CHR_ORBITAL[self.key]


class SymmetryJ2(Symmetry):
    name = "J2"
    symbol = "J"

    def __init__(self, value):
        self.value = value
        self.key = self._to_int_(math.sqrt(4 * self.value + 1) - 1)
        self.str_value = f"{self.key // 2}" if self.key % 2 == 0 else f"{self.key}/2"

        if self.key % 2 == 0:
            self.J = self.key // 2
        else:
            self.J = HalfInt(self.key)


class SymmetryJz(Symmetry):
    name = "Jz"
    symbol = "M"

    def __init__(self, value):
        self.value = value
        self.key = self._to_int_(2 * self.value)
        self.str_value = f"{(self.key // 2):+d}" if self.key % 2 == 0 else f"{self.key:+d}/2"

        if self.key % 2 == 0:
            self.M = self.key // 2
        else:
            self.M = HalfInt(self.key)

class SymmetryTau(Symmetry):
    name = "tau"
    symbol = "𝜏"

    def __init__(self, value):
        self.value = value
        self.key = self._to_int_(self.value)
        self.str_value = "*ab"[self.key]


class SymmetryNum(Symmetry):
    name = "num"
    symbol = "#"

    def __init__(self, value):
        self.value = value
        self.key = self._to_int_(self.value)
        self.str_value = str(self.key)


SYMMETRY = {
    "S2": SymmetryS2,
    "GR/7": SymmetryGR7,
    "GR/5": SymmetryGR5,
    "GG/2": SymmetryGG2,
    "L2": SymmetryL2,
    "J2": SymmetryJ2,
    "Jz": SymmetryJz,
    "tau": SymmetryTau,
    "num": SymmetryNum,
}


class SymmetryList:

    def __init__(self, values, name):
        if not name in SYMMETRY:
            raise ValueError(f"Unknown operator {name}")

        self.name = name
        self.values = list(values)
        self.object_class = SYMMETRY[name]
        self.objects = [self.object_class(value) for value in values]
        self.keys = [sym.key for sym in self.objects]

    def count(self):
        keys = set([obj.key for obj in self.objects])
        return dict([(key, self.count_key(key)) for key in keys])

    def find(self, key):
        for obj in self.objects:
            if obj.key == key:
                return obj
        raise ValueError(f"Unknown symmetry key {key}!")

    def count_key(self, key: int) -> int:
        return len([obj for obj in self.objects if obj.key == key])

    def split_syms(self, last_slices=None):
        if not last_slices:
            last_slices = [slice(0, len(self))]
        assert last_slices[0].start == 0 and last_slices[-1].stop == len(self)
        for i in range(1, len(last_slices)):
            assert last_slices[i - 1].stop == last_slices[i].start

        new_slices = []
        i = 0
        for last_slice in last_slices:
            for j in range(last_slice.start + 1, last_slice.stop + 1):
                if j == last_slice.stop or self.objects[j].key != self.objects[i].key:
                    new_slices.append(slice(i, j))
                    i = j
            i = last_slice.stop
        return new_slices

    def __getitem__(self, item):
        return self.objects[item]

    def __add__(self, other):
        if not other:
            return self
        if not isinstance(other, SymmetryList):
            return NotImplemented
        if self.name != other.name:
            raise ValueError(f"Cannot add lists of different symmetries!")
        return SymmetryList(self.values + other.values, self.name)

    def __radd__(self, other):
        return self.__add__(other)

    def __len__(self):
        return len(self.objects)

    def __str__(self):
        count = self.count()
        return ", ".join(f"{self.find(key)}: {count[key]}" for key in sorted(count.keys()))
