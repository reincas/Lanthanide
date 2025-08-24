##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

from itertools import combinations
from functools import reduce

SINGLE_VERSION = 0
BIT_LENGTH = 4
INDEX_MASK = (1 << BIT_LENGTH) - 1
INITIAL_SHIFT = [(1 * BIT_LENGTH,), (3 * BIT_LENGTH, 2 * BIT_LENGTH), (5 * BIT_LENGTH, 4 * BIT_LENGTH, 3 * BIT_LENGTH)]
FINAL_SHIFT = [(0,), (1 * BIT_LENGTH, 0), (2 * BIT_LENGTH, 1 * BIT_LENGTH, 0)]
PART_MASK = [(1 << i * BIT_LENGTH) - 1 for i in range(1, 4)]


def index_key(indices: tuple) -> int:
    """ Build an integer key from a tuple of electron numbers. The maximum electron number must be less
    than 2**BIT_LENGTH. """

    # assert all(index < 1 << BIT_LENGTH for index in indices)
    return reduce(lambda num, index: (num << BIT_LENGTH) | index, indices, 0)


def braket_key(initial: tuple, final: tuple, parity: int) -> int:
    # assert parity in (0, 1)
    # assert len(initial) == len(final)
    # assert 1 <= len(initial) <= 3
    return (index_key(initial + final) << 1 | parity) << 3 | len(initial)


def braket_split_lower(key: int) -> (int, int):
    return key >> BIT_LENGTH, (key >> 3) & 1


def braket_split_upper(key: int) -> (int, int):
    num = key & 7
    parity = (key >> 3) & 1
    key >>= BIT_LENGTH
    shift = num * BIT_LENGTH
    mask = PART_MASK[num - 1]
    initial = (key >> shift) & mask
    final = key & mask
    return (final << shift) | initial, parity


def key_pair(key: int, num: int):
    initial = tuple((key >> shift) & INDEX_MASK for shift in INITIAL_SHIFT[num - 1])
    final = tuple((key >> shift) & INDEX_MASK for shift in FINAL_SHIFT[num - 1])
    return initial, final


def determinant_one(element, electron_initial, electron_final, parity):
    element.append(braket_key(electron_initial, electron_final, parity))


def determinant_two(element, electrons_a_initial, electrons_a_final, parity):
    electrons_b_initial = (electrons_a_initial[1], electrons_a_initial[0])
    electrons_b_final = (electrons_a_final[1], electrons_a_final[0])

    pos_parity = parity % 2
    neg_parity = (parity + 1) % 2

    element.append(braket_key(electrons_a_initial, electrons_a_final, pos_parity))
    element.append(braket_key(electrons_a_initial, electrons_b_final, neg_parity))
    element.append(braket_key(electrons_b_initial, electrons_a_final, neg_parity))
    element.append(braket_key(electrons_b_initial, electrons_b_final, pos_parity))


def determinant_three(element, electrons_initial, electrons_final, element_parity):
    initial_electrons = [
        electrons_initial,
        (electrons_initial[2], electrons_initial[1], electrons_initial[0]),
        (electrons_initial[1], electrons_initial[2], electrons_initial[0]),
        (electrons_initial[1], electrons_initial[0], electrons_initial[2]),
        (electrons_initial[2], electrons_initial[0], electrons_initial[1]),
        (electrons_initial[0], electrons_initial[2], electrons_initial[1])]

    final_electrons = [
        electrons_final,
        (electrons_final[2], electrons_final[1], electrons_final[0]),
        (electrons_final[1], electrons_final[2], electrons_final[0]),
        (electrons_final[1], electrons_final[0], electrons_final[2]),
        (electrons_final[2], electrons_final[0], electrons_final[1]),
        (electrons_final[0], electrons_final[2], electrons_final[1])]

    for i, initial in enumerate(initial_electrons):
        for j, final in enumerate(final_electrons):
            parity = (element_parity + i + j) % 2
            key = braket_key(initial, final, parity)
            element.append(key)


def single_one(matrix):
    single_index = []
    single_data = []

    # No different electron (diagonal element)
    for state_index, state_electrons in enumerate(matrix[0]):
        element = []
        for electron in state_electrons:
            determinant_one(element, (electron,), (electron,), 0)
        single_index.append([state_index, state_index, len(element)])
        single_data += element

    # One different electron
    for initial_index, final_index, same_electrons, initial_electrons, final_electrons, parity in matrix[1]:
        element = []
        determinant_one(element, initial_electrons, final_electrons, parity)
        single_index.append([initial_index, final_index, len(element)])
        single_data += element

    return single_index, single_data


def single_two(matrix):
    single_index = []
    single_data = []

    # No different electron (diagonal element)
    for state_index, state_electrons in enumerate(matrix[0]):
        element = []
        for electrons in combinations(state_electrons, 2):
            determinant_two(element, electrons, electrons, 0)
        single_index.append([state_index, state_index, len(element)])
        single_data += element

    # One different electron
    for initial_index, final_index, same_electrons, initial_electrons, final_electrons, parity in matrix[1]:
        element = []
        for same in same_electrons:
            initial = (same,) + initial_electrons
            final = (same,) + final_electrons
            determinant_two(element, initial, final, parity)
        single_index.append([initial_index, final_index, len(element)])
        single_data += element

    # Two different electrons
    for initial_index, final_index, same_electrons, initial_electrons, final_electrons, parity in matrix[2]:
        element = []
        determinant_two(element, initial_electrons, final_electrons, parity)
        single_index.append([initial_index, final_index, len(element)])
        single_data += element

    return single_index, single_data


def single_three(matrix):
    single_index = []
    single_data = []

    # No different electron (diagonal element)
    for state_index, state_electrons in enumerate(matrix[0]):
        element = []
        for electrons in combinations(state_electrons, 3):
            determinant_three(element, electrons, electrons, 0)
        single_index.append([state_index, state_index, len(element)])
        single_data += element

    # One different electron
    for initial_index, final_index, same_electrons, initial_electrons, final_electrons, parity in matrix[1]:
        element = []
        for same in combinations(same_electrons, 2):
            initial = same + initial_electrons
            final = same + final_electrons
            determinant_three(element, initial, final, parity)
        single_index.append([initial_index, final_index, len(element)])
        single_data += element

    # Two different electrons
    for initial_index, final_index, same_electrons, initial_electrons, final_electrons, parity in matrix[2]:
        element = []
        for same in same_electrons:
            initial = (same,) + initial_electrons
            final = (same,) + final_electrons
            determinant_three(element, initial, final, parity)
        single_index.append([initial_index, final_index, len(element)])
        single_data += element

    # Three different electrons
    for initial_index, final_index, same_electrons, initial_electrons, final_electrons, parity in matrix[3]:
        element = []
        determinant_three(element, initial_electrons, final_electrons, parity)
        single_index.append([initial_index, final_index, len(element)])
        single_data += element

    return single_index, single_data


def matrix_elements(states):
    """ Generate a list of potentially non-zero matrix elements. Return number of different electrons, indices
    of the matrix element, same electrons, different electrons in the initial and final states, and parity. """

    # Check every matrix element if it might be non-zero
    for final_index, (final_state, final_diff, final_keys) in enumerate(states):
        for initial_index, (initial_state, initial_diff, initial_keys) in enumerate(states[:final_index]):

            # Initial and final state of diagonal elements always share all electrons. This case is treated separately.
            if initial_index == final_index:
                continue

            # Find the minimum number of different electrons between initial and final state of the matrix element
            # in the range 1-3.
            for num in range(3):

                # There might be at most one match of the same electrons
                match = initial_keys[num] & final_keys[num]
                if len(match) > 1:
                    raise RuntimeError("Multiple state matches!")

                # Initial and final state differ in num+1 electrons
                elif len(match) == 1:

                    # Get tuples of same electrons, different initial and final electrons and swap numbers
                    key = match.pop()
                    same_electrons, initial_electrons, initial_swaps = initial_diff[num][key]
                    _, final_electrons, final_swaps = final_diff[num][key]

                    # Calculate parity based on the sum of swaps
                    parity = (initial_swaps + final_swaps) % 2

                    # Return indices of the matrix element, same and different electrons and parity
                    yield (num + 1, initial_index, final_index, same_electrons,
                           initial_electrons, final_electrons, parity)

                    # Deliver only the match with the smallest number of different electrons
                    break


def diff_electrons(diff, state):
    """ Build and return a diff-dictionary for a given state. Each electron in the state tuple is represented by an
    index to one of 2*(2l+1) different electrons. Take the tuple of num ordered electrons of the state and determine
    every possible splitting into the ordered tuples 'same' containing num-diff electrons and 'other' with diff
    electrons. Store the two tuples and the number of swap operations required for the splitting in the returned
    diff-dictionary. Use an integer key which identifies the tuple 'same'. The set of keys of the diff-dictionary
    provides an easy way to determine if two states differ in exactly diff electrons or not. """

    # Number of electrons
    num_electrons = len(state)

    # Build dictionary with all combinations of diff 'other' electrons
    diff_dict = {}
    if diff <= num_electrons:

        # Go through all combinations of diff electrons picked from the initial tuple
        indices = range(num_electrons)
        for selected in combinations(indices, diff):

            # num-diff electrons remaining in the 'same' tuple and keeping their order
            same = tuple(state[i] for i in indices if i not in selected)

            # Build an individual integer key from the 'same' tuple
            key = index_key(same)
            # assert key not in diff_dict

            # diff electrons in the 'other' tuple, also keeping their order
            other = tuple(state[i] for i in selected)

            # Number of neighbor swaps required to move the 'other' electrons to the end of the initial tuple
            swaps = diff * num_electrons - sum(range(diff + 1)) - sum(selected)

            # Store split tuples and the swap number in the dictionary
            diff_dict[key] = (same, other, swaps % 2)

    # Return the dictionary
    return diff_dict


def single_elements(states):
    """ Determine all matrix elements which might be non-zero for a given set of states. Return lists of
    parameters for elementary one-, two-, or three-electron tensor operators to be evaluated for the respective
    matrix elements. """

    print("Creating vault single ...")

    # Collect comparison data for each product state
    diff_states = []
    for electrons in states:
        # Get the diff-dictionaries for 1, 2, or 3 'other' electrons
        diff_dicts = [diff_electrons(diff, electrons) for diff in range(1, 4)]

        # Build sets of the 'same' key numbers
        diff_keys = [set(diff.keys()) for diff in diff_dicts]

        # Store state, three diff-dictionaries and the three respective sets of keys
        diff_states.append((electrons, diff_dicts, diff_keys))

    # Initialize a list of length 4, containing lists of all matrix elements (state pairs) which differ in exactly
    # 0-3 electrons. Diagonal matrix elements connect identical states, which differ in 0 electrons. Thus, the first
    # element contains the list of states
    elements = [states, [], [], []]

    # Collect all matrix elements with a minimum diff of 1, 2, or 3 electrons. All other elements are discarded.
    for diff_num, *data in matrix_elements(diff_states):
        elements[diff_num].append(data)

    # Number of electrons in the configuration
    num_electrons = len(states[0])

    # Determine parameters for all elementary one-electron tensor operators to be evaluated for each
    # potential matrix element (diff is 1)
    print("Vault single: collect one-electron indices ...")
    one = single_one(elements) if num_electrons >= 1 else None

    # Determine parameters for all elementary two-electron tensor operators to be evaluated for each
    # potential matrix element (diff is 1 or 2)
    print("Vault single: collect two-electron indices ...")
    two = single_two(elements) if num_electrons >= 2 else None

    # Determine parameters for all elementary three-electron tensor operators to be evaluated for each
    # potential matrix element (diff is 1, 2, or 3)
    print("Vault single: collect three-electron indices ...")
    three = single_three(elements) if num_electrons >= 3 else None

    # Return evaluation parameters for elementary one-, two-, or three-electron tensor operators
    print("Vault single done.")
    return one, two, three


class SingleElements:

    def __init__(self, group, num, states):
        assert num == len(states[0])

        self.group = group
        self.num = num
        self.states = states

    def elements(self, num: int):
        index = 0
        for initial_index, final_index, size in self[num]["indices"]:
            yield initial_index, final_index, slice(index, index + size)
            index += size

    def lower_keys(self, key_slice: slice, num: int):
        for key in self[num]["elements"][key_slice]:
            yield braket_split_lower(int(key))

    def upper_keys(self, key_slice: slice, num: int):
        for key in self[num]["elements"][key_slice]:
            yield braket_split_upper(int(key))

    def index_pair(self, key: int, num: int) -> (tuple, tuple):
        return key_pair(key, num)

    def __len__(self):
        return self.num

    def __getitem__(self, num):
        key = ["one", "two", "three"][num - 1]
        return self.group[key]

    def __iter__(self):
        if self.num >= 1:
            yield self.group["one"]
        if self.num >= 2:
            yield self.group["two"]
        if self.num >= 3:
            yield self.group["three"]


def init_single(vault, group_name, num, states):
    if group_name in vault:
        if not vault.attrs["valid"] or "version" not in vault[group_name].attrs or \
                vault[group_name].attrs["version"] != SINGLE_VERSION:
            del vault[group_name]
            vault.flush()

    if group_name not in vault:
        vault.attrs["valid"] = False
        vault.create_group(group_name)
        vault[group_name].attrs["version"] = SINGLE_VERSION

        single = single_elements(states)

        if num >= 1:
            group = vault[group_name].create_group("one")
            group.create_dataset("indices", data=single[0][0], compression="gzip", compression_opts=9)
            group.create_dataset("elements", data=single[0][1], compression="gzip", compression_opts=9)

        if num >= 2:
            group = vault[group_name].create_group("two")
            group.create_dataset("indices", data=single[1][0], compression="gzip", compression_opts=9)
            group.create_dataset("elements", data=single[1][1], compression="gzip", compression_opts=9)

        if num >= 3:
            group = vault[group_name].create_group("three")
            group.create_dataset("indices", data=single[2][0], compression="gzip", compression_opts=9)
            group.create_dataset("elements", data=single[2][1], compression="gzip", compression_opts=9)

        vault.flush()

    return SingleElements(vault[group_name], num, states)
