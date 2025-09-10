##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

def table_to_columns(input_string):
    # Split the input string into lines
    lines = input_string.strip().split('\n')

    # Find the first non-empty line to determine the number of columns
    num_columns = 0
    for line in lines:
        line = line.strip()
        if line:
            num_columns = len(line.split())
            break

    # Prepare lists for each column
    columns = [[] for _ in range(num_columns)]

    # Process each line
    for line in lines:
        line = line.strip()
        line = line.replace('~', '')
        line = line.replace('O', '0')
        assert "􀀰" not in line
        if line:
            split_line = line.split()
            _ = map(float, split_line)
            if len(split_line) != num_columns:
                print(f">>> {split_line} <<<")
            for i in range(num_columns):
                columns[i].append(split_line[i])

    # Join each column's data with commas and return
    columns = [', '.join(column) for column in columns]
    columns = [f'"U{2 * i + 2}":\n    [{column}],' for i, column in enumerate(columns)]
    for column in columns:
        print(column)


def column_to_list(input_string, name=None):
    if name is None:
        name = "energies"
    lines = input_string.strip().split('\n')
    result = []
    for line in lines:
        line = line.strip()
        line = line.replace(' ', '')
        assert "􀀰" not in line
        _ = float(line)
        result.append(line)
    result = ', '.join(result)
    result = f'"{name}":\n    [{result}],'
    print(result)

a = """
"""

column_to_list(a)
#column_to_list(b)
#table_to_columns(c)

