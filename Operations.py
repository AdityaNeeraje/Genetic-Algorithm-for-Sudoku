def transpose(matrix):
    result = [[matrix[i][j] for i in range(9)] for j in range(9)]
    return result


def display_matrix(matrix):
    if len(matrix) == 9:
        for row in matrix:
            print(row)
    if len(matrix) == 81:
        for i in range(9):
            print(matrix[9*i: 9*i + 9])

def blocks(matrix):
    result = [[], [], [], [], [], [], [], [], []]
    for i in range(9):
        for j in range(9):
            result[3 * int(i / 3) + int(j / 3)].append(matrix[i][j])
    return result


def block_number(index, j=-1):
    if j == -1:
        column = index % 9
        row = (index - column) / 9
        return 3 * int(row / 3) + int(column / 3)
    else:
        return 3 * int(index / 3) + int(j / 3)


def block_as_a_list(index):
    result = []
    for i in range(27 * int(index / 3) + 3*(index%3), 27 * int(index / 3) + (3*(index%3)) + 27, 9):
        for j in range(3):
            result.append(i + j)
    return result


