from Operations import *


class SudokuSolver():
    def __init__(self, matrix):
        self.matrix = matrix
        matrix_transpose = transpose(matrix)
        matrix_blocks = blocks(matrix)
        if self.check_input():
            self.choices = []
            self.unfilled = 0
            self.row_valid = [0] * 81
            self.column_valid = [0] * 81
            self.block_valid = [0] * 81
            self.check = False

            for row in matrix:
                valid = [i for i in range(1, 10) if i not in row]
                for element in row:
                    if element:
                        self.choices.append([])
                    else:
                        self.choices.append(valid)
            for i in range(9):
                valid = [k for k in range(1, 10) if k not in matrix_transpose[i]]
                for j in range(9):
                    self.choices[i + 9 * j] = [k for k in self.choices[i + 9 * j] if k in valid]
            for i in range(9):
                valid = [k for k in range(1, 10) if k not in matrix_blocks[i]]
                for j in range(27 * int(i / 3) + 3 * (i % 3), 27 * int(i / 3) + 3 * (i % 3 + 1)):
                    self.choices[j] = [k for k in self.choices[j] if k in valid]
                    if self.choices[j]:
                        self.unfilled += 1
                    self.choices[9 + j] = [k for k in self.choices[9 + j] if k in valid]
                    if self.choices[9 + j]:
                        self.unfilled += 1
                    self.choices[18 + j] = [k for k in self.choices[18 + j] if k in valid]
                    if self.choices[18 + j]:
                        self.unfilled += 1

            for i in range(81):
                for element in self.choices[i]:
                    self.row_valid[9 * int(i / 9) + element - 1] += 1
                    self.column_valid[9 * (i % 9) + element - 1] += 1
                    self.block_valid[9 * block_number(i) + element - 1] += 1

            self.checked_combinations = []
            if not self.fill_matrix():
                self.matrix = []
        else:
            self.matrix = []

    def check_input(self):
        for row in self.matrix:
            for i in range(1, 10):
                if row.count(i) > 1:
                    return False
        for i in range(9):
            column = [row[i] for row in self.matrix]
            for j in range(1, 10):
                if column.count(j) > 1:
                    return False
        for i in range(9):
            block = [self.matrix[int(j / 9)][j % 9] for j in block_as_a_list(i)]
            for j in range(1, 10):
                if block.count(j) > 1:
                    return False
        return True

    def remove_choice(self, choice, index):
        self.row_valid[9 * int(index / 9) + choice - 1] -= 1
        self.column_valid[9 * (index % 9) + choice - 1] -= 1
        self.block_valid[9 * block_number(index) + choice - 1] -= 1

    def fill_choice(self, index, fill=0):
        # global choices, matrix, unfilled, row_valid, column_valid, block_valid
        self.unfilled -= 1
        if fill == 0:
            fill = self.choices[index][0]
        row = int(index / 9)
        column = index % 9
        self.matrix[row][column] = fill
        for other_possib in self.choices[index]:
            self.row_valid[9 * row + other_possib - 1] -= 1
            self.column_valid[9 * column + other_possib - 1] -= 1
            self.block_valid[9 * block_number(row, column) + other_possib - 1] -= 1
        self.choices[index] = []
        for j in range(9 * row, 9 * (row + 1)):
            if j != index and not self.choices[j] and self.matrix[row][j % 9] == fill:
                return False
            if fill in self.choices[j]:
                self.remove_choice(fill, j)
                self.choices[j].remove(fill)
                if len(self.choices[j]) == 1:
                    if not self.fill_choice(j):
                        return False
                elif not self.choices[j]:
                    return False
        for j in range(column, column + 81, 9):
            if j != index and not self.choices[j] and self.matrix[int(j / 9)][column] == fill:
                return False
            if fill in self.choices[j]:
                self.remove_choice(fill, j)
                self.choices[j].remove(fill)
                if len(self.choices[j]) == 1:
                    if not self.fill_choice(j):
                        return False
                elif not self.choices[j]:
                    return False
        for i in range(3 * int(row / 3), 3 * (int(row / 3) + 1)):
            for j in range(9 * i + column - column % 3, 9 * i + 3 + column - column % 3):
                if j != index and not self.choices[j] and self.matrix[int(j / 9)][j % 9] == fill:
                    return False
                if fill in self.choices[j]:
                    self.remove_choice(fill, j)
                    self.choices[j].remove(fill)
                    if len(self.choices[j]) == 1:
                        if not self.fill_choice(j):
                            return False
                    elif not self.choices[j]:
                        return False
        return True

    def fill_matrix(self):
        num_tries = 0
        while self.unfilled and num_tries < 100:
            change_exists = False
            num_tries += 1

            for i in range(81):
                if len(self.choices[i]) == 1:
                    if not self.fill_choice(i):
                        return False
                    change_exists = True
                if not self.choices[i] and not self.matrix[int(i / 9)][i % 9]:
                    return False

            if not self.unfilled:
                break

            for i in range(81):
                if self.row_valid[i] == 1:
                    element = 1 + i % 9
                    for j in range(9 * int(i / 9), 9 * (int(i / 9) + 1)):
                        if element in self.choices[j]:
                            if not self.fill_choice(j, element):
                                return False
                            change_exists = True
                            break
                if self.column_valid[i] == 1:
                    element = 1 + i % 9
                    for j in range(int(i / 9), 81 + int(i / 9), 9):
                        if element in self.choices[j]:
                            if not self.fill_choice(j, element):
                                return False
                            change_exists = True
                            break
                if self.block_valid[i] == 1:
                    element = 1 + i % 9
                    for j in block_as_a_list(int(i / 9)):
                        if element in self.choices[j]:
                            if not self.fill_choice(j, element):
                                return False
                            change_exists = True
                            break

            if not self.unfilled:
                break

            for i in range(9):
                block_list = block_as_a_list(i)
                for x in range(9):
                    if len(self.choices[block_list[x]]) == 2:
                        for y in range(x + 1, 9):
                            if self.choices[block_list[y]] == self.choices[block_list[x]]:
                                X = block_list[x]
                                Y = block_list[y]
                                if 100 * X + Y in self.checked_combinations:
                                    continue
                                self.checked_combinations.append(100 * X + Y)
                                for choice in self.choices[X]:
                                    for index in block_list:
                                        if index != X and index != Y and choice in self.choices[index]:
                                            change_exists = True
                                            self.remove_choice(choice, index)
                                            self.choices[index].remove(choice)
                                            if len(self.choices[index]) == 1:
                                                if not self.fill_choice(index):
                                                    return False
                                            elif not self.choices[index] and not self.matrix[int(index / 9)][index % 9]:
                                                return False
                                    if int(X / 9) == int(Y / 9):
                                        for index in range(9 * int(X / 9), 9 * int(X / 9) + 9):
                                            if index != X and index != Y and choice in self.choices[index]:
                                                self.remove_choice(choice, index)
                                                self.choices[index].remove(choice)
                                                change_exists = True
                                                if len(self.choices[index]) == 1:
                                                    self.fill_choice(index)
                                                elif not self.choices[index] and not self.matrix[int(index / 9)][
                                                    index % 9]:
                                                    return False
                                    if X % 9 == Y % 9:
                                        for index in range(X % 9, 81 + (X % 9), 9):
                                            if index != X and index != Y and choice in self.choices[index]:
                                                self.remove_choice(choice, index)
                                                self.choices[index].remove(choice)
                                                change_exists = True
                                                if len(self.choices[index]) == 1:
                                                    self.fill_choice(index)
                                                elif not self.choices[index] and not self.matrix[int(index / 9)][
                                                    index % 9]:
                                                    return False
            if not self.unfilled:
                break

            if not change_exists:
                minIndex = 0
                while not self.choices[minIndex]:
                    minIndex += 1
                for i in range(81):
                    if self.choices[i] and len(self.choices[i]) < len(self.choices[minIndex]):
                        minIndex = i
                for guess in self.choices[minIndex]:
                    # print(matrix)
                    old_matrix = [[num for num in row] for row in self.matrix]
                    old_choices = [[choice for choice in choice_list] for choice_list in self.choices]
                    old_row_valid = [num for num in self.row_valid]
                    old_column_valid = [num for num in self.column_valid]
                    old_block_valid = [num for num in self.block_valid]
                    old_unfilled = self.unfilled
                    if not self.fill_choice(minIndex, guess) or not self.fill_matrix():
                        self.matrix = old_matrix
                        self.choices = old_choices
                        self.choices[minIndex].remove(guess)
                        self.row_valid = old_row_valid
                        self.column_valid = old_column_valid
                        self.block_valid = old_block_valid
                        self.remove_choice(guess, minIndex)
                        self.unfilled = old_unfilled
                    else:
                        break

        if not self.unfilled:
            return True
        return False

    def display_self_matrix(self):
        display_matrix(self.matrix)
