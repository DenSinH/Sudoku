import numpy as np
import itertools as it
import re


class Sudoku(object):

    def __init__(self, sudoku="", report=True):
        self.field = np.zeros((9, 9)).astype(int)
        self.report = report
        if not sudoku:
            with open("input.txt", "r") as f:
                sudoku = "".join([re.sub("\D", "", line) for line in f.readlines()])
        for i in range(9):
            for j in range(9):
                self.field[i, j] = int(sudoku[9 * i + j])

        if self.report:
            print(self.field)
            print()

        self.possibilities = np.ones((9, 9, 10), dtype=np.int8)  # 10 is so we don't account for 0
        self.possibilities[:, :, 0] = 0

    def set_possilities(self, irange, jrange, prev=""):
        # check possible values for i, j in irange, jrange
        isubsquares = set([i // 3 for i in irange])
        jsubsquares = set([j // 3 for j in jrange])

        if self.report:
            print(prev, irange, jrange, isubsquares, jsubsquares)

        for IRANGE in (irange, range(9)):
            for JRANGE in (range(9), jrange):
                for i in IRANGE:
                    for j in JRANGE:
                        if self.field[i, j]:
                            self.possibilities[i, j, :] = 0
                        self.possibilities[i, :, self.field[i, j]] = 0
                        self.possibilities[:, j, self.field[i, j]] = 0

        for iss in isubsquares:
            for jss in jsubsquares:
                for i in range(3):
                    for j in range(3):
                        if self.field[3*iss + i, 3*jss + j]:
                            self.possibilities[3*iss + i, 3*jss + j, :] = 0
                            self.possibilities[3*iss:3*iss + 3, 3*jss:3*jss + 3, self.field[3*iss + i, 3*jss + j]] = 0

        if not self.report:
            error = False
            if np.any(np.logical_and(np.sum(self.possibilities, axis=-1) == 0, self.field == 0)):
                error = True
            else:
                for iss in range(3):
                    for jss in range(3):
                        # region, row, col:
                        for slice in [((3*iss,3*iss + 3), (3*jss,3*jss + 3)),
                                      ((None, None), (3*iss + jss, 3*iss + jss + 1)),
                                      ((3*iss + jss, 3*iss + jss + 1), (None, None))]:
                            region = self.field[slice[0][0]:slice[0][1], slice[1][0]:slice[1][1]]
                            pos_reg = self.possibilities[slice[0][0]:slice[0][1], slice[1][0]:slice[1][1], :]
                            if np.sum(region) + sum(n for n in range(10)
                                                    if n not in region and np.any(pos_reg[:, :, n])) != 45:
                                error = True
                                break
            if error:
                raise AssertionError("Invalid attempt")

        if np.any(self.possibilities):
            self.square_fill()
            self.row_col_fill()
            self.set_group()
            self.xwing_swordfish()

    def square_fill(self):
        # fill squares based on if they can only be one number
        one_possibility_pos = np.argwhere(self.possibilities.sum(2) == 1)
        for pos in one_possibility_pos:
            tpos = tuple(pos)
            self.field[tpos] = int(np.where(self.possibilities[tpos] == 1)[0][0])
            self.possibilities[tpos] = 0
        if one_possibility_pos.size:
            self.set_possilities(one_possibility_pos[:, 0], one_possibility_pos[:, 1], prev="square_fill")

    def row_col_fill(self):
        # checks possibilities for rows and columns. If in some subsquare, a number can only be on positions in a line,
        # the rest of the line cannot be this number. Automatically also checks if a number is the only number in a line
        # that has the possibilitiy to be a certain number
        changed_i, changed_j = set(), set()
        for iss in range(3):
            for jss in range(3):
                for n in range(1, 10):
                    possibilities = np.nonzero(self.possibilities[3*iss:3*iss + 3, 3*jss:3*jss + 3, n])
                    for dir in (0, 1):  # i, j
                        if len(set(possibilities[dir])) == 1:
                            x_num = 3*[iss, jss][dir] + possibilities[dir][0]
                            y_num = np.mod(3*[jss, iss][dir] + 3 + np.arange(6), 9)

                            if np.any(self.possibilities[[x_num, y_num][dir], [y_num, x_num][dir], n] != 0):
                                self.possibilities[[x_num, y_num][dir], [y_num, x_num][dir], n] = 0
                                changed_i |= set(y_num if dir else [x_num])
                                changed_j |= set([x_num] if dir else y_num)

                            if possibilities[0].size == 1:
                                self.field[3*iss + possibilities[0][0], 3*jss + possibilities[1][0]] = n
                                changed_i.add(3*iss + possibilities[0][0])
                                changed_j.add(3*jss + possibilities[1][0])

        if changed_i:
            self.set_possilities(changed_i, changed_j, prev="row_col_fill")

    def set_group(self):
        changed_i, changed_j = set(), set()
        for iss in range(3):
            for jss in range(3):
                # region, row, col:
                for slice in [((3*iss,3*iss + 3), (3*jss,3*jss + 3)),
                              ((None, None), (3*iss + jss, 3*iss + jss + 1)),
                              ((3*iss + jss, 3*iss + jss + 1), (None, None))]:
                    region = self.field[slice[0][0]:slice[0][1], slice[1][0]:slice[1][1]]
                    pos_reg = self.possibilities[slice[0][0]:slice[0][1], slice[1][0]:slice[1][1], :]
                    reg_unused = set(range(9)) - set(region.flatten())
                    reg_num_poss = np.sum(pos_reg, axis=-1)

                    for group_size in range(2, len(reg_unused)):

                        # hidden group:
                        for group in it.combinations(reg_unused, group_size):
                            for n in group:
                                if np.sum(pos_reg[:, :, n]) != group_size:
                                    break
                            else:
                                totals = np.sum(pos_reg[..., group], axis=-1)
                                if np.all(np.logical_or(totals == 0, totals == group_size)):
                                    for pos in np.transpose(np.nonzero(totals)):
                                        to_change = [n for n in range(1, 10) if n not in group]
                                        if np.any(pos_reg[pos[0], pos[1], to_change]):
                                            pos_reg[pos[0], pos[1], to_change] = 0
                                            changed_i.add(int(slice[0][0] or 0) + pos[0])
                                            changed_j.add(int(slice[1][0] or 0) + pos[1])

                        # naked group:
                        if np.count_nonzero(reg_num_poss == group_size) == group_size:  # unnecessary line, but speeds up
                            group = set()
                            for n in range(1, 10):
                                if np.any(pos_reg[:, :, n][reg_num_poss == group_size]):
                                    group.add(n)
                            if len(group) == group_size:
                                group_positions = np.transpose(np.nonzero(reg_num_poss == group_size)).tolist()
                                for i in range(pos_reg.shape[0]):
                                    for j in range(pos_reg.shape[1]):
                                        if [i, j] not in group_positions:
                                            if np.any(pos_reg[i, j, list(group)]):
                                                pos_reg[i, j, list(group)] = 0
                                                changed_i.add(int(slice[0][0] or 0) + i)
                                                changed_j.add(int(slice[1][0] or 0) + j)

        if changed_i:
            self.set_possilities(changed_i, changed_j, prev="set_group")

    def xwing_swordfish(self):
        changed_i = set()
        changed_j = set()
        for n in range(1, 10):
            for dir in (0, 1):  # i, j
                for mode in (2, 3):  # xwing, swordfish
                    # line with 2 (or 3) possibilities, in direction dir
                    lines = list(np.nonzero(np.logical_and(np.sum(self.possibilities[:, :, n], axis=1 - dir) >= 2,
                                                           np.sum(self.possibilities[:, :, n], axis=1 - dir) <= mode))[0])
                    for line in it.combinations(lines, mode):
                        if np.count_nonzero(np.sum(np.transpose(self.possibilities, (dir, 1 - dir, 2))[line, :, n], axis=0)) == mode:
                            for perp in np.nonzero(np.sum(np.transpose(self.possibilities, (dir, 1 - dir, 2))[line, :, n], axis=0))[0]:
                                to_change = [i for i in range(9) if i not in line]
                                if np.any(self.possibilities[perp if dir else to_change, to_change if dir else perp, n]):
                                    self.possibilities[perp if dir else to_change, to_change if dir else perp, n] = 0
                                    changed_i |= set([perp] if dir else to_change)
                                    changed_j |= set(to_change if dir else [perp])

        if changed_i:
            self.set_possilities(changed_i, changed_j, prev="xwing/swordfish")

    def brute_force(self):
        tot_possible = np.sum(self.possibilities, axis=2)
        min_pos = np.transpose(np.where(tot_possible == (tot_possible[tot_possible > 0]).min()))
        # print(tot_possible)
        # print((tot_possible[tot_possible > 0]).min())
        # print(min_pos)
        for pos in min_pos:
            for n in np.nonzero(self.possibilities[pos[0], pos[1]])[0]:
                new_sudoku = "".join(str(d) for row in self.field for d in row)
                new_sudoku = new_sudoku[:9*pos[0] + pos[1]] + str(n) + new_sudoku[9*pos[0] + pos[1] + 1:]
                attempt = Sudoku(new_sudoku, report=False)
                try:
                    if self.report:
                        print("OUTER ATTEMPT")
                    else:
                        print("    INNER ATTEMPT")
                    if attempt.solve():
                        self.field = attempt.field
                        return True
                except AssertionError:
                    pass

    def solve(self):
        self.set_possilities(range(9), range(9), prev="solve")
        sums = [val == 45 for ax in (0, 1) for val in np.sum(self.field, axis=ax)]
        if (not all(sums)) and np.any(self.possibilities):
            if self.report:
                print("BRUTE FORCING")
            return self.brute_force()
        elif all(sums) and not np.any(self.possibilities):
            return True
        else:
            return False


if __name__ == "__main__":
    sudoku = Sudoku()
    sudoku.solve()

    for i in range(9):
        print(" | ".join(str(" ".join(str(n) for n in sudoku.field[i, 3*jss:3*jss + 3]))for jss in range(3)), "=", np.sum(sudoku.field[i]))
        if i in (2, 5):
            print("-" * 21)
    print([np.sum(sudoku.field[:, j]) for j in range(9)])
