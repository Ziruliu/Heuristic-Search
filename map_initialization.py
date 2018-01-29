# coding=utf-8
import numpy as np
import math
from random import randint

# Global Variable
row = 120
column = 160

def httCellsPlaced(map):
    centercells = np.zeros((8, 2))
    for i in range(0, 8):
        xrand = randint(0, map.shape[0] - 1)
        yrand = randint(0, map.shape[1] - 1)

        for j in range(0, i):
            while xrand == centercells[j][0] and yrand == centercells[j][1]:
                xrand = randint(0, map.shape[0] - 1)
                yrand = randint(0, map.shape[1] - 1)
        centercells[i][0] = xrand
        centercells[i][1] = yrand

    for i in range(0, 8):
        x = centercells[i][0]
        y = centercells[i][1]
        x_count = 0
        y_count = 0
        # move up:
        while (x > 0):
            x -= 1
            x_count += 1
            if x_count == 15:
                break
        # move left:
        while (y > 0):
            y -= 1
            y_count += 1
            if y_count == 15:
                break
        # Place harder to traverse cells. '2' means harder to traverse cells
        if (x + 31 > map.shape[0] - 1):
            x_bound = map.shape[0] - 1
        else:
            x_bound = x + 31
        if (y + 31 > map.shape[1] - 1):
            y_bound = map.shape[1] - 1
        else:
            y_bound = y + 31
        for m in range(int(x), int(x_bound)):
            for n in range(int(y), int(y_bound)):
                prob = randint(0, 1)
                if prob == 0:
                    map[m][n] = '2'

    return map, centercells


def highwaysPlaced(map):
    # Generate numbers to indicate boundaries
    n_bound = 0
    s_bound = 1
    w_bound = 2
    e_bound = 3
    # Placement of highways.
    # Use ’a’ to indicate a regular unblocked cell with a highway
    # Use ’b’ to indicate a hard to traverse cell with a highway
    i = 0
    while i < 4:
        start = randint(0, 3)
        temp_map = map.copy()
        re = 0
        # Start at north
        if start == n_bound:
            x = 0
            y = randint(0, map.shape[1] - 1)
            while temp_map[x, y] == 'a' or temp_map[x, y] == 'b':
                y = randint(0, map.shape[1] - 1)
            if temp_map[x, y] == '1':
                temp_map[x, y] = 'a'
            else:
                temp_map[x, y] = 'b'
            count = 1
            total_count = 1
            while count < 20:
                x += 1
                count += 1
                total_count += 1
                if temp_map[x, y] == 'a' or temp_map[x, y] == 'b':
                    re = 1
                    break
                else:
                    if temp_map[x, y] == '1':
                        temp_map[x, y] = 'a'
                    else:
                        temp_map[x, y] = 'b'
            if re == 1:
                continue
            countforloop = 0
            while x != 0 and x != map.shape[0] - 1 and y != 0 and y != map.shape[1] - 1:
                if countforloop > 50:
                    re = 1
                    break
                countforloop += 1
                re = 0
                prob = randint(0, 9)
                count = 0
                temp_sub = temp_map.copy()
                temp_x = x
                temp_y = y
                # Countinue
                if prob < 6:
                    while count < 20:
                        temp_x += 1
                        if temp_x == map.shape[0] - 1:
                            if temp_sub[temp_x, temp_y] == '1':
                                temp_sub[temp_x, temp_y] = 'a'
                            else:
                                temp_sub[temp_x, temp_y] = 'b'
                            break
                        count += 1
                        if temp_sub[temp_x, temp_y] == 'a' or temp_sub[temp_x, temp_y] == 'b':
                            re = 1
                            break
                        else:
                            if temp_sub[temp_x, temp_y] == '1':
                                temp_sub[temp_x, temp_y] = 'a'
                            else:
                                temp_sub[temp_x, temp_y] = 'b'
                # Turn left
                elif prob < 8:
                    while count < 20:
                        temp_y += 1
                        if temp_y == map.shape[1] - 1:
                            if temp_sub[temp_x, temp_y] == '1':
                                temp_sub[temp_x, temp_y] = 'a'
                            else:
                                temp_sub[temp_x, temp_y] = 'b'
                            break
                        count += 1
                        if temp_sub[temp_x, temp_y] == 'a' or temp_sub[temp_x, temp_y] == 'b':
                            re = 1
                            break
                        else:
                            if temp_sub[temp_x, temp_y] == '1':
                                temp_sub[temp_x, temp_y] = 'a'
                            else:
                                temp_sub[temp_x, temp_y] = 'b'
                # Turn right
                else:
                    while count < 20:
                        temp_y -= 1
                        if temp_y == 0:
                            if temp_sub[temp_x, temp_y] == '1':
                                temp_sub[temp_x, temp_y] = 'a'
                            else:
                                temp_sub[temp_x, temp_y] = 'b'
                            break
                        count += 1
                        if temp_sub[temp_x, temp_y] == 'a' or temp_sub[temp_x, temp_y] == 'b':
                            re = 1
                            break
                        else:
                            if temp_sub[temp_x, temp_y] == '1':
                                temp_sub[temp_x, temp_y] = 'a'
                            else:
                                temp_sub[temp_x, temp_y] = 'b'
                if re == 1:
                    continue
                total_count += count
                x = temp_x
                y = temp_y
                temp_map = temp_sub.copy()
            if total_count < 100 or re == 1:
                continue
            else:
                i += 1
                map = temp_map.copy()
        # Start at south
        if start == s_bound:
            x = map.shape[0] - 1
            y = randint(0, map.shape[1] - 1)
            while temp_map[x, y] == 'a' or temp_map[x, y] == 'b':
                y = randint(0, map.shape[1] - 1)
            if temp_map[x, y] == '1':
                temp_map[x, y] = 'a'
            else:
                temp_map[x, y] = 'b'
            count = 1
            total_count = 1
            while count < 20:
                x -= 1
                count += 1
                total_count += 1
                if temp_map[x, y] == 'a' or temp_map[x, y] == 'b':
                    re = 1
                    break
                else:
                    if temp_map[x, y] == '1':
                        temp_map[x, y] = 'a'
                    else:
                        temp_map[x, y] = 'b'
            if re == 1:
                continue
            countforloop = 0
            while x != 0 and x != map.shape[0] - 1 and y != 0 and y != map.shape[1] - 1:
                if countforloop > 50:
                    re = 1
                    break
                countforloop += 1
                re = 0
                prob = randint(0, 9)
                count = 0
                temp_sub = temp_map.copy()
                temp_x = x
                temp_y = y
                # Countinue
                if prob < 6:
                    while count < 20:
                        temp_x -= 1
                        if temp_x == 0:
                            if temp_sub[temp_x, temp_y] == '1':
                                temp_sub[temp_x, temp_y] = 'a'
                            else:
                                temp_sub[temp_x, temp_y] = 'b'
                            break
                        count += 1
                        if temp_sub[temp_x, temp_y] == 'a' or temp_sub[temp_x, temp_y] == 'b':
                            re = 1
                            break
                        else:
                            if temp_sub[temp_x, temp_y] == '1':
                                temp_sub[temp_x, temp_y] = 'a'
                            else:
                                temp_sub[temp_x, temp_y] = 'b'
                # Turn left
                elif prob < 8:
                    while count < 20:
                        temp_y -= 1
                        if temp_y == 0:
                            if temp_sub[temp_x, temp_y] == '1':
                                temp_sub[temp_x, temp_y] = 'a'
                            else:
                                temp_sub[temp_x, temp_y] = 'b'
                            break
                        count += 1
                        if temp_sub[temp_x, temp_y] == 'a' or temp_sub[temp_x, temp_y] == 'b':
                            re = 1
                            break
                        else:
                            if temp_sub[temp_x, temp_y] == '1':
                                temp_sub[temp_x, temp_y] = 'a'
                            else:
                                temp_sub[temp_x, temp_y] = 'b'
                # Turn right
                else:
                    while count < 20:
                        temp_y += 1
                        if temp_y == map.shape[1] - 1:
                            if temp_sub[temp_x, temp_y] == '1':
                                temp_sub[temp_x, temp_y] = 'a'
                            else:
                                temp_sub[temp_x, temp_y] = 'b'
                            break
                        count += 1
                        if temp_sub[temp_x, temp_y] == 'a' or temp_sub[temp_x, temp_y] == 'b':
                            re = 1
                            break
                        else:
                            if temp_sub[temp_x, temp_y] == '1':
                                temp_sub[temp_x, temp_y] = 'a'
                            else:
                                temp_sub[temp_x, temp_y] = 'b'
                if re == 1:
                    continue
                total_count += count
                x = temp_x
                y = temp_y
                temp_map = temp_sub.copy()
            if total_count < 100 or re == 1:
                continue
            else:
                i += 1
                map = temp_map.copy()
        # Start at west
        if start == w_bound:
            x = randint(0, map.shape[0] - 1)
            y = 0
            while temp_map[x, y] == 'a' or temp_map[x, y] == 'b':
                x = randint(0, map.shape[0] - 1)
            if temp_map[x, y] == '1':
                temp_map[x, y] = 'a'
            else:
                temp_map[x, y] = 'b'
            count = 1
            total_count = 1
            while count < 20:
                y += 1
                count += 1
                total_count += 1
                if temp_map[x, y] == 'a' or temp_map[x, y] == 'b':
                    re = 1
                    break
                else:
                    if temp_map[x, y] == '1':
                        temp_map[x, y] = 'a'
                    else:
                        temp_map[x, y] = 'b'
            if re == 1:
                continue
            countforloop = 0
            while x != 0 and x != map.shape[0] - 1 and y != 0 and y != map.shape[1] - 1:
                if countforloop > 50:
                    re = 1
                    break
                countforloop += 1
                re = 0
                prob = randint(0, 9)
                count = 0
                temp_sub = temp_map.copy()
                temp_x = x
                temp_y = y
                # Countinue
                if prob < 6:
                    while count < 20:
                        temp_y += 1
                        if temp_y == map.shape[1] - 1:
                            if temp_sub[temp_x, temp_y] == '1':
                                temp_sub[temp_x, temp_y] = 'a'
                            else:
                                temp_sub[temp_x, temp_y] = 'b'
                            break
                        count += 1
                        if temp_sub[temp_x, temp_y] == 'a' or temp_sub[temp_x, temp_y] == 'b':
                            re = 1
                            break
                        else:
                            if temp_sub[temp_x, temp_y] == '1':
                                temp_sub[temp_x, temp_y] = 'a'
                            else:
                                temp_sub[temp_x, temp_y] = 'b'
                # Turn left
                elif prob < 8:
                    while count < 20:
                        temp_x -= 1
                        if temp_x == 0:
                            if temp_sub[temp_x, temp_y] == '1':
                                temp_sub[temp_x, temp_y] = 'a'
                            else:
                                temp_sub[temp_x, temp_y] = 'b'
                            break
                        count += 1
                        if temp_sub[temp_x, temp_y] == 'a' or temp_sub[temp_x, temp_y] == 'b':
                            re = 1
                            break
                        else:
                            if temp_sub[temp_x, temp_y] == '1':
                                temp_sub[temp_x, temp_y] = 'a'
                            else:
                                temp_sub[temp_x, temp_y] = 'b'
                # Turn right
                else:
                    while count < 20:
                        temp_x += 1
                        if temp_x == map.shape[0] - 1:
                            if temp_sub[temp_x, temp_y] == '1':
                                temp_sub[temp_x, temp_y] = 'a'
                            else:
                                temp_sub[temp_x, temp_y] = 'b'
                            break
                        count += 1
                        if temp_sub[temp_x, temp_y] == 'a' or temp_sub[temp_x, temp_y] == 'b':
                            re = 1
                            break
                        else:
                            if temp_sub[temp_x, temp_y] == '1':
                                temp_sub[temp_x, temp_y] = 'a'
                            else:
                                temp_sub[temp_x, temp_y] = 'b'
                if re == 1:
                    continue
                total_count += count
                x = temp_x
                y = temp_y
                temp_map = temp_sub.copy()
            if total_count < 100 or re == 1:
                continue
            else:
                i += 1
                map = temp_map.copy()
                # Start at east
        if start == e_bound:
            x = randint(0, map.shape[0] - 1)
            y = map.shape[1] - 1
            while temp_map[x, y] == 'a' or temp_map[x, y] == 'b':
                x = randint(0, map.shape[0] - 1)
            if temp_map[x, y] == '1':
                temp_map[x, y] = 'a'
            else:
                temp_map[x, y] = 'b'
            count = 1
            total_count = 1
            while count < 20:
                y -= 1
                count += 1
                total_count += 1
                if temp_map[x, y] == 'a' or temp_map[x, y] == 'b':
                    re = 1
                    break
                else:
                    if temp_map[x, y] == '1':
                        temp_map[x, y] = 'a'
                    else:
                        temp_map[x, y] = 'b'
            if re == 1:
                continue
            countforloop = 0
            while x != 0 and x != map.shape[0] - 1 and y != 0 and y != map.shape[1] - 1:
                if countforloop > 50:
                    re = 1
                    break
                countforloop += 1
                re = 0
                prob = randint(0, 9)
                count = 0
                temp_sub = temp_map.copy()
                temp_x = x
                temp_y = y
                # Countinue
                if prob < 6:
                    while count < 20:
                        temp_y -= 1
                        if temp_y == 0:
                            if temp_sub[temp_x, temp_y] == '1':
                                temp_sub[temp_x, temp_y] = 'a'
                            else:
                                temp_sub[temp_x, temp_y] = 'b'
                            break
                        count += 1
                        if temp_sub[temp_x, temp_y] == 'a' or temp_sub[temp_x, temp_y] == 'b':
                            re = 1
                            break
                        else:
                            if temp_sub[temp_x, temp_y] == '1':
                                temp_sub[temp_x, temp_y] = 'a'
                            else:
                                temp_sub[temp_x, temp_y] = 'b'
                # Turn left
                elif prob < 8:
                    while count < 20:
                        temp_x += 1
                        if temp_x == map.shape[0] - 1:
                            if temp_sub[temp_x, temp_y] == '1':
                                temp_sub[temp_x, temp_y] = 'a'
                            else:
                                temp_sub[temp_x, temp_y] = 'b'
                            break
                        count += 1
                        if temp_sub[temp_x, temp_y] == 'a' or temp_sub[temp_x, temp_y] == 'b':
                            re = 1
                            break
                        else:
                            if temp_sub[temp_x, temp_y] == '1':
                                temp_sub[temp_x, temp_y] = 'a'
                            else:
                                temp_sub[temp_x, temp_y] = 'b'
                # Turn right
                else:
                    while count < 20:
                        temp_x -= 1
                        if temp_x == 0:
                            if temp_sub[temp_x, temp_y] == '1':
                                temp_sub[temp_x, temp_y] = 'a'
                            else:
                                temp_sub[temp_x, temp_y] = 'b'
                            break
                        count += 1
                        if temp_sub[temp_x, temp_y] == 'a' or temp_sub[temp_x, temp_y] == 'b':
                            re = 1
                            break
                        else:
                            if temp_sub[temp_x, temp_y] == '1':
                                temp_sub[temp_x, temp_y] = 'a'
                            else:
                                temp_sub[temp_x, temp_y] = 'b'
                if re == 1:
                    continue
                total_count += count
                x = temp_x
                y = temp_y
                temp_map = temp_sub.copy()
            if total_count < 100 or re == 1:
                continue
            else:
                i += 1
                map = temp_map.copy()

    return map


def blockedCellsPlaced(map):
    m = map.shape[0]
    n = map.shape[1]
    # To difine number of blocked cells needed to be placed. Should be 120 * 160 + 0.2 = 3840 initially.
    blocked_cells_left = m * n * 0.2
    # Placement of blocked cells. '0' means blocked cells
    while blocked_cells_left > 0:
        xrand = randint(0, map.shape[0] - 1)
        yrand = randint(0, map.shape[1] - 1)
        if map[xrand, yrand] == 'a' or map[xrand, yrand] == 'b':
            continue
        else:
            map[xrand, yrand] = '0'
        blocked_cells_left -= 1
    return map


def start_goal_selected(map):
    re = 1
    while re == 1:
        # Selection of start vertex
        x_start = randint(0, map.shape[0] - 1)
        y_start = randint(0, map.shape[1] - 1)
        while x_start > 19 and x_start < map.shape[0] - 20:
            x_start = randint(0, map.shape[0] - 1)
        while y_start > 19 and y_start < map.shape[1] - 20:
            y_start = randint(0, map.shape[1] - 1)
        # Selection of goal vertex
        x_goal = randint(0, map.shape[0] - 1)
        y_goal = randint(0, map.shape[1] - 1)
        while x_goal > 19 and x_goal < map.shape[0] - 20:
            x_goal = randint(0, map.shape[0] - 1)
        while y_goal > 19 and y_goal < map.shape[1] - 20:
            y_goal = randint(0, map.shape[1] - 1)
        # Calculate the Euclidean Distance to finalize the start and goal vertex
        dist = math.sqrt((x_goal - x_start) ** 2 + (y_goal - y_start) ** 2)
        if dist < 100:
            continue
        else:
            return x_start, y_start, x_goal, y_goal


def main():

    for m in range(0, 5):

        # Intialize a 120 * 160 map with all unblocked cells. '1' means unblocked cells
        map = np.chararray((row, column))
        map[:] = '1'

        # Placement of harder to traverse cells. '2' means harder to traverse cells
        map, centercells = httCellsPlaced(map)

        # Placement of highways.
        # Use ’a’ to indicate a regular unblocked cell with a highway
        # Use ’b’ to indicate a hard to traverse cell with a highway
        map = highwaysPlaced(map)

        # Placement of blocked cells. '0' means blocked cells
        map = blockedCellsPlaced(map)

        for n in range(0, 10):
            # Selection of start vertex and goal vertex
            x_start, y_start, x_goal, y_goal = start_goal_selected(map)

            # Write data into files
            f = open("map_" + str(m) + "_" + str(n) + ".txt", "w+")
            f.write("%d, %d\n" % (x_start, y_start))
            f.write("%d, %d\n" % (x_goal, y_goal))
            for i in range(0, len(centercells)):
                f.write("%d, %d\n" % (centercells[i, 0], centercells[i, 1]))
            for x in range(0, map.shape[0]):
                for y in range(0, map.shape[1]):
                    f.write("%c " % map[x, y])
                f.write("\n")
            f.close


if __name__ == '__main__':
    main()
