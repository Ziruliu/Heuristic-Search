import numpy as np
import math
import heapq

# Global variables
r = 120
c = 160


class Node(object):
    def __init__(self, x, y, key):
        self.x = x
        self.y = y
        self.h = 0.0
        self.g = 0.0
        self.f = 0.0
        self.key = key
        self.parent = None

    def get_coordinates(self):
        return self.x, self.y

    def get_key(self):
        return self.key


class HeuristicFamily(object):
    def __init__(self):
        self.fringe = []
        self.closed = []
        self.path = []
        self.Matrix = []
        self.heuristic = ''
        self.weight = 0.0

    def compute_h_value(self, CurrentNode, GoalNode):
        pass

    def Scompute_hS_value(self, Cx,Cy,Gx,Gy, num):
        pass

    def replace(self, matrix_gvalue, matrix_hvalue, matrix_fvalue, matrix_parent):
        for x in range(0, r):
            for y in range(0, c):
                self.Matrix[x][y].g = matrix_gvalue[x][y]
                self.Matrix[x][y].h = matrix_hvalue[x][y]
                self.Matrix[x][y].f = matrix_fvalue[x][y]
                xyprime = matrix_parent[x][y]
                self.Matrix[x][y].parent = self.Matrix[xyprime[0]][xyprime[1]]

    def all_to_node(self, input_matrix):
        for x in range(0, r):
            row = []
            for y in range(0, c):
                row.append(Node(x, y, input_matrix[x, y]))
            self.Matrix.append(row)

    def succ(self, s):
        list = []
        s = s.get_coordinates()
        x_s, y_s = s[0], s[1]
        if x_s == 0 and y_s == 0:
            list.append((x_s + 1, y_s))
            list.append((x_s, y_s + 1))
            list.append((x_s + 1, y_s + 1))
        elif x_s == 0 and y_s == c - 1:
            list.append((x_s + 1, y_s))
            list.append((x_s, y_s - 1))
            list.append((x_s + 1, y_s - 1))
        elif x_s == r - 1 and y_s == 0:
            list.append((x_s - 1, y_s))
            list.append((x_s, y_s + 1))
            list.append((x_s - 1, y_s + 1))
        elif x_s == r - 1 and y_s == c - 1:
            list.append((x_s - 1, y_s))
            list.append((x_s, y_s - 1))
            list.append((x_s - 1, y_s - 1))
        elif x_s != 0 and x_s != r - 1 and y_s == 0:
            list.append((x_s + 1, y_s))
            list.append((x_s - 1, y_s))
            list.append((x_s, y_s + 1))
            list.append((x_s + 1, y_s + 1))
            list.append((x_s - 1, y_s + 1))
        elif x_s != 0 and x_s != r - 1 and y_s == c - 1:
            list.append((x_s + 1, y_s))
            list.append((x_s - 1, y_s))
            list.append((x_s + 1, y_s - 1))
            list.append((x_s - 1, y_s - 1))
            list.append((x_s, y_s - 1))
        elif x_s == 0 and y_s != 0 and y_s != c - 1:
            list.append((x_s + 1, y_s))
            list.append((x_s + 1, y_s + 1))
            list.append((x_s + 1, y_s - 1))
            list.append((x_s, y_s + 1))
            list.append((x_s, y_s - 1))
        elif x_s == r - 1 and y_s != 0 and y_s != c - 1:
            list.append((x_s - 1, y_s))
            list.append((x_s - 1, y_s + 1))
            list.append((x_s - 1, y_s - 1))
            list.append((x_s, y_s + 1))
            list.append((x_s, y_s - 1))
        else:
            list.append((x_s - 1, y_s))
            list.append((x_s + 1, y_s))
            list.append((x_s, y_s - 1))
            list.append((x_s, y_s + 1))
            list.append((x_s - 1, y_s - 1))
            list.append((x_s - 1, y_s + 1))
            list.append((x_s + 1, y_s - 1))
            list.append((x_s + 1, y_s + 1))

        return list

    # Get cost of transition between two neighboring vertices s1 and s2
    def cost(self, s1, s2):
        c = 0
        s1_key = s1.get_key()
        s2_key = s2.get_key()
        s1, s2 = s1.get_coordinates(), s2.get_coordinates()
        x_s1, y_s1 = s1[0], s1[1]
        x_s2, y_s2 = s2[0], s2[1]
        # Move diagonally. Highways cannot be enforced
        if x_s1 != x_s2 and y_s1 != y_s2:
            if s1_key == '1' and s2_key == '1':
                c = np.sqrt(2)
            elif s1_key == '2' and s2_key == '2':
                c = np.sqrt(8)
            elif (s1_key == '1' and s2_key == '2') or (s1_key == '2' and s2_key == '1'):
                c = (np.sqrt(2) + np.sqrt(8)) / 2
            elif s1_key == 'a' and s2_key == 'a':
                c = np.sqrt(2)
            elif s1_key == 'b' and s2_key == 'b':
                c = np.sqrt(8)
            elif (s1_key == 'a' and s2_key == 'b') or (s1_key == 'b' and s2_key == 'a'):
                c = (np.sqrt(2) + np.sqrt(8)) / 2
            elif (s1_key == '1' and s2_key == 'a') or (s1_key == 'a' and s2_key == '1'):
                c = np.sqrt(2)
            elif (s1_key == '2' and s2_key == 'b') or (s1_key == 'b' and s2_key == '2'):
                c = np.sqrt(8)
            elif (s1_key == '1' and s2_key == 'b') or (s1_key == 'b' and s2_key == '1'):
                c = (np.sqrt(2) + np.sqrt(8)) / 2
            elif (s1_key == '2' and s2_key == 'a') or (s1_key == 'a' and s2_key == '2'):
                c = (np.sqrt(2) + np.sqrt(8)) / 2
        # Move horizontally or vertically. Highways can be enforced.
        else:
            if s1_key == '1' and s2_key == '1':
                c = 1.0
            elif s1_key == '2' and s2_key == '2':
                c = 2.0
            elif (s1_key == '1' and s2_key == '2') or (s1_key == '2' and s2_key == '1'):
                c = 1.5
            elif s1_key == 'a' and s2_key == 'a':
                c = 0.25
            elif s1_key == 'b' and s2_key == 'b':
                c = 0.5
            elif (s1_key == 'a' and s2_key == 'b') or (s1_key == 'b' and s2_key == 'a'):
                c = 0.375
            elif (s1_key == '1' and s2_key == 'a') or (s1_key == 'a' and s2_key == '1'):
                c = 1.0
            elif (s1_key == '2' and s2_key == 'b') or (s1_key == 'b' and s2_key == '2'):
                c = 2.0
            elif (s1_key == '1' and s2_key == 'b') or (s1_key == 'b' and s2_key == '1'):
                c = 1.5
            elif (s1_key == '2' and s2_key == 'a') or (s1_key == 'a' and s2_key == '2'):
                c = 1.5
        return c

    # start and goal are illustrated by tuples
    def find_path(self, input_matrix, start, goal):
        start_x, start_y = start[0], start[1]
        goal_x, goal_y = goal[0], goal[1]
        self.Matrix[start_x][start_y].g = 0
        self.Matrix[start_x][start_y].h = self.compute_h_value(self.Matrix[start_x][start_y], self.Matrix[goal_x][goal_y])
        self.Matrix[start_x][start_y].f = self.Matrix[start_x][start_y].g + self.Matrix[start_x][start_y].h

        heapq.heappush(self.fringe, (self.Matrix[start_x][start_y].f, self.Matrix[start_x][start_y]))
        # self.fringe_xy.append((start_x, start_y))
        self.Matrix[start_x][start_y].parent = self.Matrix[start_x][start_y]

        while len(self.fringe):
            best_f_value, best = heapq.heappop(self.fringe)
            best_xy = best.get_coordinates()

            if best_xy == (goal_x, goal_y):
                self.closed.append((goal_x, goal_y))
                self.path.append(best)
                while (best.parent.x, best.parent.y) != (start_x, start_y):
                    self.path.append(best.parent)
                    best = best.parent
                self.path.append(self.Matrix[start_x][start_y])
                return self.path, self.Matrix[goal_x][goal_y].g, len(self.closed)

            self.closed.append(best_xy)

            successors = self.succ(best)
            for i in successors:
                if i not in self.closed and input_matrix[i[0], i[1]] != '0':
                    if self.Matrix[i[0]][i[1]].f == 0:
                        self.Matrix[i[0]][i[1]].g = float('inf')
                        if best.g + self.cost(best, self.Matrix[i[0]][i[1]]) < self.Matrix[i[0]][i[1]].g:
                            self.Matrix[i[0]][i[1]].g = best.g + self.cost(best, self.Matrix[i[0]][i[1]])
                            self.Matrix[i[0]][i[1]].h = self.compute_h_value(self.Matrix[i[0]][i[1]], self.Matrix[goal_x][goal_y])
                            self.Matrix[i[0]][i[1]].f = self.Matrix[i[0]][i[1]].g + self.Matrix[i[0]][i[1]].h
                            self.Matrix[i[0]][i[1]].parent = best
                            heapq.heappush(self.fringe, (self.Matrix[i[0]][i[1]].f, self.Matrix[i[0]][i[1]]))
                    else:
                        if best.g + self.cost(best, self.Matrix[i[0]][i[1]]) < self.Matrix[i[0]][i[1]].g:
                            self.fringe.remove((self.Matrix[i[0]][i[1]].f, self.Matrix[i[0]][i[1]]))
                            self.Matrix[i[0]][i[1]].g = best.g + self.cost(best, self.Matrix[i[0]][i[1]])
                            self.Matrix[i[0]][i[1]].h = self.compute_h_value(self.Matrix[i[0]][i[1]], self.Matrix[goal_x][goal_y])
                            self.Matrix[i[0]][i[1]].f = self.Matrix[i[0]][i[1]].g + self.Matrix[i[0]][i[1]].h
                            self.Matrix[i[0]][i[1]].parent = best
                            heapq.heappush(self.fringe, (self.Matrix[i[0]][i[1]].f, self.Matrix[i[0]][i[1]]))
        return None, None, None


class UniformCostSearch(HeuristicFamily):
    def __init__(self):
        super(UniformCostSearch, self).__init__()

    def compute_h_value(self, CurrentNode, GoalNode):
        return 0


class AStarSearch(HeuristicFamily):
    def __init__(self):
        super(AStarSearch, self).__init__()

    def compute_h_value(self, CurrentNode, GoalNode):
        # Manhattan Distance
        if self.heuristic == '0':
            return (abs(CurrentNode.x - GoalNode.x) + abs(CurrentNode.y - GoalNode.y))
        # Euclidean Distance
        elif self.heuristic == '1':
            return (math.sqrt(math.pow(CurrentNode.x - GoalNode.x, 2) + math.pow(CurrentNode.y - GoalNode.y, 2)))
        # Chebyshev Distance
        elif self.heuristic == '2':
            return (max(abs(CurrentNode.x - GoalNode.x), abs(CurrentNode.y - GoalNode.y)))
        # Octile Distance
        elif self.heuristic == '3':
            return (abs(CurrentNode.x - GoalNode.x)+ abs(CurrentNode.y - GoalNode.y))\
                   + (np.sqrt(2) - 2) * min(abs(CurrentNode.x - GoalNode.x), abs(CurrentNode.y - GoalNode.y))
        # Euclidean Distance / 4
        elif self.heuristic == '4':
            return (math.sqrt(math.pow(CurrentNode.x - GoalNode.x, 2) + math.pow(CurrentNode.y - GoalNode.y, 2)))/4
        # Euclidean Distance, squared
        elif self.heuristic == '5':
            return (math.pow(CurrentNode.x - GoalNode.x, 2) + math.pow(CurrentNode.y - GoalNode.y, 2))


class WeightedAStarSearch(HeuristicFamily):
    def __init__(self):
        super(WeightedAStarSearch, self).__init__()

    def compute_h_value(self, CurrentNode, GoalNode):
        # Manhattan Distance
        if self.heuristic == '0':
            return (abs(CurrentNode.x - GoalNode.x) + abs(CurrentNode.y - GoalNode.y)) * self.weight
        # Euclidean Distance
        elif self.heuristic == '1':
            return (math.sqrt(math.pow(CurrentNode.x - GoalNode.x, 2) + math.pow(CurrentNode.y - GoalNode.y, 2))) * self.weight
        # Chebyshev Distance
        elif self.heuristic == '2':
            return (max(abs(CurrentNode.x - GoalNode.x), abs(CurrentNode.y - GoalNode.y))) * self.weight
        # Octile Distance
        elif self.heuristic == '3':
            return ((abs(CurrentNode.x - GoalNode.x) + abs(CurrentNode.y - GoalNode.y)) \
                   + (np.sqrt(2) - 2) * min(abs(CurrentNode.x - GoalNode.x), abs(CurrentNode.y - GoalNode.y))) * self.weight
        # Euclidean Distance / 4
        elif self.heuristic == '4':
            return ((math.sqrt(math.pow(CurrentNode.x - GoalNode.x, 2) + math.pow(CurrentNode.y - GoalNode.y, 2))) / 4) * self.weight
        # Euclidean Distance, squared
        elif self.heuristic == '5':
            return (math.pow(CurrentNode.x - GoalNode.x, 2) + math.pow(CurrentNode.y - GoalNode.y, 2)) * self.weight


class SequentialAStarSearch(HeuristicFamily):
    def __init__(self):
        super(SequentialAStarSearch, self).__init__()

    def LocalCost(self, x_s1, y_s1, x_s2, y_s2):
        c = 0
        s1_key = self.Matrix[x_s1][y_s1]
        s2_key = self.Matrix[x_s2][y_s2]
        # Move diagonally. Highways cannot be enforced
        if x_s1 != x_s2 and y_s1 != y_s2:
            if s1_key == '1' and s2_key == '1':
                c = np.sqrt(2)
            elif s1_key == '2' and s2_key == '2':
                c = np.sqrt(8)
            elif (s1_key == '1' and s2_key == '2') or (s1_key == '2' and s2_key == '1'):
                c = (np.sqrt(2) + np.sqrt(8)) / 2
            elif s1_key == 'a' and s2_key == 'a':
                c = np.sqrt(2)
            elif s1_key == 'b' and s2_key == 'b':
                c = np.sqrt(8)
            elif (s1_key == 'a' and s2_key == 'b') or (s1_key == 'b' and s2_key == 'a'):
                c = (np.sqrt(2) + np.sqrt(8)) / 2
            elif (s1_key == '1' and s2_key == 'a') or (s1_key == 'a' and s2_key == '1'):
                c = np.sqrt(2)
            elif (s1_key == '2' and s2_key == 'b') or (s1_key == 'b' and s2_key == '2'):
                c = np.sqrt(8)
            elif (s1_key == '1' and s2_key == 'b') or (s1_key == 'b' and s2_key == '1'):
                c = (np.sqrt(2) + np.sqrt(8)) / 2
            elif (s1_key == '2' and s2_key == 'a') or (s1_key == 'a' and s2_key == '2'):
                c = (np.sqrt(2) + np.sqrt(8)) / 2
        # Move horizontally or vertically. Highways can be enforced.
        else:
            if s1_key == '1' and s2_key == '1':
                c = 1.0
            elif s1_key == '2' and s2_key == '2':
                c = 2.0
            elif (s1_key == '1' and s2_key == '2') or (s1_key == '2' and s2_key == '1'):
                c = 1.5
            elif s1_key == 'a' and s2_key == 'a':
                c = 0.25
            elif s1_key == 'b' and s2_key == 'b':
                c = 0.5
            elif (s1_key == 'a' and s2_key == 'b') or (s1_key == 'b' and s2_key == 'a'):
                c = 0.375
            elif (s1_key == '1' and s2_key == 'a') or (s1_key == 'a' and s2_key == '1'):
                c = 1.0
            elif (s1_key == '2' and s2_key == 'b') or (s1_key == 'b' and s2_key == '2'):
                c = 2.0
            elif (s1_key == '1' and s2_key == 'b') or (s1_key == 'b' and s2_key == '1'):
                c = 1.5
            elif (s1_key == '2' and s2_key == 'a') or (s1_key == 'a' and s2_key == '2'):
                c = 1.5
        return c

    def Scompute_hS_value(self, Cx,Cy,Gx,Gy, num):
        # Manhattan Distance
        if num == 1:
            return (abs(Cx - Gx) + abs(Cy - Gy)) * 2.0
        # Euclidean Distance
        elif num == 4:
            return (math.sqrt(math.pow(Cx - Gx, 2) + math.pow(Cy -Gy, 2))) * 2.0
        # Chebyshev Distance
        elif num == 2:
            return (max(abs(Cx - Gx), abs(Cy - Gy))) * 2.0
        # Octile Distance
        elif num == 3:
            return (abs(Cx - Gx)+ abs(Cy - Gy))+ (np.sqrt(2) - 2) * min(abs(Cx - Gx), abs(Cy - Gy)) * 2.0
        # Euclidean Distance / 4
        elif num == 0:
            return ((math.sqrt(math.pow(Cx - Gx, 2) + math.pow(Cy -Gy, 2))) / 4) * 2.0
        # Euclidean Distance, squared
        elif num == 5:
            return (math.pow(Cx - Gx, 2) + math.pow(Cy - Gy, 2)) * 2.0


    # start and goal are illustrated by tuples
    def find_path(self, input_matrix, start, goal):

        def createListofvalue():
            matrixlist = []
            for num in range (1,5):
                matrix = []
                for x in range(0, r):
                    row = []
                    for y in range(0, c):
                        row.append(0)
                    matrix.append(row)
                matrixlist.append(matrix)
            return matrixlist
        def createListofparent(xcoordinates, ycoordinates):
            matrixlist = []
            for num in range (1,5):
                matrix = []
                for x in range(0, r):
                    row = []
                    for y in range(0, c):
                        if x == xcoordinates and y == ycoordinates:
                            xy = (x, y)
                        else:
                            xy = (0,0)
                        row.append(xy)
                    matrix.append(row)
                matrixlist.append(matrix)
            return matrixlist

        def LocalSucc(x_s,y_s):
            list = []
            if x_s == 0 and y_s == 0:
                list.append((x_s + 1, y_s))
                list.append((x_s, y_s + 1))
                list.append((x_s + 1, y_s + 1))
            elif x_s == 0 and y_s == c - 1:
                list.append((x_s + 1, y_s))
                list.append((x_s, y_s - 1))
                list.append((x_s + 1, y_s - 1))
            elif x_s == r - 1 and y_s == 0:
                list.append((x_s - 1, y_s))
                list.append((x_s, y_s + 1))
                list.append((x_s - 1, y_s + 1))
            elif x_s == r - 1 and y_s == c - 1:
                list.append((x_s - 1, y_s))
                list.append((x_s, y_s - 1))
                list.append((x_s - 1, y_s - 1))
            elif x_s != 0 and x_s != r - 1 and y_s == 0:
                list.append((x_s + 1, y_s))
                list.append((x_s - 1, y_s))
                list.append((x_s, y_s + 1))
                list.append((x_s + 1, y_s + 1))
                list.append((x_s - 1, y_s + 1))
            elif x_s != 0 and x_s != r - 1 and y_s == c - 1:
                list.append((x_s + 1, y_s))
                list.append((x_s - 1, y_s))
                list.append((x_s + 1, y_s - 1))
                list.append((x_s - 1, y_s - 1))
                list.append((x_s, y_s - 1))
            elif x_s == 0 and y_s != 0 and y_s != c - 1:
                list.append((x_s + 1, y_s))
                list.append((x_s + 1, y_s + 1))
                list.append((x_s + 1, y_s - 1))
                list.append((x_s, y_s + 1))
                list.append((x_s, y_s - 1))
            elif x_s == r - 1 and y_s != 0 and y_s != c - 1:
                list.append((x_s - 1, y_s))
                list.append((x_s - 1, y_s + 1))
                list.append((x_s - 1, y_s - 1))
                list.append((x_s, y_s + 1))
                list.append((x_s, y_s - 1))
            else:
                list.append((x_s - 1, y_s))
                list.append((x_s + 1, y_s))
                list.append((x_s, y_s - 1))
                list.append((x_s, y_s + 1))
                list.append((x_s - 1, y_s - 1))
                list.append((x_s - 1, y_s + 1))
                list.append((x_s + 1, y_s - 1))
                list.append((x_s + 1, y_s + 1))
            return list

        start_x, start_y = start[0], start[1]
        goal_x, goal_y = goal[0], goal[1]
        self.Matrix[start_x][start_y].g = 0
        self.Matrix[start_x][start_y].h = self.Scompute_hS_value(start_x, start_y, goal_x,goal_y,0)
        self.Matrix[start_x][start_y].f = self.Matrix[start_x][start_y].g + self.Matrix[start_x][start_y].h
        heapq.heappush(self.fringe, (self.Matrix[start_x][start_y].f, self.Matrix[start_x][start_y]))
        self.Matrix[start_x][start_y].parent = self.Matrix[start_x][start_y]
        self.Matrix[goal_x][goal_y].g = 100000
        # For other four heuristic data structure
        G_MatrixList = createListofvalue()
        H_MatrixList = createListofvalue()
        F_MatrixList = createListofvalue()
        parent_MatrixList = createListofparent(start_x, start_y)
        openMatrixList = []
        closedMatrixList = []
        for iteration in range (1,4):
            realnumber = iteration-1
            currentmatrix_gvalue = G_MatrixList[realnumber]
            currentmatrix_hvalue = H_MatrixList[realnumber]
            currentmatrix_fvalue = F_MatrixList[realnumber]
            currentmatrix_parent = parent_MatrixList[realnumber]
            currentclosed = []
            currentfringe = []
            currentmatrix_gvalue[start_x][start_y] = 0
            currentmatrix_hvalue[start_x][start_y] = self.Scompute_hS_value(start_x, start_y, goal_x,goal_y, realnumber)
            currentmatrix_fvalue[start_x][start_y] = currentmatrix_gvalue[start_x][start_y] + currentmatrix_hvalue[start_x][start_y]
            currentmatrix_parent[start_x][start_y] = (start_x, start_y)
            currentmatrix_gvalue[goal_x][goal_y] = 100000
            G_MatrixList[realnumber] = currentmatrix_gvalue
            H_MatrixList[realnumber] = currentmatrix_hvalue
            F_MatrixList[realnumber] = currentmatrix_fvalue
            heapq.heappush(currentfringe,(currentmatrix_fvalue[start_x][start_y],(start_x, start_y)))
            parent_MatrixList[realnumber] = currentmatrix_parent
            closedMatrixList.append(currentclosed)
            openMatrixList.append(currentfringe)
        # For other four heuristic data structure
        CURRENTNODE_f, CURRENTNODE = heapq.heappop(self.fringe)
        heapq.heappush(self.fringe, (CURRENTNODE_f, CURRENTNODE))
        while len(self.fringe):
            for NUM in range(1, 4):
                realnumber = NUM - 1
                OPENImatrix_gvalue = G_MatrixList[realnumber]
                OPENImatrix_hvalue = H_MatrixList[realnumber]
                OPENImatrix_fvalue = F_MatrixList[realnumber]
                OPENImatrix_parent = parent_MatrixList[realnumber]
                OPENIclosed = closedMatrixList[realnumber]
                OPENIfringe = openMatrixList[realnumber]
                iCell_f_value, iCell_xy = heapq.heappop(OPENIfringe)
                heapq.heappush(OPENIfringe, (iCell_f_value, iCell_xy))
                if  iCell_f_value <= 1.2 * CURRENTNODE_f:
                    if currentmatrix_gvalue[goal_x][goal_y] <= iCell_f_value:
                        if currentmatrix_gvalue[goal_x][goal_y] < 100000:
                            # Change self.Matrix to the better one and return:
                            self.replace(OPENImatrix_gvalue, OPENImatrix_hvalue, OPENImatrix_fvalue, OPENImatrix_parent)
                            temp_xy = (goal_x, goal_y)
                            OPENIclosed.append(temp_xy)
                            best = self.Matrix[goal_x][goal_y]
                            self.path.append(best)
                            while (best.x, best.y) != (start_x, start_y):
                                self.path.append(best.parent)
                                best = best.parent
                            self.path.append(self.Matrix[start_x][start_y])
                            return self.path, currentmatrix_gvalue[goal_x][goal_y], len(self.closed)
                    else:
                        iCell_f_value, iCell_xy = heapq.heappop(OPENIfringe)
                        successors = LocalSucc(iCell_xy[0], iCell_xy[1])
                        for i in successors:
                            if input_matrix[i[0], i[1]] != '0':
                                if  OPENImatrix_gvalue[i[0]][i[1]] == 0:
                                    OPENImatrix_gvalue[i[0]][i[1]] = 100000
                                    OPENImatrix_gvalue[i[0]][i[1]] = (-1,-1)
                                if currentmatrix_gvalue[iCell_xy[0]][iCell_xy[1]] + self.LocalCost(iCell_xy[0], iCell_xy[1], i[0],i[1]) < OPENImatrix_gvalue[i[0]][i[1]]:
                                    OPENImatrix_gvalue[i[0]][i[1]] = OPENImatrix_gvalue[iCell_xy[0]][iCell_xy[1]] + self.LocalCost(iCell_xy[0], iCell_xy[1], i[0],i[1] )
                                    OPENImatrix_hvalue[i[0]][i[1]] = self.Scompute_hS_value(i[0],i[1], goal_x,goal_y, NUM)
                                    OPENImatrix_fvalue[i[0]][i[1]] = OPENImatrix_gvalue[i[0]][i[1]] + OPENImatrix_hvalue[i[0]][i[1]]
                                    OPENImatrix_parent[i[0]][i[1]] = iCell_xy
                                    if i not in OPENIclosed:
                                        heapq.heappush(OPENIfringe,(OPENImatrix_fvalue[i[0]][i[1]], i))
                                        OPENIclosed.append((goal_x, goal_y))
                else:
                    if self.Matrix[goal_x][goal_y].g <= CURRENTNODE_f:
                        if self.Matrix[goal_x][goal_y].g < 100000:
                            self.closed.append((goal_x, goal_y))
                            CURRENTNODE = self.Matrix[goal_x][goal_y]
                            self.path.append(CURRENTNODE)
                            while (CURRENTNODE.x, CURRENTNODE.y) != (start_x, start_y):
                                self.path.append(CURRENTNODE)
                                CURRENTNODE = CURRENTNODE.parent
                            self.path.append(self.Matrix[start_x][start_y])
                            return self.path, self.Matrix[goal_x][goal_y].g, len(self.closed)
                    else:
                        CURRENTNODE_f,CURRENTNODE = heapq.heappop(self.fringe)
                        self.closed.append((CURRENTNODE.x, CURRENTNODE.y))
                        successors = self.succ(CURRENTNODE)
                        for i in successors:
                            if i not in self.closed and input_matrix[i[0], i[1]] != '0':
                                if  self.Matrix[i[0]][i[1]].g == 0:
                                    self.Matrix[i[0]][i[1]].g = 100000
                                    self.Matrix[i[0]][i[1]].parent = None
                                if CURRENTNODE.g + self.cost(CURRENTNODE, self.Matrix[i[0]][i[1]]) < self.Matrix[i[0]][i[1]].g:
                                    self.Matrix[i[0]][i[1]].g = CURRENTNODE.g + self.cost(CURRENTNODE, self.Matrix[i[0]][i[1]])
                                    self.Matrix[i[0]][i[1]].h = self.Scompute_hS_value(i[0],i[1], goal_x,goal_y, 0)
                                    self.Matrix[i[0]][i[1]].f = self.Matrix[i[0]][i[1]].g + self.Matrix[i[0]][i[1]].h
                                    self.Matrix[i[0]][i[1]].parent = CURRENTNODE
                                    heapq.heappush(self.fringe, (self.Matrix[i[0]][i[1]].f, self.Matrix[i[0]][i[1]]))
                G_MatrixList[realnumber] = OPENImatrix_gvalue
                H_MatrixList[realnumber] = OPENImatrix_hvalue
                F_MatrixList[realnumber] = OPENImatrix_fvalue
                parent_MatrixList[realnumber] = OPENImatrix_parent
                closedMatrixList[realnumber] = OPENIclosed
                openMatrixList[realnumber] = OPENIfringe
        return None, None, None
