import gc
import numpy as np
import HeuristicFamily as hf
from Tkinter import *
import time
import tkMessageBox
import os
import psutil

row = 120
column = 160


def main(argv):
    memory_usage0 = sum(sys.getsizeof(i) for i in gc.get_objects())
    if os.path.isfile(argv[1]) == False:
        root = Tk()
        root.withdraw()
        tkMessageBox.showerror("Input Error", "File does not exist." +
                               "\nPlease re-enter with the format: map_i_j.txt.\n(i = 0 to 4, j = 0 to 9)")
        return
    f = open(argv[1], "r")
    start_x, start_y = f.readline().replace(" ", "").strip(',').split(",")
    goal_x, goal_y = f.readline().replace(" ", "").strip(',').split(",")
    start_x, start_y, goal_x, goal_y = int(start_x), int(start_y), int(goal_x), int(goal_y)
    centercells = []

    for i in range(0, 8):
        centercells.append(f.readline())
    Matrix = np.chararray((row, column))

    for r in range(0, row):
        line = f.readline().replace(" ", "")
        for c in range(0, column):
            Matrix[r, c] = line[c]

    if argv[2] == '0':
        a = hf.UniformCostSearch()
    elif argv[2] == '1':
        a = hf.AStarSearch()
    elif argv[2] == '2':
        a = hf.WeightedAStarSearch()
    elif argv[2] == '3':
        a = hf.SequentialAStarSearch()
    else:
        root = Tk()
        root.withdraw()
        tkMessageBox.showerror("Input Error", "Search mode is incorrect." +
                               "\nPlease re-enter a number from 0 to 3." +
                               "\n(0 = Uniform Cost Search, 1 = A* Search, 2 = Weighted A* Search, 3 = Sequential A* Search)")
        return

    if argv[2] != '0' and argv[2] != '3':
        if argv[3] != '0' and argv[3] != '1' and argv[3] != '2' and argv[3] != '3' and argv[3] != '4' and argv[3] != '5':
            root = Tk()
            root.withdraw()
            tkMessageBox.showerror("Input Error", "Heuristic type is incorrect." +
                                   "\nPlease re-enter a number from 0 to 4." +
                                   "\n(0 = Manhattan Distance, 1 = Euclidean Distance, " +
                                   "2 = Chebyshev Distance, 3 = Octile Distance, 4= Euclidean Distance / 4, 5 = Euclidean Distance, squared)")
            return

    if argv[2] == '3':
        a.heuristic = 0
    else:
        a.heuristic = argv[3]
    if argv[2] == '2' or argv[3] == '3':
        a.weight = float(argv[4])
    start_time = int(round(time.time() * 1000))
    a.all_to_node(Matrix)
    path, len_path, num_expanded_nodes = a.find_path(Matrix, (start_x, start_y), (goal_x, goal_y))
    time_to_compute = int(round(time.time() * 1000)) - start_time

    if path is None or len_path is None or num_expanded_nodes is None:
        root = Tk()
        root.withdraw()
        tkMessageBox.showerror("Error", "Map is not solvable. \nExit!")
        return


    memory_usage = (sum(sys.getsizeof(i) for i in gc.get_objects())) - memory_usage0
    for i in path:
        Matrix[i.x, i.y] = '3'

    ### GUI ###

    root = Tk()
    root.resizable(0, 0)
    cell_size = 6
    my_gui = Canvas(root, width=cell_size * column, height=cell_size * 121)

    def mouse_clicked_event(event):
        cell_x, cell_y = int(event.y / cell_size), int(event.x / cell_size)
        result = "Cell Coordinates: (" + str(cell_x) + ", " + str(cell_y) + ")" \
                 + "\ng = " + str(a.Matrix[cell_x][cell_y].g) + "\nh = " + str(a.Matrix[cell_x][cell_y].h) \
                 + "\nf = " + str(a.Matrix[cell_x][cell_y].f)
        tkMessageBox.showinfo("Cell Info", result)

    for c in range(0, column * cell_size, cell_size):
        for r in range(0, row * cell_size, cell_size):
            if Matrix[int(r / cell_size)][int(c / cell_size)] == '0':
                cell = my_gui.create_rectangle(c, r, c + cell_size, r + cell_size, fill='black', outline="black")
                my_gui.tag_bind(cell, "<ButtonPress-1>", mouse_clicked_event)
            if Matrix[int(r / cell_size)][int(c / cell_size)] == '1':
                cell = my_gui.create_rectangle(c, r, c + cell_size, r + cell_size, fill='white', outline="black")
                my_gui.tag_bind(cell, "<ButtonPress-1>", mouse_clicked_event)
            if Matrix[int(r / cell_size)][int(c / cell_size)] == '2':
                cell = my_gui.create_rectangle(c, r, c + cell_size, r + cell_size, fill='gray', outline="black")
                my_gui.tag_bind(cell, "<ButtonPress-1>", mouse_clicked_event)
            if Matrix[int(r / cell_size)][int(c / cell_size)] == 'a':
                cell = my_gui.create_rectangle(c, r, c + cell_size, r + cell_size, fill='blue', outline="black")
                my_gui.tag_bind(cell, "<ButtonPress-1>", mouse_clicked_event)
            if Matrix[int(r / cell_size)][int(c / cell_size)] == 'b':
                cell = my_gui.create_rectangle(c, r, c + cell_size, r + cell_size, fill='lightblue', outline="black")
                my_gui.tag_bind(cell, "<ButtonPress-1>", mouse_clicked_event)
            if Matrix[int(r / cell_size)][int(c / cell_size)] == '3':
                if (int(r / cell_size), int(c / cell_size)) == (start_x, start_y):
                    cell = my_gui.create_rectangle(c, r, c + cell_size, r + cell_size, fill='red', outline="yellow")
                else:
                    cell = my_gui.create_rectangle(c, r, c + cell_size, r + cell_size, fill='red', outline="black")
                my_gui.tag_bind(cell, "<ButtonPress-1>", mouse_clicked_event)
    side_info = Label(root, text="Start Coordinates: " + "(" + str(start_x) + ", " + str(start_y) + ")" +
                                 "\t\tGoal Coordiantes: " + "(" + str(goal_x) + ", " + str(goal_y) + ")" +
                                 "\nLength of path: " + str(len_path) +
                                 "\t\tNumber of nodes expanded: " + str(num_expanded_nodes) +
                                 "\nComputed Time in ms: " + str(time_to_compute) +
                                 "\t\tMemory usage in MB: " + str(float(memory_usage) / 1000000), bd=1, relief=SUNKEN,
                      anchor=W)
    side_info.pack(side=TOP, fill=X)
    my_gui.pack()
    root.mainloop()


if __name__ == "__main__":
    main(sys.argv[:])
