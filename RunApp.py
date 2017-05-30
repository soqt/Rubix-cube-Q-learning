from tkinter import *
import Cube
# import RubixCube
# import MDP

opt_policy = None

GUI = True

COLOR = Cube.COLOR
ACTIONS = Cube.POSSIBLE_MOVES

heuristics = lambda s: RubixCube.HEURISTICS['h_manhattan'](s)

cube_size = 3
Width = 700
height = 700
block_size = 50
start_x = 30
start_y = 30
BLOCKS = []

master = Tk()
board = Canvas(master, width=Width, height=Width)
board.pack()

def create_cube(cube):
    global BLOCKS
    F = cube.faces['F']
    B = cube.faces['B']
    U = cube.faces['U']
    D = cube.faces['D']
    L = cube.faces['L']
    R = cube.faces['R']


    item_counter = 0
    for h in range(cube_size):
        for r in range(cube_size):
            color = COLOR[U.item(item_counter)]
            item_counter += 1
            BLOCKS.append(board.create_rectangle(start_x+(block_size * cube_size) + r * block_size, start_y + h * block_size,
                                   start_x +(block_size * cube_size) +block_size + r * block_size, start_y + block_size + h * block_size,
                                  fill=color, width=1))

    height_offset = block_size * cube_size
    sides = [L, F, R, B]
    for i in range(4):
        item_counter = 0
        for h in range(cube_size):
            for r in range(cube_size):
                color = COLOR[sides[i].item(item_counter)]
                item_counter += 1
                BLOCKS.append(board.create_rectangle(start_x + i *(block_size * cube_size) + r * block_size, start_y + h * block_size + height_offset,
                                       start_x + i * (block_size * cube_size) + block_size + r * block_size,
                                       start_y + block_size + h * block_size + height_offset,
                                       fill=color, width=1))
    height_offset = 2 * height_offset
    item_counter = 0
    for h in range(cube_size):
        for r in range(cube_size):
            color = COLOR[D.item(item_counter)]
            item_counter += 1
            BLOCKS.append(board.create_rectangle(start_x+(block_size * cube_size) + r * block_size, start_y + h * block_size + height_offset,
                                   start_x +(block_size * cube_size) +block_size + r * block_size, start_y + block_size + h * block_size + height_offset,
                                  fill=color, width=1))


def update_cube(cube):
    F = cube.faces['F']
    B = cube.faces['B']
    U = cube.faces['U']
    D = cube.faces['D']
    L = cube.faces['L']
    R = cube.faces['R']
    list = [U, L, F, R, B, D]
    count = 0
    for i in range(6):
        for j in range(cube_size*cube_size):
            board.itemconfig(BLOCKS[count], fill=COLOR[list[i].item(j)])
            count += 1


def run():
    global opt_policy
    cube = Cube.Cube()
    # cube.generate_random_cube()
    # rubix_cube_MDP = MDP.MDP()
    # rubix_cube_MDP.register_start_state(cube)
    # rubix_cube_MDP.register_actions(ACTIONS)

    if GUI:
        create_cube(cube)
        for i in range(0):
            cube.generate_random_cube(1)
            update_cube(cube)
            master.update()
            # cube_GUI(cube)
        print('done')
        master.mainloop()
    else:
        for i in range(1000):
            cube.generate_random_cube(1)
            print(cube)
        print("done")
run()