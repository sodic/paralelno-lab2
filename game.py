from os import system
from collections import namedtuple
from typing import List, Optional

import numpy as np

from mpi4py import MPI
from itertools import groupby

EXIT = 'exit'
DONE = 'done'
READY = 'ready'
WORK = 'work'

NUMBER_OF_COLS = 7
MAX_DEPTH = 4

EMPTY = 0
COMPUTER = 1
PLAYER = 2

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
number_of_processes = comm.Get_size()

Position = namedtuple("Position", "row col")
Table = np.array
State = namedtuple("State", "position last_player table")

VALUES_FOR_WINNER = {
    COMPUTER: 1,
    PLAYER: -1,
    None: 0
}


def opponent(player):
    return player % 2 + 1


def flatten(l):
    return [x for sublist in l for x in sublist]


def four_in_a_row(player, row):
    return any(sum(1 for _ in it) >= 4 for value, it
               in groupby(row) if value == player)


def did_win_vertical(player: int, table: np.array, last_col: int):
    return four_in_a_row(player, table[:, last_col])


def did_win_horizontal(player: int, table: np.array, last_row: int):
    return four_in_a_row(player, table[last_row, :])


def elements_for_indices(table, indices):
    rows, cols = table.shape
    return (table[x][y] for x, y in indices
            if 0 <= x < rows and 0 <= y < cols)


def did_win_diagonal(player: int, table: np.array, last_row: int, last_col: int):
    d1 = zip(range(last_row - 3, last_row + 4), range(last_col + 3, last_col - 4, -1))
    d2 = zip(range(last_row - 3, last_row + 4), range(last_col - 3, last_col + 4))

    return four_in_a_row(player, elements_for_indices(table, d1)) \
           or four_in_a_row(player, elements_for_indices(table, d2))


def did_win(player: int, table: np.array, last_row, last_col):
    return did_win_vertical(player, table, last_col) \
           or did_win_horizontal(player, table, last_row) \
           or did_win_diagonal(player, table, last_row, last_col)


def winner(state: State) -> Optional[int]:
    if did_win(state.last_player, state.table, *state.position):
        return state.last_player
    else:
        return None


def print_table(table: np.array):
    system("clear")
    symbols = {
        COMPUTER: "\x1b[1m\x1b[31mO\x1b[0m",
        PLAYER: "\x1b[1m\x1b[32mO\x1b[0m",
        EMPTY: '.',
    }
    rows, cols = table.shape

    print("".join("{0:<3d}".format(i) for i in range(cols)))
    for i in range(rows - 1, -1, -1):
        print("  ".join(symbols[x] for x in table[i, :]))
    print("".join("{0:<3d}".format(i) for i in range(cols)))


def make_move(player, table, col):
    rows, cols = table.shape
    for i in range(rows):
        if table[i][col] == EMPTY:
            row = i
            break
    else:
        table = np.append(table, [[0] * cols], axis=0)
        row = rows

    table = np.copy(table)

    table[row][col] = player
    return State((row, col), player, table)


def move_and_check_victory(state: State, col):
    state = make_move(opponent(state.last_player), state.table, col)
    return state, did_win(state.last_player, state.table, *state.position)


def play_game():
    state = initial_state(NUMBER_OF_COLS)
    print_table(state.table)
    while True:
        print()
        col = int(input("U koji stupac ubacujes zeton? "))
        state, won = move_and_check_victory(state, col)
        if won:
            print("Pobijedio si")
            break
        print_table(state.table)

        print()
        print("Razmisljam...")
        state, won = move_and_check_victory(state, decide_move(state))
        if won:
            print("Pusiona")
            break
        print_table(state.table)

    print_table(state.table)
    comm.bcast(EXIT)


def successors(state: State) -> List[State]:
    table = state.table
    turn = opponent(state.last_player)
    return [make_move(turn, table, col) for col in range(table.shape[1])]


def initial_state(cols) -> State:
    table = np.array([[0 for _ in range(cols)] for _ in range(10)])
    location = None, None
    return State(location, COMPUTER, table)


def decide_move(start_state: State):
    first_level = successors(start_state)
    second_level = flatten(successors(state) for state in first_level)
    second_level_tasks = list(enumerate(second_level))

    decision_map = {col_index: [] for col_index in range(NUMBER_OF_COLS)}

    comm.bcast(WORK)
    awake = number_of_processes - 1

    if number_of_processes == 1:
        for task_index, state in second_level_tasks:
            value = find_value_of_state(state, MAX_DEPTH)
            decision_map[task_index // NUMBER_OF_COLS].append(value)

    while awake:
        status = MPI.Status()
        message = comm.recv(source=MPI.ANY_SOURCE, status=status)

        if message == READY:
            if second_level_tasks:
                comm.send(second_level_tasks.pop(), dest=status.source)
            else:
                comm.send(DONE, dest=status.source)
                awake -= 1

        else:
            task_index, value = message
            decision_map[task_index // NUMBER_OF_COLS].append(value)

    results = [state_value(first_level[col_index], decision_map[col_index])
               for col_index in range(NUMBER_OF_COLS)]

    return results.index(max(results))


def worker():
    while True:
        message = comm.bcast(None, root=0)
        if message == EXIT:
            return
        while True:
            comm.send(READY, dest=0)
            message = comm.recv(source=0)
            if message == DONE:
                break

            index, state = message
            comm.send((index, find_value_of_state(state, MAX_DEPTH)), dest=0)


def state_value_from_children(turn, successor_values):
    if all(v == 1 for v in successor_values):
        return 1
    if all(v == -1 for v in successor_values):
        return -1
    if 1 in successor_values and turn == COMPUTER:
        return 1
    if -1 in successor_values and turn == PLAYER:
        return -1

    return sum(successor_values) / NUMBER_OF_COLS


def state_value(state, successor_values):
    victor = winner(state)
    if victor:
        return VALUES_FOR_WINNER[victor]
    else:
        return state_value_from_children(opponent(state.last_player),
                                         successor_values)


def find_value_of_state(state, depth):
    if not depth or winner(state):
        return VALUES_FOR_WINNER[winner(state)]

    successor_values = [find_value_of_state(s, depth - 1) for s in successors(state)]
    turn = opponent(state.last_player)

    return state_value_from_children(turn, successor_values)


def main():
    if rank == 0:
        play_game()
    else:
        worker()


if __name__ == '__main__':
    main()
