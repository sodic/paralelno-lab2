from sys import stdout

from mpi4py import MPI
from time import sleep
from random import randint

REQUEST = 'request'
RESPONSE = 'response'

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
number_of_processes = comm.Get_size()

left = (rank + 1) % number_of_processes
right = (rank + number_of_processes - 1) % number_of_processes
neighbours = [left, right]

MISSING = "MISSING"
DIRTY = "DIRTY"
CLEAN = "CLEAN"

forks = [
    MISSING,
    MISSING
]
pending_indices = []


def main():
    assign_forks()

    while True:
        think()
        get_forks()
        eat()
        answer_pending()


def assign_forks():
    for index, neighbour in enumerate(neighbours):
        if rank <= neighbour:
            forks[index] = DIRTY


def think():
    signed_print("Thinking")
    for i in range(randint(1, 3) * 100):
        check_messages()
        sleep(0.05)


def get_fork(index):
    if forks[index] is not MISSING:
        return

    neighbour = neighbours[index]

    signed_print(f"Waiting for {neighbour}'s fork. ")
    comm.send(REQUEST, dest=neighbour)
    while forks[index] is MISSING:
        check_messages()


def get_forks():
    signed_print(f"I'm hungry. I have {total_forks()} forks.")

    while total_forks() < len(neighbours):
        for index, _ in enumerate(neighbours):
            get_fork(index)


def eat():
    signed_print("\x1b[1m\x1b[31mI'm eating.\x1b[0m")
    sleep(randint(4, 9))
    signed_print("\x1b[1m\x1b[32mI'm done.\x1b[0m")
    sleep(1)

    for index, _ in enumerate(neighbours):
        forks[index] = DIRTY


def answer_pending():
    while pending_indices:
        index = pending_indices.pop()
        forks[index] = MISSING
        comm.send(RESPONSE, dest=neighbours[index])


def check_messages():
    if not comm.iprobe(source=MPI.ANY_SOURCE):
        return

    message, source = get_message()
    if is_request(message):
        process_request(source)
    elif is_response(message):
        process_response(source)


def get_message():
    status = MPI.Status()
    message = comm.recv(source=MPI.ANY_SOURCE, status=status)
    return message, status.source


def is_request(message):
    return message == REQUEST


def is_response(message):
    return message == RESPONSE


def process_request(source):
    for index in indices(source):
        if forks[index] is DIRTY:
            forks[index] = MISSING
            comm.send(RESPONSE, dest=source)

        elif forks[index] is CLEAN:
            pending_indices.append(index)

        else:
            raise ValueError("Problem")


def process_response(source):
    for index in indices(source):
        if forks[index] is MISSING:
            forks[index] = CLEAN


def indices(neighbour):
    return (i for i, n in enumerate(neighbours) if n == neighbour)


def total_forks():
    return count_forks(lambda state: state is not MISSING)


def dirty_forks():
    return count_forks(lambda state: state is DIRTY)


def count_forks(predicate):
    return sum(1 for state in forks if predicate(state))


def signed_print(*args, **kwargs):
    print("\t" * rank, f"{rank}:".upper(), *args, **kwargs)
    stdout.flush()


if __name__ == '__main__':
    main()
