# %%
from numba import stencil, cuda
import numpy as np
import math
import mnn
import memory as mem


# %%
def peek(name, dev_arr):
    print('contents of array', name)
    print(dev_arr.copy_to_host())


# %%
if __name__ == '__main__':
    print('Feeding memory')
    builder = mem.AutoMemoryBuilder(x_shape=(1, 4), n_samples=3)
    print('1')
    builder.append([4, 2, 5, 6])
    print('2')
    builder.append([2, 3, 8, 4])
    print('3')
    builder.append([2, 5, 1, 4])

    print('Building memory')
    mem = builder.build()

    print('Recall')

    def recall(mem, x, how='w'):
        print(mem.recall(x, how))

    recall(mem, [4, 2, 5, 6])
    recall(mem, [0, 2, 5, 6])

    recall(mem, [4, 2, 5, 6], how='m')
    recall(mem, [5, 2, 7, 8], how='m')
