# %%
from numba import stencil, cuda
import numpy as np
import math

import mnn
from PIL import Image

import matplotlib.pyplot as plt

# %%
def peek(name, dev_arr):
    print('contents of array', name)
    print(dev_arr.copy_to_host())


class AutoMemoryBuilder:
    def __init__(self, x_shape, n_samples):
        assert len(x_shape) == 2
        self._x_size = x_shape[0] * x_shape[1]
        self._n_samples = n_samples
        self._xmem_shape = (n_samples, self._x_size)

        self._i = 0
        self._dev_mem = cuda.device_array(self._xmem_shape)

    def append(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        if self.is_full():
            raise RuntimeError('Memory is full')

        if np.size(x) != self._x_size:
            raise ValueError('Wrong sample size')

        cuda.to_device(x.reshape(-1), to=self._dev_mem[self._i, :])
        self._i += 1

    def is_full(self):
        assert self._i <= self._n_samples
        return self._i == self._n_samples

    def build(self):
        if not self.is_full():
            raise RuntimeError('Memory is not full')

        mem_shape = (self._x_size, self._x_size)

        morph_mem_m = cuda.device_array(mem_shape)
        morph_mem_w = cuda.device_array(mem_shape)

        xmem = self._dev_mem

        mnn.mat_morph_mul_max_tt_minus(xmem, xmem, morph_mem_m)
        mnn.mat_morph_mul_min_tt_minus(xmem, xmem, morph_mem_w)

        return Memory(morph_mem_m, morph_mem_w, self._x_size)


class Memory:
    def __init__(self, dev_mem_m, dev_mem_w, x_size):
        self._dev_mem_m = dev_mem_m
        self._dev_mem_w = dev_mem_w
        self._x_size = x_size

    def recall(self, x, how):
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        if how == 'm':
            mem = self._dev_mem_m
        elif how == 'w':
            mem = self._dev_mem_w
        else:
            raise ValueError('Expected how=\'m\' or how=\'w\'')

        x_shape = (1, x_size)

        out_x = cuda.device_array(x_shape)
        mnn.mat_morph_mul_max_ii_plus(mem, x.reshape(x_shape), out_x)

        return out_x.copy_to_host()


# %%
x = np.matrix([[4, 2, 5, 6],
               [2, 3, 8, 4],
               [2, 5, 1, 4]])

x_size = 4
x_count = 3

if __name__ == '__main__':
    builder = AutoMemoryBuilder(x_shape=(1, 4), n_samples=3)
    builder.append([4, 2, 5, 6])
    builder.append([2, 3, 8, 4])
    builder.append([2, 5, 1, 4])

    mem = builder.build()
    x = [4, 2, 5, 6]
    y = mem.recall(x, 'w')

    print(y)
    #
    # x_shape = (1, x_size)
    # xt_shape = (x_size, 1)
    #
    # mem_in_shape = (x_count, x_size)
    # mem_shape = (x_size, x_size)
    #
    # # %%
    # x_memory = cuda.to_device(x)
    #
    # # %%
    # morph_mem_m = cuda.device_array(mem_shape)
    # morph_mem_w = cuda.device_array(mem_shape)
    #
    # mnn.mat_morph_mul_max_tt_minus(x_memory, x_memory, morph_mem_m)
    # mnn.mat_morph_mul_min_tt_minus(x_memory, x_memory, morph_mem_w)
    #
    # peek('memory_W', morph_mem_w)
    # peek('memory_M', morph_mem_m)
    #
    # # %%
    # input = x[0].reshape(x_shape)
    #
    # test_x = cuda.to_device(input)
    # peek('test_x', test_x)
    #
    # out_x = cuda.device_array(xt_shape)
    # mnn.mat_morph_mul_max_it_plus(morph_mem_w, test_x.T, out_x)
    #
    # peek('out_x', out_x)
