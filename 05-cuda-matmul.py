# %%
from numba import stencil, cuda
import numpy as np
import math

import mnn
from PIL import Image

import matplotlib.pyplot as plt

# %%
def load_image(path):
    im = Image.open(path)
    bands = im.getbands()

    assert len(bands) <= 2

    return np.array(im.getdata(0)) / 255

def _getuniq(arr, fn):
    s = set(fn(x) for x in arr)
    assert len(s) == 1
    return list(s)[0]


@cuda.jit
def invert_impl(a, out, m, n):
    i, j = cuda.grid(2)
    if i < m and j < n:
        out[i, j] = -a[i, j]


def invert(a, out):
    m, n = a.shape
    if m * n > 1024:
        threads_per_block = [32, 32]
        blocks_per_grid = [math.ceil(m / 32), math.ceil(n / 32)]
    else:
        threads_per_block = [m, n]
        blocks_per_grid = [1, 1]

    return invert_impl[blocks_per_grid, threads_per_block](a, out, m, n)


letters = ['A', 'X']


if __name__ == '__main__':
    images = [load_image('data/{0}.png'.format(l)) for l in letters]

    imcount = len(letters)
    imsize = _getuniq(images, np.size)

    assert imsize == 32*32

    x_shape = (1, imsize)
    xt_shape = (imsize, 1)

    mem_in_shape = (imcount, imsize)
    im_shape = (32, 32)
    mem_shape = (imsize, imsize)

    # %%
    x_memory = cuda.device_array(mem_in_shape)

    for i, im in enumerate(images):
        cuda.to_device(im, to=x_memory[i])

    # %%
    x_mem_inv = cuda.device_array_like(x_memory)
    invert(x_memory, x_mem_inv)

    morph_mem_m = cuda.device_array(mem_shape)
    morph_mem_w = cuda.device_array(mem_shape)

    mnn.mat_morph_max_mul(x_memory.T, x_mem_inv, morph_mem_m)
    mnn.mat_morph_min_mul(x_memory.T, x_mem_inv, morph_mem_w)

    # %%
    test_x = cuda.to_device(images[1]).reshape(x_shape)
    out_x = cuda.device_array(xt_shape)

    mnn.mat_morph_max_mul(morph_mem_m, test_x.T, out_x)

    check_out = out_x.copy_to_host(out_x).T.reshape(im_shape)

    # %%
    plt.imshow(check_out)
    plt.show()
