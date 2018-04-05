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
    morph_mem_m = cuda.device_array(mem_shape)
    morph_mem_w = cuda.device_array(mem_shape)

    mnn.mat_morph_mul_max_tt_minus(x_memory, x_memory, morph_mem_m)
    mnn.mat_morph_mul_min_tt_minus(x_memory, x_memory, morph_mem_w)

    # %%
    test_x = cuda.to_device(images[1].reshape(x_shape))
    out_x = cuda.device_array(xt_shape)

    mnn.mat_morph_mul_max_it_plus(morph_mem_w, test_x.T, out_x)

    # %%
    hout_x = out_x.copy_to_host().T.reshape(im_shape)
    
    plt.imshow(hout_x)
    plt.show()
