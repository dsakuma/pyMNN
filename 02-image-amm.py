# %%
from numba import stencil, cuda
import numpy as np
import math

import mnn
import memory as mem

from PIL import Image

import matplotlib.pyplot as plt

# %%
def load_image(path):
    im = Image.open(path)
    bands = im.getbands()

    assert len(bands) <= 2

    return np.reshape(np.array(im.getdata(0)) / 255, (32, 32))

def _getuniq(arr, fn):
    s = set(fn(x) for x in arr)
    assert len(s) == 1
    return list(s)[0]


letters = ['A', 'X', '7']


if __name__ == '__main__':
    print('Loading images')
    images = [load_image('data/{0}.png'.format(l)) for l in letters]

    imcount = len(letters)
    imshape = _getuniq(images, np.shape)

    # %%
    print('Feeding memory')
    builder = mem.AutoMemoryBuilder(imshape, imcount)
    for im in images:
        builder.append(im)

    # %%
    print('Building memory')
    memory = builder.build()

    # %%
    print('Preparing samples')
    test_a = images[0]
    test_x = images[1]

    samples_a = []
    samples_x = []
    n_tests = 10

    for i in range(n_tests):
        prob = (i / n_tests, 1 - i / n_tests)
        noise = np.random.choice([True, False], (32, 32), p=prob)

        sample_a = np.copy(test_a)
        sample_a[noise] = 1

        sample_x = np.copy(test_x)
        sample_x[noise] = 0

        samples_a.append(sample_a)
        samples_x.append(sample_x)

    # %%
    print('Performing test')
    fig, axes = plt.subplots(nrows=n_tests, ncols=4, sharex=True, sharey=True)
    for i in range(n_tests):
        print(' ', i)

        ax = axes[i]
        samp_a = samples_a[i]
        samp_x = samples_x[i]
        out_a = memory.recall(samp_a, how='m')
        out_x = memory.recall(samp_x, how='w')

        ax[0].imshow(samp_a)
        ax[1].imshow(out_a)

        ax[2].imshow(samp_x)
        ax[3].imshow(out_x)

    # %%
    print('Recall done')
    plt.show()
