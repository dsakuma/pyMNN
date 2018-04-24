# %%
from numba import stencil, cuda
import numpy as np
import math

import mnn
import memory as mem
import test_util

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


letters = ['A', 'X']
cmap = 'gray'


if __name__ == '__main__':
    print('Loading images')
    images = [load_image('data-example/{0}.png'.format(l)) for l in letters]

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
    n_tests = 5

    for i in range(n_tests):
        amount = i / n_tests
        samples_a.append(test_util.dilative_bool_noise(test_a, amount))
        samples_x.append(test_util.erosive_bool_noise(test_x, amount))

    # %%
    print('Performing test')
    fig, axes = plt.subplots(nrows=4, ncols=n_tests, sharex=True, sharey=True)
    axes = axes.T
    for i in range(n_tests):
        print(' ', i)

        ax = axes[i]
        samp_a = samples_a[i]
        samp_x = samples_x[i]
        out_a = memory.recall(samp_a, how='m')
        out_x = memory.recall(samp_x, how='w')

        if i == 0:
            ax[0].set_ylabel('in_dilated')
            ax[1].set_ylabel('output')
            ax[2].set_ylabel('in_eroded')
            ax[3].set_ylabel('output')

        ax[0].set_title('dist={0}%'.format(int(i / n_tests * 100)))
        im = ax[0].imshow(samp_a, cmap=cmap)
        ax[1].imshow(out_a, cmap=cmap)
        ax[2].imshow(samp_x, cmap=cmap)
        ax[3].imshow(out_x, cmap=cmap)

    # %%
    print('Recall done')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig('data_out/example-01.png', dpi=300)
    plt.show()
