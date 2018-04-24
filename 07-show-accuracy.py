from skimage.measure import compare_ssim

import math
import matplotlib.pyplot as plt
import memory
import test_util

cmap = 'gray'

if __name__ == '__main__':
    nsamples_load = 20
    nsamples = 3
    noises = [0.1, 0.5, 0.9]
    size = 32

    data = test_util.InputData(size)
    images = data.load_n_random(nsamples_load)

    builder = memory.AutoMemoryBuilder((size, size), nsamples_load)
    for im in images:
        builder.append(im)

    mem = builder.build()

    nrows = 3
    ncols = nsamples

    while ncols > 10:
        nrows *= 2
        ncols = math.ceil(ncols / 2)

    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    axes = axes.T.reshape(int(ncols * nrows / 3), 3)

    for i, (im, ax, noise) in enumerate(zip(images, axes, noises)):
        inp = test_util.erosive_bool_noise(im, noise)
        res = mem.recall(inp, how='w')

        sim = compare_ssim(res, im)

        if i % 10 == 0:
            ax[0].set_ylabel('char')
            ax[1].set_ylabel('mem_in')
            ax[2].set_ylabel('mem_out')

        ax[0].set_title('sim={:.2}'.format(sim))
        ax[0].imshow(im, cmap=cmap)
        ax[1].imshow(inp, cmap=cmap)
        ax[2].imshow(res, cmap=cmap)

    plt.savefig('data_out/example-accuracy.png', dpi=300)
    plt.show()
