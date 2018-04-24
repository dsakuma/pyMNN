from multiprocessing.pool import Pool
from skimage.measure import compare_ssim

import math
import memory
import numpy as np
import os
import pickle
import test_util

import matplotlib.pyplot as plt

def test_accuracy(imsize, nsamples, amounts, im_repeats):
    data = test_util.InputData(imsize)
    images = data.load_n_random(nsamples)

    builder = memory.AutoMemoryBuilder((imsize, imsize), nsamples)
    for im in images:
        builder.append(im)

    mem = builder.build()
    sims = np.empty((len(amounts), nsamples, im_repeats))

    for k, im in enumerate(images):
        print('image', k)
        for j, amount in enumerate(amounts):
            for r in np.arange(im_repeats):
                inp = test_util.erosive_bool_noise(im, amount)
                inp = mem.recall(inp, how='w')

                sim = compare_ssim(inp, im)
                sims[j, k, r] = sim

    return np.mean(sims, axis=(1,2))


if __name__ == '__main__':
    nsamples_a = [10, 40, 80]
    nsamples_repeats = 10

    size = 32
    im_repeats = 10

    amounts = np.arange(0, 1, step=0.05)
    pool = Pool()

    nsamples_in = nsamples_a * nsamples_repeats
    x = [(size, nsamples, amounts, im_repeats) for nsamples in nsamples_in]
    outs = pool.starmap(test_accuracy, x)

    obj = dict()
    obj['amounts'] = amounts
    obj['nsamples'] = nsamples_a
    obj['nsamples_repeats'] = nsamples_repeats
    obj['size'] = 32
    obj['im_repeats'] = im_repeats
    obj['sims'] = outs

    os.makedirs('data_out/accuracy/', exist_ok=True)
    with open('data_out/accuracy/acc.bin', mode='wb') as f:
        pickle.dump(file=f, obj=obj)
