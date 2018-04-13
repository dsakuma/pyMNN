import numpy as np
import os
import pickle
import test_util


if __name__ == '__main__':
    repeats = 10
    out_path = 'data_out/test_perf'
    logger = test_util.get_logger()


    ############################################################################
    print('build-insize')

    n_samples_in = [10, 40, 80]
    sizes = test_util.InputData.sizes()

    logger.debug('Warmup')
    test_util.test_word_length(sizes[0], repeats, n_samples_in, logger)

    insize_path = os.path.join(out_path, 'build-insize')
    os.makedirs(insize_path, exist_ok=True)

    for size in sizes:
        data_path = '{0}/{1}.bin'.format(insize_path, size)
        if os.path.exists(data_path):
            continue

        times = test_util.test_word_length(size, repeats, n_samples_in, logger)
        with open(data_path, mode='wb') as f:
            pickle.dump(times, f)

    ############################################################################
    print('build-nsamples')

    sizes_in = [20, 50, 80]
    nsamples = np.arange(1, 91, step=2)

    logger.debug('Warmup')
    test_util.test_nsamples(nsamples[0], repeats, sizes_in, logger)

    nsamples_path = os.path.join(out_path, 'build-nsamples')
    os.makedirs(nsamples_path, exist_ok=True)

    for ns in nsamples:
        data_path = '{0}/{1}.bin'.format(nsamples_path, ns)
        if os.path.exists(data_path):
            continue

        times = test_util.test_nsamples(ns, repeats, sizes_in, logger)
        with open(data_path, mode='wb') as f:
            pickle.dump(times, f)
